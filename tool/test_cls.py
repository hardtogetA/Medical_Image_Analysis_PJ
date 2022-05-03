import os
import random
import time
import cv2
import numpy as np
import logging
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from model import *
from util import dataset
from util import transform, config
from util.util import AverageMeter, intersectionAndUnionGPU

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pspnet50.yaml', help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss

    BatchNorm = nn.BatchNorm2d

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model = eval(args.arch).Model(args)

    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)
    print(args)

    model = nn.parallel.DataParallel(model,[0])

    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))
            raise Exception("'no weight found at '{}'".format(args.weight))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    if args.resized_val:
        val_transform = transform.Compose([
            transform.Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
    else:
        val_transform = transform.Compose([
            transform.test_Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
    val_data = dataset.SemData(data_root=args.data_root, transform=val_transform, mode='test')
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                             num_workers=args.workers, pin_memory=True, sampler=None)
    test(val_loader, model, criterion, args)

def test(val_loader, model, criterion, args):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    front_intersection_meter = 0
    front_union_meter = 0

    if args.manual_seed is not None and args.fix_random_seed_val:
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    clss = []
    predict_clss=[]

    model.eval()
    end = time.time()
    test_num = len(val_loader)
    logger.info(f"test_num{test_num} batch_size{args.batch_size_val}")
    assert test_num % args.batch_size_val == 0
    iter_num = 0
    total_time = 0
    from tqdm.auto import tqdm
    # for e in range(20):
    for i, (input, target, cls, ori_label) in enumerate(tqdm(val_loader)):
        if (iter_num-1) * args.batch_size_val >= test_num:
            break
        iter_num += 1
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        ori_label = ori_label.cuda(non_blocking=True)
        cls = cls.cuda(non_blocking=True)
        start_time = time.time()
        output, predict_cls = model(x=input, y=target, cls=cls)
        total_time = total_time + 1
        model_time.update(time.time() - start_time)

        if args.ori_resize:
            longerside = max(ori_label.size(1), ori_label.size(2))
            backmask = torch.ones(ori_label.size(0), longerside, longerside).cuda()*255
            backmask[0, :ori_label.size(1), :ori_label.size(2)] = ori_label
            target = backmask.clone().long()

        output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
        loss = criterion(output, target)

        n = input.size(0)
        loss = torch.mean(loss)

        output = output.max(1)[1]

        intersection, union, new_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target, new_target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), new_target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)

        front_intersection_meter += intersection[1]
        front_union_meter += union[1]

        # accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        accuracy = (predict_cls.argmax(dim=-1) == cls).float().mean().item()
        predict_clss.append(predict_cls.argmax(dim=-1).detach().cpu().numpy())
        clss.append(cls.detach().cpu().numpy())

        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        accuracy_meter.update(accuracy)
        end = time.time()
        if ((i + 1) % (test_num/100) == 0) and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(iter_num* args.batch_size_val, test_num,
                                                            data_time=data_time,
                                                            batch_time=batch_time,
                                                            loss_meter=loss_meter,
                                                            accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    FBIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    # allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    allAcc = accuracy_meter.avg

    from sklearn.metrics import confusion_matrix
    array = confusion_matrix(np.concatenate(clss),np.concatenate(predict_clss))
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt
    classes = ['covid','non-covid','normal']
    df_cm = pd.DataFrame(array, index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    svm = sn.heatmap(df_cm, annot=True, cmap="Blues", fmt="d")
    figure = svm.get_figure()    
    figure.savefig('svm_conf.png', dpi=400)

    mIoU = front_intersection_meter/(front_union_meter+ 1e-10)
    # logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(mIoU))

    if main_process():
        logger.info('Val result: Acc {:.4f}.'.format(allAcc))
        # for i in range(args.classes):
        #     logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    print('avg inference time: {:.4f}, count: {}'.format(model_time.avg, test_num))

if __name__ == '__main__':
    main()
