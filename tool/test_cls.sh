#!/bin/sh
PARTITION=Segmentation
dataset=$1
model_name=$2_cls
exp_name=$3
exp_dir=exp/${model_name}/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=data/config/${dataset}/${dataset}_${model_name}_${exp_name}.yaml
mkdir -p ${model_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp tool/test_cls.sh tool/test_cls.py ${config} ${exp_dir}
CUDA_VISIBLE_DEVICES=2 python -u -m tool.test_cls --config=${config} 2>&1 | tee ${result_dir}/test-$now.log