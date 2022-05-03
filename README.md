# Medical_Image_Analysis_PJ

### Dataset
- Prepare download the COVID-QU-Ex dataset and put it under data/datasets/COVID-QU-Ex

### Pre-trained models
- Pre-trained backbones for resnet50 and resnet101 can be found in [Google Drive](https://drive.google.com/drive/folders/1dEJL_KSkZZ0nIEy6zwqqb93L4zBDvCV-?usp=sharing)
- Download backbones and put the pth files under `initmodel/` folder
- Checkpoints for segmentation adn classification can be found in [BaiduYun Drive](https://pan.baidu.com/s/1Cbo_DFGjNpG9_CAhdDHWmw?pwd=753d)
- Download checkpoints and put the pth files under this folder

### Test and  Train
+ Specify the path of datasets and pre-trained models in the data/config file
+ Use the following command 
  ```
  sh tool/test.sh|train.sh {data} {model} {backbone}
  ```

    E.g. Test ASGNet with ResNet50 on COVID-QU-Ex:
    ```
    sh tool/test.sh covid asgnet resnet50
    ```

    E.g. Test ASGNet with Classification on COVID-QU-Ex:
    ```
    sh tool/test_cls.sh covid asgnet resnet50
    ```
