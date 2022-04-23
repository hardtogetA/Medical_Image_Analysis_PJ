# Medical_Image_Analysis_PJ

### Dataset
- Prepare download the COVID-QU-Ex dataset and put it under data/datasets/COVID-QU-Ex datasets

### Pre-trained models
- Pre-trained backbones for resnet50 and resnet101 can be found in [Google Driver](https://drive.google.com/drive/folders/1dEJL_KSkZZ0nIEy6zwqqb93L4zBDvCV-?usp=sharing)
- Download backbones and put the pth files under `initmodel/` folder

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