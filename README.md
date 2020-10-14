# Fine-tuned Domain-adapted Infant Pose (FiDIP)

Codes and experiments for the following paper: 


Xiaofei Huang, Nihang Fu, Sarah Ostadabbas, "Infant Pose Learning with Small Data."

Contact: 

[Xiaofei Huang](xhuang@ece.neu.edu)

[Nihang Fu](nihang@ece.neu.edu)

[Sarah Ostadabbas](ostadabbas@ece.neu.edu)

## Table of Contents
  * [Introduction](#introduction)
  * [Main Results](#main-results)
  * [Environment](#environment)
  * [Quick Start](#quick-start)
  * [Synthetic Infant Data Generation](#synthetic-infant-data-generation)
  * [Citation](#citation)
  * [Acknowledgments](#acknowledgments)

## Introduction
This is an official pytorch implementation of [*Infant Pose Learning with Small Data*](https://arxiv.org/abs/2010.06100). This work proposes a fine-tuned domain-adapted infant pose (FiDIP) estimation model, that transfers the knowledge of adult poses into estimating infant pose with the supervision of a domain adaptation technique on our released synthetic and real infant pose, called [SyRIP dataset](https://coe.northeastern.edu/Research/AClab/SyRIP/). On SyRIP test dataset, our FiDIP model outperforms other state-of-the-art human pose estimation model for the infant pose estimation, with the mean average precision (AP) as high as 92.2. And the implementation of synthetic infant data generation is located under the root path.   </br>

## Main Results
### Performance comparison between FiDIP network and the SOTA pose estimators the COCO Val2017 and SyRIP test dataset
| Pose Estimation Model | Backbone Network | Input Image Size | COCO Val2017-AP | SyRIP Test-AP |
|---|---|---|---|---|
| Faster R-CNN | ResNet-50-FPN | Flexible | 65.5 | 70.1 |
| Faster R-CNN | ResNet-101-FPN | Flexible | 66.1 | 64.4 |
| DarkPose | ResNet-50 | 128x96 | 64.5 | 65.9 |
| DarkPose | HRNet-W48 | 128x96 | 74.2 | 82.1 |
| DarkPose | HRNet-W32 | 256x192 | 77.9 | 88.5 |
| DarkPose | HRNet-W48 | 384x288 | 79.2 | 88.4 |
| Pose-ResNet | ResNet-50 | 256x192 | 72.4 | 80.4 |
| Pose-ResNet | ResNet-50 | 384x288 | 72.3 | 82.4 |
| FiDIP(ours) | ResNet-50 | 384x288 | 59.1 | 92.2 |

## Environment
The code is developed using python 3.6 on Ubuntu 18.04. NVIDIA GPUs are needed. The code is developed and tested using one NVIDIA TITAN Xp GPU card. Other platforms or GPU cards are not fully tested.

## Quick Start
### Installation
1. Install pytorch = v1.6.0 with cuda 10.1 following [official instruction](https://pytorch.org/).

2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT} and get the following directory.
   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── pose_estimation
   ├── README.md
   ├── requirements.txt
   └── syn_generation

   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
5. Install [COCOAPI](https://github.com/cocodataset/cocoapi)

### Download pretrained models
 (1) download pytorch imagenet pretrained models from [pytorch model zoo](https://pytorch.org/docs/stable/model_zoo.html#module-torch.utils.model_zoo) and caffe-style pretrained models from [GoogleDrive](https://drive.google.com/drive/folders/1yJMSFOnmzwhA4YYQS71Uy7X1Kl_xq9fN?usp=sharing). 
 (2) download coco pretrained models from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW0D5ZE4ArK9wk_fvw) or [GoogleDrive](https://drive.google.com/drive/folders/13_wJ6nC7my1KKouMkQMqyr9r1ZnLnukP?usp=sharing). 
 (3) download our FiDIP pretrained model from [GoogleDrive](https://drive.google.com/file/d/13xa0Rpns_9a2KEqgpyv7BXIK3i9fiFYV/view?usp=sharing). 
   Please download them under ${POSE_ROOT}/models/pytorch, and make them look like this:

   ```
   ${POSE_ROOT}
    `-- models
        `-- pytorch
            |-- imagenet
            |   |-- resnet50-19c8e357.pth
            |   `-- resnet50-caffe.pth.tar
            `-- pose_coco
                |-- pose_resnet_50_256x192.pth.tar
                |-- pose_resnet_50_384x288.pth.tar
                `-- FiDIP.pth.tar
   ```
   
### Data preparation
For SyRIP data, please download from [SyRIP dataset](https://coe.northeastern.edu/Research/AClab/SyRIP/). Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train_pre_infant.json
        |   |-- person_keypoints_train_infant.json   
        |   `-- person_keypoints_validate_infant.json
        `-- images
            |-- train_pre_infant
            |   |-- train00001.jpg
            |   |-- train00002.jpg
            |   |-- train00003.jpg
            |   |-- ... 
            |-- train_infant
            |   |-- train00001.jpg
            |   |-- train00002.jpg
            |   |-- train00003.jpg
            |   |-- ...  
            |   |-- train10001.jpg
            |   |-- train10002.jpg
            |   |-- train10003.jpg
            |   |-- ...  
            `-- validate_infant
                |-- test0.jpg
                |-- test1.jpg
                |-- test2.jpg
                |-- ... 
```

### Valid on SyRIP validate dataset using FiDIP pretrained models

```
python pose_estimation/valid.py \
    --cfg experiments/coco/resnet50/384x288_d256x3_adam_lr1e-3_infant.yaml\
    --model-file models/pytorch/pose_coco/FiDIP.pth.tar
```

### Training on SyRIP dataset

```
python pose_estimation/train_adaptive_model.py  --cfg experiments/coco/resnet50/384x288_d256x3_adam_lr1e-3_infant.yaml  --checkpoint models/pytorch/pose_coco/pose_resnet_50_384x288.pth.tar
```

## Synthetic Infant Data Generation
Please follow the [README.md](./syn_generation/README.md) in the folder `syn_generation`.

## Citation

If you use our code or models in your research, please cite with:

```

```


## Acknowledgement
Thanks for the open-source Pose-ResNet
* [Simple Baselines for Human Pose Estimation and Tracking, Xiao, Bin and Wu, Haiping and Wei, Yichen](https://github.com/microsoft/human-pose-estimation.pytorch)

## License 
* This code is for non-commertial purpose only. 
* For further inquiry please contact: Augmented Cognition Lab at Northeastern University: http://www.northeastern.edu/ostadabbas/ 


