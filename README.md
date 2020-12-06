# Fine-tuned Domain-adapted Infant Pose (FiDIP)

Codes and experiments for the following paper: 


Xiaofei Huang, Nihang Fu, Shuangjun Liu, Kathan Vyas, Amirreza Farnoosh, Sarah Ostadabbas, "Invariant Representation Learning for Infant Pose Estimation with Small Data."

Contact: 

[Xiaofei Huang](xhuang@ece.neu.edu)

[Nihang Fu](nihang@ece.neu.edu)

[Sarah Ostadabbas](ostadabbas@ece.neu.edu)

Sarah Ostadabbas
## Table of Contents
  * [Introduction](#introduction)
  * [Main Results](#main-results)
  * [Environment](#environment)
  * [Quick Start](#quick-start)
  * [Synthetic Infant Data Generation](#synthetic-infant-data-generation)
  * [Citation](#citation)
  * [Acknowledgments](#acknowledgments)

## Introduction
This is an official pytorch implementation of [*Invariant Representation Learning for Infant Pose Estimation with Small Data*](https://arxiv.org/abs/2010.06100). This work proposes a fine-tuned domain-adapted infant pose (FiDIP) estimation model, that transfers the knowledge of adult poses into estimating infant pose with the supervision of a domain adaptation technique on our released synthetic and real infant pose (SyRIP) dataset. On SyRIP test dataset, our FiDIP model outperforms other state-of-the-art human pose estimation model for the infant pose estimation, with the mean average precision (AP) as high as 90.1 on Test100. And the implementation of synthetic infant data generation is located under the root path.   </br>

## Main Results
### Performance comparison between FiDIP network and the SOTA pose estimators on the COCO Val2017, SyRIP Test500 and SyRIP Test100 dataset
| Pose Estimation Model | Backbone Network | Input Image Size | COCO Val2017-AP | SyRIP Test500-AP | SyRIP Test100-AP |
|---|---|---|---|---|---|
| Faster R-CNN | ResNet-50-FPN | Flexible | 65.5 | 93.4 | 70.1 |
| Faster R-CNN | ResNet-101-FPN | Flexible | 66.1 | 91.9 | 64.4 |
| DarkPose | ResNet-50 | 128x96 | 64.5 | 95.2 | 65.9 |
| DarkPose | HRNet-W48 | 128x96 | 74.2 | 97.4 | 82.1 |
| DarkPose | HRNet-W32 | 256x192 | 77.9 | 97.7 | 88.5 |
| DarkPose | HRNet-W48 | 384x288 | 79.2 | 98.0 | 88.4 |
| Pose-ResNet | ResNet-50 | 256x192 | 72.4 | 97.3 | 80.4 |
| Pose-ResNet | ResNet-50 | 384x288 | 72.3 | 97.5 | 82.4 |
| RMPE | VGG_SSD | 500×500 | 61.8 | 76.2 | 76.3 |
| UDP | HRNet-W32 | 256×192 | 74.4 | 81.2 | 79.8 | 
| UDP + Pose-ResNet | ResNet-50 | 256×192 | 70.4 | 83.4 | 78.2 | 
| UDP + Pose-ResNet | ResNet-152 | 384×288 | 74.3 | 84.2 | 79.1 |
| FiDIP(ours) | ResNet-50 | 384x288 | 59.1 | 98.2 | 90.1|
 
## Environment
The code is developed using python 3.6 on Ubuntu 18.04. NVIDIA GPUs are needed. The code is developed and tested using one NVIDIA TITAN Xp GPU card. Other platforms or GPU cards are not fully tested.

## Quick Start
### Installation
1. Install pytorch = v1.7.0 with cuda 10.1 following [official instruction](https://pytorch.org/).

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
 (3) download our FiDIP pretrained model from [GoogleDrive](https://drive.google.com/file/d/18BX-3yaSgnYGl0hjSQSegVe-AVkHJnUn/view?usp=sharing). 
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
Note: our train_pre_infant dataset with only real/synthetic labels consists of 1904 samples from COCO Val2017 dataset and 2000 synthetic adult images from SURREAL dataset. If you need it to train model, please contact us. Or you can create your own train_pre_infant dataset.
### Valid on SyRIP validate dataset using FiDIP pretrained models

```
python pose_estimation/valid.py \
    --cfg experiments/coco/resnet50/384x288_d256x3_adam_lr1e-3_infant.yaml\
    --model-file models/pytorch/pose_coco/FiDIP.pth.tar
```

### Training on SyRIP dataset

```
python pose_estimation/train_adaptive_model.py \
    --cfg experiments/coco/resnet50/384x288_d256x3_adam_lr1e-3_infant.yaml\
    --checkpoint models/pytorch/pose_coco/pose_resnet_50_384x288.pth.tar
```

## Synthetic Infant Data Generation
Please follow the [README.md](./syn_generation/README.md) in the folder `syn_generation`.

## Citation

If you use our code or models in your research, please cite with:

```
@article{huang2020infant,
  title={Infant Pose Learning with Small Data},
  author={Huang, Xiaofei and Fu, Nihang and Ostadabbas, Sarah},
  journal={arXiv preprint arXiv:2010.06100},
  year={2020}
}
```

## Acknowledgement
Thanks for the open-source Pose-ResNet
* [Simple Baselines for Human Pose Estimation and Tracking, Xiao, Bin and Wu, Haiping and Wei, Yichen](https://github.com/microsoft/human-pose-estimation.pytorch)
Thanks for the SURREAL dataset
* [Learning from Synthetic Humans, Gül Varol, Javier Romero, Xavier Martin, Naureen Mahmood, Michael J. Black, Ivan Laptev and Cordelia Schmid](https://github.com/gulvarol/surreal)

## License 
* This code is for non-commertial purpose only. 
* For further inquiry please contact: Augmented Cognition Lab at Northeastern University: http://www.northeastern.edu/ostadabbas/ 




