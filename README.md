# Fine-tuned Domain-adapted Infant Pose (FiDIP)

Codes and experiments for the following paper: 

Xiaofei Huang, Nihang Fu, Shuangjun Liu, Sarah Ostadabbas, “Invariant representation learning for infant pose estimation withsmall data,” in IEEE International Conference on Automatic Face and Gesture Recognition (FG), Dec. 2021


Contact: 

[Xiaofei Huang](xhuang@ece.neu.edu)

[Sarah Ostadabbas](ostadabbas@ece.neu.edu)

## Table of Contents
  * [Introduction](#introduction)
  * [What's New](#what's_new) :fire:
  * [Main Results](#main-results)
  * [Environment](#environment)
  * [Quick Start](#quick-start)
  * [Synthetic Infant Data Generation](#synthetic-infant-data-generation)
  * [Citation](#citation)
  * [Acknowledgments](#acknowledgments)

## Introduction
This is an official pytorch implementation of [*Invariant Representation Learning for Infant Pose Estimation with Small Data*](https://arxiv.org/abs/2010.06100). This work proposes a fine-tuned domain-adapted infant pose (FiDIP) estimation model, that transfers the knowledge of adult poses into estimating infant pose with the supervision of a domain adaptation technique on our released synthetic and real infant pose (SyRIP) dataset. Integrated with pose estimation backbone networks with varying complexity,  FiDIP performs consistently better than the fine-tuned versions of those models.  One of our best infant pose estimation performers on the state-of-the-art DarkPose model shows mean average precision (mAP) of 93.6. And the implementation of synthetic infant data generation is located under the root path.   </br>



## What's New :fire:
  * Release [expanded SyRIP dataset](https://coe.northeastern.edu/Research/AClab/Expanded_SyRIP/) including extra 400 real images in training set. Please contact us for the password of our dataset.
  * Release [new FiDIP model](https://drive.google.com/file/d/1UHQC63mSEL4Vcuenqg9R3OaNP0Wv_6gd/view?usp=drive_link) trained on the expanded SyRIP dataset.



## Main Results
#### Performance comparison of three SOTA pose estimation models (SimpleBaseline, DarkPose, Pose-MobileNet) fine-tuned on MINI-RGBD, SyRIP-syn (synthesized data only) and SyRIP whole set and tested on SyRIP Test100.
| Train Set | Method | Backbone | Input size | AP | AP50 | AP75 | AR | AR50 | AR75 |
|---|---|---|---|---|---|---|---|---|---|
| MINI-RGBD | SimpleBaseline | ResNet-50 | 384x288 | 69.2 | 95.8 | 78.0 | 72.4 | 97.0 | 81.0 |
| SyRIP-syn | SimpleBaseline | ResNet-50 | 384x288 | 85.3 | 97.1 | 91.8 | 87.4 | 98.0 | 93.0 |
| SyRIP | SimpleBaseline | ResNet-50 | 384x288 | **90.1** | 98.5 | 97.2 | 91.6 | 99.0 | 98.0 |
| MINI-RGBD | DrakPose | HRNet-W48 | 384x288 | 85.2 | 98.6 | 95.3 | 87.0 | 99.0 | 96..0 |
| SyRIP-syn | DrakPose | HRNet-W48 | 384x288 | 91.4 | 98.5 | 98.5 | 92.7 | 99.0 | 99.0 |
| SyRIP | DrakPose | HRNet-W48 | 384x288 | **92.7** | 98.5 | 98.5 | 93.9 | 99.0 | 99.0 |
| MINI-RGBD | Pose-MobileNet | MobileNetV2 | 224x224 | 12.3 | 38.1 | 3.8 | 21.6 | 52.0 | 14.0 |
| SyRIP-syn | Pose-MobileNet | MobileNetV2 | 224x224 | 60.3 | 91.1 | 62.7 | 68.4 | 95.0 | 72.0 |
| SyRIP | Pose-MobileNet | MobileNetV2 | 224x224 | **78.9** | 97.2 | 90.6 | 84.2 | 98.0 | 94.0 |

#### Generality of our FiDIP method to three models on the SyRIP Test100.
| Method | Backbone | Input size | \# Params | GFLOPs | AP | AP50 | AP75 | AR | AR50 | AR75 |
|---|---|---|---|---|---|---|---|---|---|---|
| SimpleBaseline | ResNet-50 | 384x288 | 32.42M | 20.23 | 82.4 | 98.9 | 92.2 | 83.8 | 99.0 | 93.0 |
| SimpleBaseline + Finetune | ResNet-50 | 384x288 | 32.42M | 20.23 | 90.1 | 98.5 | 97.2 | 91.6 | 99.0 | 98.0 |
| SimpleBaseline + FiDIP | ResNet-50 | 384x288 | 32.42M | 20.23 | **91.1** | 98.5 | 98.5 | 92.6 | 99.0 | 99.0 |
| DarkPose | HRNet-W48 | 384x288 | 60.65M | 32.88 | 88.5 | 98.5 | 98.5 | 90.0 | 99.0 | 99.0 |
| DarkPose + Finetune | HRNet-W48 | 384x288 | 60.65M | 32.88 | 92.7 | 98.5 | 98.5 | 93.9 | 99.0 | 99.0 |
| DarkPose + FiDIP | HRNet-W48 | 384x288 | 60.65M | 32.88 | **93.6** | 98.5 | 98.5 | 94.6 | 99.0 | 99.0 |
| Pose-MobileNet | MobileNetV2 | 224x224 | 3.91M | 0.46 | 46.5 | 85.7 | 45.6 | 56.2 | 89.0 | 59.0 |
| Pose-MobileNet + Finetune | MobileNetV2 | 224x224 | 3.91M | 0.46 | 78.9 | 97.2 | 90.6 | 84.2 | 98.0 | 94.0 |
| Pose-MobileNet + FiDIP | MobileNetV2 | 224x224 | 3.91M | 0.46 | **79.2** | 99.0 | 89.4 | 84.1 | 99.0 | 92.0 |

 
## Environment
The code is developed using python 3.12 with CUDA 12.1 on Ubuntu 18.04. NVIDIA GPUs are needed. The code is developed and tested using one NVIDIA TITAN Xp GPU card. Other platforms or GPU cards are not fully tested.

## Quick Start
### Installation
1. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT} and get the following directory.
   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── tools
   ├── README.md
   └── syn_generation

   ```

2. ```
   conda env create -f fidip_env.yml
   ```

3. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```

### Download pretrained models
 (1) download DarkPose pretrained models from [TGA_models](https://drive.google.com/drive/folders/14kAA1zXuKODYgrRiQmKnVcipbY7RedVV). 
 (2) download our FiDIP pretrained model from [FiDIP_models](https://drive.google.com/drive/folders/108P-1SnTqaj3xNtjYZ1o7T8z6UvUYuiC?usp=sharing). 
 Please download them under ${POSE_ROOT}/models, and make them look like this:

   ```
   ${POSE_ROOT}
    `-- models
        |-- hrnet_fidip.pth
        |-- mobile_fidip.pth
        `-- coco
            `-- w48_384x288.pth
   ```
   
### Data preparation
For SyRIP data, please download from [SyRIP dataset](https://coe.northeastern.edu/Research/AClab/SyRIP/). The original SyRIP data can be download from **`SyRIP.zip'**, and the data for FiDIP model training can be download from **'syrip_for_train.zip'**. Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- syrip
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
### Test on SyRIP validate dataset using FiDIP pretrained models

```
python tools/test_adaptive_model.py \
    --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml \
    TEST.MODEL_FILE models/hrnet_fidip.pth TEST.USE_GT_BBOX True
```

### Training on SyRIP dataset

```
python tools/train_adaptive_model_hrnet.py \
    --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml
```

## Synthetic Infant Data Generation
Please follow the [README.md](./syn_generation/README.md) in the folder `syn_generation`.

## Citation

If you use our code or models in your research, please cite with:

```
@inproceedings{huang2021infant,
  title={Invariant Representation Learning for Infant Pose Estimation with Small Data},
  author={Huang, Xiaofei and Fu, Nihang and Liu, Shuangjun and Ostadabbas, Sarah},
  booktitle={IEEE International Conference on Automatic Face and Gesture Recognition (FG), 2021},
  month     = {December},
  year      = {2021}
}
```

## Acknowledgement
Thanks for the open-source DarkPose
* [Distribution-Aware Coordinate Representation for Human Pose Estimation, Feng Zhang, Xiatian Zhu, Hanbin Dai, Mao Ye, Ce Zhu](https://github.com/ilovepose/DarkPose)

Thanks for the SURREAL dataset
* [Learning from Synthetic Humans, Gül Varol, Javier Romero, Xavier Martin, Naureen Mahmood, Michael J. Black, Ivan Laptev and Cordelia Schmid](https://github.com/gulvarol/surreal)

## License 
* This code is for non-commertial purpose only. 
* For further inquiry please contact: Augmented Cognition Lab at Northeastern University: http://www.northeastern.edu/ostadabbas/ 




