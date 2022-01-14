## Syntheitc Infant Data Generation

## Table of Contents
  * [Description](#description) 
  * [Environment](#environment)
  * [Quick Start](#quick-start)
  * [Citation](#citation)
  * [Acknowledgments](#acknowledgments)


## Description

This repository contains the fitting code borrowing from [Expressive Body Capture: 3D Hands, Face, and Body from a Single Image](https://smpl-x.is.tue.mpg.de/) 
so that fit the Skinned Multi-Infant Linear model (SMIL) model to provided poses, and rendering code to generate 2D synthetic images. 

## Environment
The code is developed using python 3.6 on Ubuntu 18.04. NVIDIA GPUs are needed. The code is developed and tested using one NVIDIA TITAN Xp GPU card. Other platforms or GPU cards are not fully tested.

## Quick Start
### Directory of synthetic infant image generation. 
   ```
   ${syn_generation}
   ├── cfg_files
   ├── data
   ├── models
   ├── output
   ├── priors
   ├── results
   ├── smplifyx
   ├── vposer
   ├── load_prior.py
   ├── README.md
   ├── requirements.txt
   ├── optional-requirements.txt
   └── render
   	├── bg_img
   	├── bodies
   	├── outputs
   	├── textures
   	├── smil_webuser
   	├── image_generation.py
   	├── smil_web.pkl
   	└── template.obj

   ```
### Installation
1. Install [Opendr 0.78](https://pypi.org/project/opendr/)
2. Install dependencies:
   ```
   pip install -r requirements.txt
   pip install -r optional-requirements.txt
   ```
### Preparation for fitting SMIL
   (1) Download [SMIL](https://www.iosb.fraunhofer.de/en/competences/image-exploitation/object-recognition/sensor-networks/motion-analysis.html) model `smil_web.pkl` and put it in `syn_generation/models` folder, place `smil_pose_prior` file into `syn_generation/priors`.
   (2) Place your infant images and corresponding poses in `syn_generation/data/images` and `syn_generation/data/keypoints` separately, as the example files in these folders.

Note: By downloading and/or using SMIL model, you need to agree to the license terms.
### Fitting SMIL
```
cd syn_generation
python smplifyx/main.py \
   --config cfg_files/fit_smil.yaml \
   --data_folder data \
   --output_folder output \
   --visualize=True \
   --model_folder models

```

### Preparation for rendering
   (1) Download background images from LSUN dataset using [this code](https://github.com/fyu/lsun). Or you can use any other images.
   (2) The infant textures in `syn_generation/render/textures` folder are downloaded from the Moving INfants In RGB-D [MINI-RGBD](https://www.iosb.fraunhofer.de/en/competences/image-exploitation/object-recognition/sensor-networks/motion-analysis.html) dataset. 
   In order to make appearances diverse, you can download adult clothing images from [SMPL](lsh.paris.inria.fr/SURREAL/smpl_data/textures.tar.gz) data.
   (3) Copy SMIL model `smil_web.pkl` to `syn_generation/render` folder.
  
### Rendering
```
cd render
python image_generation.py 

```
The generated synthetic images are saved in `syn_generation/render/output` folder.

## Citation
If you use our code or models in your research, please cite with:

```
@article{huang2020invariant,
  title={Invariant representation learning for infant pose estimation with small data},
  author={Huang, Xiaofei and Fu, Nihang and Liu, Shuangjun and Vyas, Kathan and Farnoosh, Amirreza and Ostadabbas, Sarah},
  journal={arXiv preprint arXiv:2010.06100},
  year={2020}
}
```

## Acknowledgement
Thanks for the open-source
* SmplifyX: [Expressive Body Capture: 3D Hands, Face, and Body from a Single Image, Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.](https://github.com/vchoutas/smplify-x).
* SURREAL: [Learning from Synthetic Humans (SURREAL), Gül Varol, Javier Romero, Xavier Martin, Naureen Mahmood, Michael J. Black, Ivan Laptev and Cordelia Schmid](https://github.com/gulvarol/surreal)
