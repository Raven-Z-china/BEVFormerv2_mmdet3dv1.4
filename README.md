# BEVFormerv2

A 3D object detection framework based on BEV (Bird's Eye View) representation.

https://github.com/fundamentalvision/BEVFormer/tree/master

## 📋 Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Testing](#testing)
- [Notes](#notes)
- [Directory Structure](#directory-structure)

## Installation

### 1. Create and Activate Conda Environment
```bash
conda create --name BEVFormerv2 python==3.10 -y
conda activate BEVFormerv2
```
### 2. Install PyTorch
```bash
pip install torch==2.1.0 torchvision==0.16.0
```
### 3. Install MMLab Series Toolkits
```bash
pip install -U openmim
mim install mmengine==0.11.0rc0 mmcv==2.1.0 mmdet==3.2.0
# Note: If you encounter issues installing mmcv._ext, you can manually download and install it:
# wget https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl
# pip install mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl
```
### 4. Install Other Dependencies
```bash
pip install fvcore seaborn scikit_image scikit_learn opencv-python-headless==4.6.0.66 numpy==1.26.4
pip install pytorch-lightning==1.9.5 deepspeed wandb ninja --upgrade
```
### 5. Install Detectron2
Method 1: Install directly via Git
```
bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
Method 2: Install from a local clone
```
bash
git clone https://github.com/facebookresearch/detectron2.git
python detectron2/setup.py develop
```
### 6. Install mmdetection3d v1.4
Please refer to the official documentation to install mmdetection3d v1.4.

### 7. Enter the Project Directory
```bash
cd BEVFormerv2
```

## Data Preparation
nuScenes Dataset
Prepare the nuScenes dataset by running the following command:
```
bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
```
## Training
Start training using the distributed training script:
```
bash
bash tools/dist_train.sh projects/BEVFormerv2/configs/bevformerv2/bevformerv2-r50-t2-24ep.py 1 --amp
``` 
## Testing
Evaluate the model using the distributed testing script:
```
bash
bash tools/dist_test.sh projects/BEVFormerv2/configs/bevformerv2/bevformerv2-r50-t2-24ep.py ./ckpts/fcos3d_r50_fp16.pth 1
```
## Notes
Ensure that the corresponding versions of CUDA and cuDNN are correctly installed.

Download the nuScenes dataset and place it in the ./data/nuscenes directory before data preparation.

The number 1 in the training and testing scripts indicates the number of GPUs used. Adjust this according to your actual situation.

The --amp flag enables automatic mixed precision training to save GPU memory and accelerate training.

## Directory Structure
```text
BEVFormerv2/
├── projects/
│   └── BEVFormerv2/
│       └── configs/
│           └── bevformerv2/
│               └── bevformerv2-r50-t2-24ep.py
├── tools/
│   ├── create_data.py
│   ├── dist_train.sh
│   └── dist_test.sh
├── data/
│   └── nuscenes/
├── ckpts/
└── ...
```
For questions, please refer to the relevant documentation or submit an issue.
