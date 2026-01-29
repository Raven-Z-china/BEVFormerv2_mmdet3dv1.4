Installation
conda create --name BEVFormerv2 python==3.10 -y
conda activate BEVFormerv2
pip install torch==2.1.0 torchvision==0.16.0
pip install -U openmim
#If mmcv._ext is not installed, you can download it via wget:
#wget https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl
mim install mmengine==0.11.0rc0 mmcv==2.1.0 mmdet==3.2.0
pip install fvcore seaborn scikit_image scikit_learn opencv-python-headless==4.6.0.66 numpy==1.26.4
pip install pytorch-lightning==1.9.5 deepspeed wandb ninja --upgrade
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# Or, to install it from a local clone:
#git clone https://github.com/facebookresearch/detectron2.git
#python detectron2/setup.py develop

#NOTE: install mmdetection3dv1.4 here

cd BEVFormerv2

Prepare nuScenes data
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data


Train
bash tools/dist_train.sh projects/BEVFormerv2/configs/bevformerv2/bevformerv2-r50-t2-24ep.py 1 --amp

Test
bash tools/dist_test.sh projects/BEVFormerv2/configs/bevformerv2/bevformerv2-r50-t2-24ep.py ./ckpts/fcos3d_r50_fp16.pth 1