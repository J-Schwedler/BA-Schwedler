#!/usr/bin/env bash
git clone https://github.com/OpenGVLab/InternImage.git
cd ./InternImage/segmentation/
conda create -n internimage python=3.7 -y
conda activate internimage
sudo apt install nvidia-driver-535
CUDNN_TAR_FILE="cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz"
wget https://developer.nvidia.com/compute/cudnn/secure/8.5.0/local_installers/11.7/cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz
tar -xzvf ${CUDNN_TAR_FILE}

sudo cp -P cuda/include/cudnn.h /usr/local/cuda-11.7/include
sudo cp -P cuda/lib/libcudnn* /usr/local/cuda-11.7/lib64/
sudo chmod a+r /usr/local/cuda-11.7/lib64/libcudnn*

nvidia-smi
nvcc -V

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
conda install -c conda-forge termcolor yacs pyyaml scipy pip -y
pip install opencv-python
pip install -U openmim
mim install mmcv-full==1.5.0
mim install mmsegmentation==0.27.0
pip install timm==0.6.11 mmdet==2.28.1
cd ./ops_dcnv3
sh ./make.sh
# unit test (should see all checking is True)
python test.py
