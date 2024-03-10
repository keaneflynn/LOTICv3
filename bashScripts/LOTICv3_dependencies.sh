#!/bin/bash

sudo apt update
sudo apt -y upgrade
sudo apt -y autoremove

wget https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
sudo apt-get -y install python3-pip libopenblas-base libopenmpi-dev libomp-dev cmake git
pip3 install Cython 
sudo pip3 install -U jetson-stats
sudo systemctl restart jtop.service
pip3 install numpy==1.23.1 torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

sudo apt-get -y install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.16.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.16.1
python3 setup.py install --user

cd ~/
pip3 install ultralytics
rm -rf torchvision
echo 'export PATH=/home/keane/.local/bin:$PATH' >> ~/.bashrc
echo 'alias python=python3' >> ~/.bashrc
echo 'alias pip=pip3' >> ~/.bashrc
sudo /usr/sbin/nvpmodel -m 8
sudo reboot

