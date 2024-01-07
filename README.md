## Cephalometric Landmark Detection System:
A user-friendly GUI that allows users to load cephalometric X-ray images, visualize the model's output, and conveniently edit landmark positions. 

![main.png](utils%2Fmain.png)
*2023 SW인재페스티벌 출품작*

## Team Introduction

* **황현성**
  * 인공지능 구현
  * GUI
* 정연주
  * 인공지능 구현
  * GUI
* 한승용
  * 인공지능 구현
  * GUI

# Requirements

* Python 3.8+
* PyTorch
* PyQT5
* CUDA

Our development environment
* Windows 11
* PyTorch 1.13.0
* PyQt5
* CUDA 11.7


# Installation

Create a virtual environment

    conda create -n cl python=3.8

Install PyTorch and torchaudio following the [official download guide](https://pytorch.org/get-started/locally/) e.g.,

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Install requirements

    pip install -r requirements.txt

Clone repository

    git clone https://github.com/NHCARD/cephalometric-landmark-detection-system.git

Load the dataset from private submodule

    git submodule init
    git submodule update

Run GUI

    python plot.py

# Stacks

![Python](https://img.shields.io/badge/Python-3.8-3776AB?logo=python)

![PyTorch](https://img.shields.io/badge/PyTorch-1.13.0-EE4C2C?logo=pytorch)

![PyQt](https://img.shields.io/badge/PyQt5-1.13.0-041CD52?logo=qt)
