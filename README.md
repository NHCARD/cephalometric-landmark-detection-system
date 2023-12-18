## 두부계측분석 인공지능 GUI 프로그램

---
인공지능을 사용해 두부 X-ray 이미지에서 계측점 검출을 진행하고 GUI 프로그램을 사용하여 사용합니다.
계측점 검출을 자동화하고, 이를 저장함으로써 업무량을 줄일 수 있습니다.
추가요망.....


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

---

* python 3.8+
* PyTorch
* PyQT5
* CUDA

Our development environment
* windows 11
* PyTorch 1.13.0
* PyQt5
* CUDA 11.7


# Installation

---
Create a virtual environments

    conda create -n cl python=3.8

Install PyTorch and torchaudio following the [official download guide](https://pytorch.org/get-started/locally/) e.g.,

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Install requirements

    pip install -r requirements.txt

Clone this repository

    git clone https://github.com/NHCARD/CL-Contest.git

Load Dataset from private submodule

    git submodule init
    git submodule update

# Stacks

---

![Python](https://img.shields.io/badge/Python-3.8-3776AB?logo=python)

![PyTorch](https://img.shields.io/badge/PyTorch-1.13.0-EE4C2C?logo=pytorch)

![PyQt](https://img.shields.io/badge/PyQt5-1.13.0-041CD52?logo=qt)
