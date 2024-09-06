# 在 Cyber RT 中运行任务链

## 简介

该项目展示了如何在 [Apollo Cyber RT](https://github.com/ApolloAuto/apollo) 中创建并运行一个任务链。Cyber RT 是 Apollo 框架中的实时通信基础设施，支持模块化开发并允许通过任务链进行模块之间的异步通信。本指南将引导你从环境配置到任务链的运行。

## 前提条件

在运行任务链之前，请确保你已经安装了以下依赖：

- Ubuntu 18.04 或更高版本
- Apollo 安装（请参考 Apollo 官方安装文档）
- Docker（用于启动 Apollo 容器）
- 基本的 C++ 和 Python 开发环境

## 安装和配置

### 1. 克隆 Apollo 仓库

首先，克隆 Apollo 仓库并进入项目目录：

```bash
git clone https://github.com/ApolloAuto/apollo.git
cd apollo
```

### 2. 启动 Cyber 容器

进入项目目录后，启动 Cyber 的 Docker 容器：
```bash
bash docker/scripts/cyber_start.sh
bash docker/scripts/cyber_into.sh
```
### 3. 安装 GPU 支持

运行以下脚本来安装 GPU 支持：
(Enter the CyberRT docker container)
```bash
sudo bash docker/build/installers/install_gpu_support.sh
bazel build third_party/gpus/...  
```
### 4. 安装 Opencv 支持

运行以下脚本来安装 Opencv 支持：

```bash
sudo bash docker/build/installers/install_opencv.sh
bazel build third_party/opencv/...  
```
### 5. 安装 libnuma1 包
```bash
sudo apt-get install libnuma1
sudo ldconfig  
```

### 6. Install glog Library
```bash
sudo apt-get update
sudo apt-get install libgoogle-glog-dev 
```

### 7. 把case_study目录放在apollo/cyber下
首先编译case_study
```bash
bazel build cyber/case_study/...
```
此时可能会出现文件缺失的情况，需要执行以下操作
```bash
cp /usr/local/cuda-11.1/lib64/libcudart.so /apollo/bazel-out/k8-fastbuild/bin/_solib_k8/_U@local_Uconfig_Ucuda_S_Scuda_Ccudart___Ulocal_Uconfig_Ucuda_Scuda_Scuda_Slib
```


