## Enabling Efficient GPU Resource Allocation and Real-time Scheduling for Autonomous Driving System

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

### 2. 启动 Cyber 容器

进入项目目录后，启动 Cyber 的 Docker 容器：

bash docker/scripts/cyber_start.sh
bash docker/scripts/cyber_into.sh

