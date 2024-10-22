# Running Task Chains in Cyber RT

## Introduction

This project demonstrates how to create and run a task chain in Apollo Cyber RT. Cyber RT is the real-time communication infrastructure in the Apollo framework. It supports modular development and allows asynchronous communication between modules through task chains. This guide will walk you through the process from environment setup to running the task chain.

## Prerequisites

Before running the task chain, make sure you have the following dependencies installed:

Ubuntu 18.04 or later
Apollo installation (refer to the official Apollo installation documentation)
Docker (to start the Apollo container)
Basic C++ and Python development environment

## Installation and Setup

### 1. Clone the Apollo Repository

First, clone the Apollo repository and navigate to the project directory:

```bash
git clone https://github.com/ApolloAuto/apollo.git
cd apollo
```

### 2. Start the Cyber Container

Once inside the project directory, start the Cyber Docker container:
```bash
bash docker/scripts/dev_start.sh 
bash docker/scripts/dev_into.sh
```

### 3. Build case_study

```bash
bazel build cyber/case_study/...
```

### 4. Run case_study

```bash
cyber_launch start cyber/case_study/case_study.launch
```

