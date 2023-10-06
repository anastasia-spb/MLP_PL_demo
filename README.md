# MNIST Classification with PyTorchLightning

## Use Dev Container for development

Prerequisites:
* Visual Studio Code
* Docker
* NVIDIA Container Toolkit
* Dev Container extension

1. Start VS Code and press Ctrl+Shift+P
2. Type: Dev Containers: Open Folder in Container
3. Select repository root folder
4. Wait for docker image download


Detailed info:
https://code.visualstudio.com/docs/devcontainers/containers 
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

## Train model

Run train_pl_model.py 

## Eval model

Enter path to the selected checkpoint file in config.py and run eval.py
