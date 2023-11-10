# CZ4042-Neural-Network-Deep-Learning Group Project

## Getting started

The following instructions setups a new virtual environment for python and installs the needed libraries.  
These instructions assumes that the user is using a Windows machine.

```cmd
python -m venv .venv
.venv\Scripts\activate
```

To install CUDA-supported PyTorch: https://pytorch.org/get-started/locally/
(12.1 in this example)

```cmd
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

To set-up Jupyter server to run your .venv environment:

```cmd
pip install jupyter ipykernel
python -m ipykernel install --user --name=cz4042_group_proj
```

## Abstract

Neural networks and deep learning have revolutionized image classification, with the Fashion MNIST dataset serving as a
benchmark for evaluating model performance. This project explores techniques to improve classification accuracy using deep
learning methods, focusing on transfer learning, deformable convolution, and the CutMix data augmentation strategy. The
dataset's characteristics, including limited classes, uniform image size, and simplified items, pose unique challenges
that necessitate efficient and adaptable model architectures. Our experiments include a simple CNN model, transfer
learning with ShuffleNet and MobileNet, deformable convolution, and the application of CutMix. We analyze the results,
highlighting the effectiveness of transfer learning and discussing the impact of CutMix on model performance. The
findings underscore the importance of selecting appropriate techniques tailored to the dataset's characteristics,
providing valuable insights for image classification tasks in constrained environments.

## Contributors

[Jeremy U Keat](https://github.com/jeremyu25)  
[Ng Yue Jie Alphaeus](https://github.com/AlphaeusNg)  
[Wong Yi Pun](https://github.com/ypwong99)

## Acknowledgements

[Fashion MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html)
