# CZ4042-Neural-Network-Deep-Learning Group Project

## Setup instructions

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

## Assignment details
E. Clothing Classification
Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples
and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from
10 classes. One can design a convolutional neural network or Transformer to address the classification
problem. Some tasks to consider:
1. Modify some previously published architectures e.g., increase the network depth, reducing their
parameters, etc. Explore more advanced techniques such as deformable convolution or visual
prompt tuning for Transformers.
2. Use more advanced transformation techniques such as MixUp (see the original paper and its
PyTorch implementation here)
3. Comparing the performance of different network architectures
References
3. Deep Learning CNN for Fashion-MNIST Clothing Classification
Datasets:
1. The dataset is available in TorchVision:
https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html

## Done by
Wong Yi Pun  
Jeremy U Keat  
Ng Yue Jie Alphaeus 

## Acknowledgements
Fashion MNIST dataset: https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html
