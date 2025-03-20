# Self-Contrastive-Forward-Forward
This repo implements the official code of the article: ["Self-Contrastive Forward-Forward Algorithm"](http://arxiv.org/abs/2409.11593)
## Description

The Forward-Forward (FF) algorithm is a recent, purely forward-mode learning method, that updates weights locally and layer-wise and supports supervised as well as unsupervised learning. These features make it ideal for applications such as brain-inspired learning, low-power hardware neural networks, and distributed learning in large models. However, while FF has shown promise on written digit recognition tasks, its performance on natural images and time-series remains a challenge. A key limitation is the need to generate high-quality negative examples for contrastive learning, especially in unsupervised tasks, where versatile solutions are currently lacking. To address this, we introduce the Self-Contrastive Forward-Forward (SCFF) method, inspired by self-supervised contrastive learning. SCFF generates positive and negative examples applicable across different datasets, surpassing existing local forward algorithms for unsupervised classification accuracy on MNIST (MLP: 98.7%), CIFAR-10 (CNN: 80.75%), and STL-10 (CNN: 77.3%). Additionally, SCFF is the first to enable FF training of recurrent neural networks, opening the door to more complex tasks and continuous-time video and text processing.

## Getting Started

### Dependencies

* python=3.10.9; cuda version: 11.8
* Other required packages are listed in requirement.txt
* Environment: Linux (Ubuntu 22.04.2 LTS)

### Installing

* Creat a virtual conda environment to avoid conflicts of version
```
conda create --name scff python=3.10.9
```
* Activate the environment
```
conda activate scff
```
* Install the Dependencies
```
pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```
## Overview
### SCFF supports two different training strategies:

* [Greedy layer-wise training](#greedy-layer-wise-training) ([CIFAR-10](#run-the-cifar-10-experiment), [STL-10](#run-the-STL-10-experiment), [MNIST (MLP)](#run-the-mnist-mlp-experiment), [FSDD](#run-the-fsdd-experiment))
* [Parallel training of all layers simultaneously](#parallel-training-of-all-layers-simultaneously)


## Greedy layer-wise training
### Run the CIFAR-10 experiment

* Run the SCFF_CIFAR.py file to train different layers; the output and model are saved in the folder "./results"  
--epochs: number of epochs  
--NL: layer index to train  
--save_model: save the trained layer  
--device_num: device number if using GPU    
--tr_and_eval: test the accuracy at each epoch of training  

* SCFF training of the first layer and save the best model for the next layer train
```
nohup python -u SCFF_CIFAR.py --epochs 6 --NL 1 --device_num 0 --save_model > ./results/SCFF_CIFAR_l1.txt 2>&1 &
```
* SCFF training of the second layer and save the best model for the next layer train
```
nohup python -u SCFF_CIFAR.py --epochs 4 --NL 2 --device_num 0 --save_model > ./results/SCFF_CIFAR_l2.txt 2>&1 &
```
* SCFF training of the third layer
```
nohup python -u SCFF_CIFAR.py --epochs 21 --NL 3 --device_num 0  > ./results/SCFF_CIFAR_l3.txt 2>&1 &
```

### Run the STL-10 experiment

* Run the SCFF_STL.py file to train different layers; the output and model are saved in the folder "./results";  
--epochs: number of epochs  
--NL: layer index to train  
--save_model: save the trained layer  
--device_num: device number if using GPU    
--tr_and_eval: test the accuracy at each epoch of training  

* SCFF training of the first layer and save the best model for the next layer train
```
nohup python -u SCFF_STL.py --epochs 4 --NL 1 --device_num 0 --save_model > ./results/SCFF_STL_l1.txt 2>&1 &
```
* SCFF training of the second layer and save the best model for the next layer train
```
nohup python -u SCFF_STL.py --epochs 5 --NL 2 --device_num 0 --save_model > ./results/SCFF_STL_l2.txt 2>&1 &
```
* SCFF training of the third layer and save the best model for the next layer train
```
nohup python -u SCFF_STL.py --epochs 12 --NL 3 --device_num 0 --save_model > ./results/SCFF_STL_l3.txt 2>&1 &
```
* SCFF training of the fourth layer
```
nohup python -u SCFF_STL.py --epochs 12 --NL 4 --device_num 0 > ./results/SCFF_STL_l4.txt 2>&1 &
```

### Run the MNIST (MLP) experiment
* Run the SCFF_MNIST.py file to train different layers; the output and model are saved in the folder "./results";  
--epochs: number of epochs  
--NL: layer index to train  
--save_model: save the trained layer  
--device_num: device number if using GPU    
--tr_and_eval: test the accuracy at each epoch of training  

* SCFF training of the first layer and save the best model for the next layer train
```
nohup python -u SCFF_MNIST.py --epochs 20 --NL 1 --device_num 0 --save_model > ./results/SCFF_MNIST_l1.txt 2>&1 &
```
* SCFF training of the second layer 
```
nohup python -u SCFF_MNIST.py --epochs 9 --NL 2 --device_num 0  > ./results/SCFF_MNIST_l2.txt 2>&1 &
```

### Run the FSDD experiment
* The FSDD dataset and audio data preprocessing are downloaded from this [repo](https://github.com/aniruddhapal211316/spoken_digit_recognition)  
* The dataset.py file is import in SCFF_FSDD.py for preprocessing the audio input  
* Run the SCFF_FSDD.py file to train different layers; the output and model are saved in the folder "./results";  
--epochs: number of epochs  
--save_model: save the trained layer   
--enable_gpu: enable gpu if needed  
--device_num: device number if using GPU        

* SCFF training of the first layer
```
nohup python -u SCFF_FSDD.py --enable_gpu --device_num 0  > ./results/SCFF_FSDD_l1.txt 2>&1 &
```

## Parallel training of all layers simultaneously


## Authors

Contributors names and contact info

[@XingCHEN](xing.chen@cnrs-thales.fr)



## License


## Citation
@article{chen2024self,
  title={Self-Contrastive Forward-Forward Algorithm},
  author={Chen, Xing and Liu, Dongshu and Laydevant, Jeremie and Grollier, Julie},
  journal={arXiv preprint arXiv:2409.11593},
  year={2024}
}