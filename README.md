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
conda create --name cff python=3.10.9
```
* Activate the environment
```
conda activate cff
```
* Install the Dependencies
```
pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

## Run the CIFAR-10 experiment

* Run the ContrastFF_CIFAR.py file to train different layers; the output and model are saved in the folder "./results"  
--epochs: number of epochs  
--NL: layer index to train  
--save_model: save the trained layer  
--device_num: device number if using GPU    
--tr_and_eval: test the accuracy at each epoch of training  

### SCFF training of the first layer and save the best model for the next layer train
```
nohup python -u ContrastFF_CIFAR.py --epochs 6 --NL 1 --device_num 0 --save_model > ./results/ContrastFF_CIFAR_l1.txt 2>&1 &
```
### SCFF training of the second layer and save the best model for the next layer train
```
nohup python -u ContrastFF_CIFAR.py --epochs 4 --NL 2 --device_num 0 --save_model > ./results/ContrastFF_CIFAR_l2.txt 2>&1 &
```
### SCFF training of the third layer
```
nohup python -u ContrastFF_CIFAR.py --epochs 21 --NL 3 --device_num 0  > ./results/ContrastFF_CIFAR_l3.txt 2>&1 &
```

## Run the STL-10 experiment

* Run the ContrastFF_STL.py file to train different layers; the output and model are saved in the folder "./results";  
--epochs: number of epochs  
--NL: layer index to train  
--save_model: save the trained layer  
--device_num: device number if using GPU    
--tr_and_eval: test the accuracy at each epoch of training  

### SCFF training of the first layer and save the best model for the next layer train
```
nohup python -u ContrastFF_STL.py --epochs 4 --NL 1 --device_num 0 --save_model > ./results/ContrastFF_STL_l1.txt 2>&1 &
```
### SCFF training of the second layer and save the best model for the next layer train
```
nohup python -u ContrastFF_STL.py --epochs 5 --NL 2 --device_num 0 --save_model > ./results/ContrastFF_STL_l2.txt 2>&1 &
```
### SCFF training of the third layer and save the best model for the next layer train
```
nohup python -u ContrastFF_STL.py --epochs 12 --NL 3 --device_num 0 --save_model > ./results/ContrastFF_STL_l3.txt 2>&1 &
```
### SCFF training of the fourth layer
```
nohup python -u ContrastFF_STL.py --epochs 12 --NL 4 --device_num 0 > ./results/ContrastFF_STL_l4.txt 2>&1 &
```

## Run the MNIST experiment
* Run the ContrastFF_MNIST.py file to train different layers; the output and model are saved in the folder "./results";  
--epochs: number of epochs  
--NL: layer index to train  
--save_model: save the trained layer  
--device_num: device number if using GPU    
--tr_and_eval: test the accuracy at each epoch of training  

### SCFF training of the first layer and save the best model for the next layer train
```
nohup python -u ContrastFF_MNIST.py --epochs 20 --NL 1 --device_num 0 --save_model > ./results/ContrastFF_MNIST_l1.txt 2>&1 &
```
### SCFF training of the second layer 
```
nohup python -u ContrastFF_MNIST.py --epochs 9 --NL 2 --device_num 0  > ./results/ContrastFF_MNIST_l2.txt 2>&1 &
```

## Run the FSDD experiment
* The FSDD dataset and audio data preprocessing are downloaded from this [repo](https://github.com/aniruddhapal211316/spoken_digit_recognition)  
* The dataset.py file is import in ContrastFF_FSDD.py for preprocessing the audio input  
* Run the ContrastFF_FSDD.py file to train different layers; the output and model are saved in the folder "./results";  
--epochs: number of epochs  
--save_model: save the trained layer   
--enable_gpu: enable gpu if needed  
--device_num: device number if using GPU        

### SCFF training of the first layer
```
nohup python -u ContrastFF_FSDD.py --enable_gpu --device_num 0  > ./results/ContrastFF_FSDD_l1.txt 2>&1 &
```

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