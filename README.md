# Self-Contrastive-Forward-Forward
This repo implements the official code of the article: ["Self-Contrastive Forward-Forward Algorithm"](http://arxiv.org/abs/2409.11593)
## Description

An in-depth paragraph about your project and overview of use.

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
* The dataset.py file is import in ContrastFF_FSDD.py for preprocessing  
* Run the ContrastFF_FSDD.py file to train different layers; the output and model are saved in the folder "./results";  
--epochs: number of epochs  
--save_model: save the trained layer   
--enable_gpu: enable gpu if needed  
--device_num: device number if using GPU        

### SCFF training of the first layer
```
nohup python -u ContrastFF_FSDD.py --epochs 20 --NL 1 --device_num 0  > ./results/ContrastFF_FSDD_l1.txt 2>&1 &
```

## Authors

Contributors names and contact info

[@XingCHEN](xing.chen@cnrs-thales.fr)



## License


## Acknowledgments
*[Speech Recognition on Spoken Digit Dataset](https://github.com/aniruddhapal211316/spoken_digit_recognition)