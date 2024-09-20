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



## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)