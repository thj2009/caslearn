# **caslearn**
# casLearn is a collection of machine learning algorithms written in casadi. For now, three ML algorithm has been implemented: polynoamil chaos expansion (PCE), Gaussian process (GP), and neural network (NN)


## **Installation**

Install python 3.7 on your system. Recommandation: python with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

In conda command promp, 


```
conda create -n your_env python=3.7
```
activate the environment
```
conda activate your_env
```
install dependent packages
```
conda install numpy matplotlib
```

---
### install casadi 3 in the environment
The easiest way is to use conda to install casadi in the environment
```
conda install casadi
```
or through pip
```
pip install casadi
```
or you could download the lastest casadi package from https://github.com/casadi/casadi/releases/tag/3.5.1, and then unzip in your machine

---
### install MesTool
```
conda activate your_env
conda develop caslearn
```

---


## **Try the example in /example/** 

1. "build_nn.py" is an example about setting up single task neural network. Try your own example and datasets
2. "build_nn_multipleout.py" is an example about setting up multi-task neural network




