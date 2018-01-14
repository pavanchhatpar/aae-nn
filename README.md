# Endodontic Case Difficulty Assessment using Artificial Neural Networks
This is a part of a research paper dealing with predicting the difficulty of an Endodontic case.

## Setup

### Virtual Environment (Optional but Recommended)
```
$ virtualenv virt-env                       #For default python (python 3)

$ virtualenv virt-env2 --python=python2.7   #For Python 2

$ source virt-env/bin/activate              #Start py3 virt-env

(virt-env)$ deactivate                     #Stop py2 virt-env

$ source virt-env2/bin/activate             #Start py2 virt-env2

(virt-env2)$ deactivate                     #Stop py2 virt-env2
```
### Install pip packages
```
(virt-env)$ pip install -r requirements.txt

(virt-env2)$ pip install -r requirements2.txt
```
### Install apt packages
This package enables us to use `tensorflow_model_server` via command line
```
$ sudo apt-get update | sudo apt-get install tensorflow-model-server
```