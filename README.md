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
## Preparing data
```
(virt-env)$ python json_to_mat.py
```

## Octave
```
>> aaeNN
```

## TensorFlow

### Training
```
(virt-env)$ python train.py
```

### Predict with Python
Change model path accordingly in `predict.py`
```
(virt-env)$ python predict.py
```

### Predict with Java
- JNI is already installed, in case of manual installation refer to `jni_installer.sh`
- TensorFlow jar file is downloaded already, latest can be found otherwise through Google
- Change model path accordingly in `Predict.java`
```
$ javac -cp libtensorflow-1.4.1.jar Predict.java

$ java -cp libtensorflow-1.4.1.jar:. -Djava.library.path=./jni Predict
```

### Using TensorFlow Model Server
- Change `ServingInputReceiver Function` in `train.py` and use the currently commented one
- Change export dir if you want to at `train.py:121`
- Train model as shown above
- Start serving the model on port 9000. Change `model_base_path` accordingly. Be sure to use **absolute path**
```
$ tensorflow_model_server --port=9000 --model_base_path=/home/pavan/Projects/aae-nn/tmp/serving_exported
```
- Switch to virt-env2 to use Python2 as tensorflow-serving-api is supported only on Python2
- Run example prediction (TODO: not working yet)
```
(virt-env2)$ python predict_serving.py 
```