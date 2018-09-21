# Version 1
Treating each question as one input giving 17 input features

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

### Freeze model and predict using freezed model
- Freeze using
```
(virt-env)$ python freeze.py
```
- Predict with the freezed model using
```
(virt-env)$ python freeze_predict.py
```
- Make sure to put proper input filenames to the freeze function and load proper frozen graph while predicting