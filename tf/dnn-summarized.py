""" Neural Network.
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).
This example is using TensorFlow layers, see 'neural_network_raw' example for
a raw implementation with variables.
Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

import tensorflow as tf

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import shutil
import time

dataset = np.load('../data/aae-summarized-data.npy').tolist()
X = np.array(dataset['X_data'])
y = np.array(dataset['y_data']).flatten().astype(int)
clw = compute_class_weight('balanced', np.unique(y), y)
skf = StratifiedKFold(n_splits=5, shuffle=True) # stratified 10-fold cross-validation

accs = []
sens = []
specs = []
precs = []
f1 = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    trainingset = {'X_data': X_train.tolist(), 'y_data': y_train.tolist()}
    testset = {'X_data': X_test.tolist(), 'y_data': y_test.tolist()}

    trweights = [clw[yi] for yi in trainingset['y_data']]
    teweights = [clw[yi] for yi in testset['y_data']]
    trweights = np.array(trweights)
    teweights = np.array(teweights)
    # Parameters
    learning_rate = 0.1
    num_steps = 1000
    # batch_size = 128
    # display_step = 100

    # Network Parameters
    n_hidden_1 = 5 # 1st layer number of neurons
    n_hidden_2 = 5 # 2nd layer number of neurons
    num_input = 17 # MNIST data input (img shape: 28*28)
    num_classes = 2 # MNIST total classes (0-9 digits)

    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[num_input])]
    shutil.rmtree('tmp/aae_model-summarized', ignore_errors=True)
    time.sleep(1)

    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[n_hidden_1,n_hidden_2],
                                            n_classes=num_classes,
                                            weight_column='weight',
                                            optimizer=tf.train.AdagradOptimizer(learning_rate=learning_rate),
                                            model_dir="tmp/aae_model-summarized")

    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(trainingset['X_data']), "weight": trweights},
        y=np.array(trainingset['y_data']),
        num_epochs=None,
        shuffle=True)

    # Train model.
    classifier.train(input_fn=train_input_fn, steps=num_steps)

    # # Define the test inputs
    # test_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": np.array(testset['X_data'])},
    #     y=np.array(testset['y_data']),
    #     num_epochs=1,
    #     shuffle=False)

    # # Evaluate accuracy.
    # accuracy_score = classifier.evaluate(input_fn=test_input_fn)
    # print(accuracy_score)

    #print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    # Classify new samples.
    new_samples = np.array(testset['X_data'])
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": new_samples, "weight": teweights},
        num_epochs=1,
        shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))
        # predicted_classes = [p["class_ids"][0] for p in predictions]
    probas = [p['logistic'][0] for p in predictions]
    val_size = len(probas)//2
    prec, rec, thresholds = precision_recall_curve(testset['y_data'][:val_size], probas[:val_size])
    best_thr = thresholds[np.argmax(2/(1/prec + 1/rec))]

    cfm = confusion_matrix(
        testset['y_data'][val_size:],
        (probas[val_size:] >= best_thr).astype(int))
    TN = cfm[0][0]
    FP = cfm[0][1]
    FN = cfm[1][0]
    TP = cfm[1][1]
    P = TP + FN
    N = FP + TN
    accs.append(((TP+FP)/(P+N))*100)
    sens.append(TP/P*100)
    specs.append(TN/N*100)
    precs.append(TP/(TP+FP)*100)
    f1.append(2*TP/(2*TP+FN+FP)*100)
print("Accuracy: {:.2f}".format(np.mean(accs)))
print("Sensitivity: {:.2f}".format(np.mean(sens)))
print("Specificity: {:.2f}".format(np.mean(specs)))
print("Precision: {:.2f}".format(np.mean(precs)))
print("F1-score: {:.2f}".format(np.mean(f1)))
# Export trained model
def serving_input_receiver_fn():
    inputs = {"x": tf.placeholder(shape=[None, num_input], dtype=tf.float32)}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

# classifier.export_savedmodel(export_dir_base="tmp/exported", serving_input_receiver_fn=serving_input_receiver_fn)

# Load saved model and predict inputs
# predict_fn = predictor.from_saved_model("tmp/exported/1514304640", signature_def_key='predict')
# prediction = predict_fn(
#     {'x': [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 5.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.0, 1.0, 1.0],
#     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]]})
# print("predictions",prediction['class_ids'])

# # Define the neural network
# def neural_net(x_dict):
#     # TF Estimator input is a dict, in case of multiple inputs
#     x = x_dict['forms']
#     # Hidden fully connected layer with 256 neurons
#     layer_1 = tf.layers.dense(x, n_hidden_1)
#     # Hidden fully connected layer with 256 neurons
#     layer_2 = tf.layers.dense(layer_1, n_hidden_2)
#     # Output fully connected layer with a neuron for each class
#     out_layer = tf.layers.dense(layer_2, num_classes)
#     return out_layer
 
# # Define the model function (following TF Estimator Template)
# def model_fn(features, labels, mode):
#     # Build the neural network
#     logits = neural_net(features)

#     # Predictions
#     pred_classes = tf.argmax(logits, axis=1)
#     pred_probas = tf.nn.softmax(logits)

#     # If prediction mode, early return
#     if mode == tf.estimator.ModeKeys.PREDICT:
#         return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

#         # Define loss and optimizer
#     loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
#         logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#     train_op = optimizer.minimize(loss_op,
#                                   global_step=tf.train.get_global_step())

#     # Evaluate the accuracy of the model
#     acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

#     # TF Estimators requires to return a EstimatorSpec, that specify
#     # the different ops for training, evaluating, ...
#     estim_specs = tf.estimator.EstimatorSpec(
#         mode=mode,
#         predictions=pred_classes,
#         loss=loss_op,
#         train_op=train_op,
#         eval_metric_ops={'accuracy': acc_op})

#     return estim_specs

# # Build the Estimator
# model = tf.estimator.Estimator(model_fn)

# # Define the input function for training
# input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={'forms': np.asarray(dataset['X_data'])}, y=np.asarray(dataset['y_data']),
#     batch_size=batch_size, num_epochs=None, shuffle=True)
# # Train the Model
# model.train(input_fn, steps=num_steps)

# # Evaluate the Model
# # Define the input function for evaluating
# input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={'forms': np.asarray(dataset['X_data'])}, y=np.asarray(dataset['y_data']),
#     batch_size=batch_size, shuffle=False)
# # Use the Estimator 'evaluate' method
# e = model.evaluate(input_fn)

# print("Testing Accuracy:", e['accuracy'])