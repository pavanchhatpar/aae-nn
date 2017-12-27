from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from grpc.beta import implementations
import tensorflow as tf

host = "0.0.0.0"
port = "9000"

channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

examples = [
    tf.train.Example(
      features=tf.train.Features(
        feature={"x": tf.train.Feature(
          float_list=tf.train.FloatList(value=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 5.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.0, 1.0, 1.0]))})),
    tf.train.Example(
      features=tf.train.Features(
        feature={"x": tf.train.Feature(
          float_list=tf.train.FloatList(value=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]))})),
  ]

# Build the RPC request.
predict_request = predict_pb2.PredictRequest()
predict_request.model_spec.name = "default"
predict_request.inputs["examples"].CopyFrom(
    tf.contrib.util.make_tensor_proto(examples, tf.float32))

# Perform the actual prediction.
result = stub.Predict(predict_request, PREDICT_DEADLINE_SECS)