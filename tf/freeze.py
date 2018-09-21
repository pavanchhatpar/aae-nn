from tensorflow.python.tools import freeze_graph

freeze_graph.freeze_graph(
    input_graph='',input_saver='',input_binary=True,input_checkpoint='',
    restore_op_name='',filename_tensor_name='',clear_devices='',initializer_nodes='',
    input_saved_model_dir='./tmp/exported/1521950031',
    output_graph = './tmp/freeze/freeze.pb',
    output_node_names='dnn/head/predictions/classes'
)