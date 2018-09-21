import argparse 
import tensorflow as tf

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="./tmp/freeze/freeze.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)

    # We can verify that we can access the list of operations in the graph
    # for op in graph.get_operations():
    #     print(op.name)
        
    x = graph.get_tensor_by_name('prefix/Placeholder:0')
    y = graph.get_tensor_by_name('prefix/dnn/logits/BiasAdd:0')
        
    # We launch a Session
    with tf.Session(graph=graph) as sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants 
        y_out = sess.run(y, feed_dict={
            x: [[0.28378378378378377,1.0,1.0,1.0,0.6666666666666666],
                [0.0,0.08080808080808081,0.0,0.0, 0.0],
                [0.5135135135135135,0.24242424242424243,1.0,0.9565217391304348,1.0],
                [0.0,0.08080808080808081,0.0,0.0,0.0],
                [0.2702702702702703,0.04040404040404041,0.5,0.34782608695652173,0.3333333333333333]]
        })
        
        print(y_out) 