import tensorflow as tf
from tensorflow.python.platform import gfile

#gives output nodes of frozen model
def input_output_node(input_Graph):
    GRAPH_PB_PATH = input_Graph
    with tf.Session() as sess:
        print("load graph")
        with gfile.FastGFile(GRAPH_PB_PATH, 'rb') as f:
            graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        graph_nodes = [n for n in graph_def.node]
        names = []
        for t in graph_nodes:
            names.append(t.name)
        print(names)

#input_output_node('/home/maedd/Documents/Bachelorarbeit/Network/xor.pb')
input_output_node('/home/maedd/Documents/ADT_training/adt-training/data/experiments/1603810436spec/musicnn_drums_librosa.pb')



