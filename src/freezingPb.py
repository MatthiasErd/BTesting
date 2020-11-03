import tensorflow as tf
from tensorflow.python.framework import graph_util

experiment_folder = '/home/maedd/Documents/ADT_training/adt-training/data/experiments/1604402355spec'

# tensorflow: loading model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

model = experiment_folder + '/.meta'

graph = tf.get_default_graph()
#saver = tf.train.Saver()
saver = tf.train.import_meta_graph(model, clear_devices=True)

# model checkpoints
saver.restore(sess, experiment_folder +'/')

#gd = sess.graph.as_graph_def()
gd = graph.as_graph_def()

for node in gd.node:
    if node.op == 'RefSwitch':
        node.op = 'Switch'
        for index in range(len(node.input)):
            if 'moving_' in node.input[index]:
                node.input[index] = node.input[index] + '/read'
    elif node.op == 'AssignSub':
        node.op = 'Sub'
        if 'use_locking' in node.attr: del node.attr['use_locking']
    elif node.op == 'AssignAdd':
        node.op = 'Add'
        if 'use_locking' in node.attr: del node.attr['use_locking']
    elif node.op == 'Assign':
        node.op = 'Identity'
        if 'use_locking' in node.attr: del node.attr['use_locking']
        if 'validate_shape' in node.attr: del node.attr['validate_shape']
        if len(node.input) == 2:
            # input0: ref: Should be from a Variable node. May be uninitialized.
            # input1: value: The value to be assigned to the variable.
            node.input[0] = node.input[1]
            del node.input[1]

node_names =[n.name for n in gd.node if 'model' in n.name]

subgraph = tf.graph_util.extract_sub_graph(gd, node_names)
tf.reset_default_graph()
tf.import_graph_def(subgraph)

output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, # The session is used to retrieve the weights
        gd, # The graph_def is used to retrieve the nodes
        node_names  #.split(",")   # The output node names are used to select the usefull nodes
    )
tf.io.write_graph(output_graph_def, experiment_folder + '/', 'musicnn_drums_librosa.pb', as_text=False)
sess.close()