import json
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


DATA_PATH = "/home/maedd/Documents/Bachelorarbeit/Network/Files/data.json"
SAVED_MODEL_PATH = "/home/maedd/Documents/Bachelorarbeit/Network/Files/SaveTest.h5"

LEARNING_RATE = 0.0001
EPOCHS = 40
BATCH_SIZE = 32
NUM_KEYWORDS = 2

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ''
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

def load_dataset(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # extract inputs and labels
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])

    #X = X[0, :, :]
    print(X.ndim)

    print(X.shape)

    return X, y

def get_data_splits(data_path, test_size=0.1, test_validation=0.1):

    # load dataset
    X, y = load_dataset(data_path)

    # create train/validation/test splits
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,
                                                                    test_size=test_validation)


    #convert inputs from 2d to 3d arrays (segments, 13)
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    print(X_train.ndim)

    return X_train, y_train, X_validation, y_validation, X_test, y_test

def build_model(input_shape, learning_rate, error="sparse_categorical_crossentropy"):

    #build network
    model = keras.Sequential()

    #conv layer 1
    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu",
                                  input_shape=input_shape,
                                  kernel_regularizer=keras.regularizers.l2(0.001), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    #conv layer 2
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    #conv layer 3
    model.add(keras.layers.Conv2D(32, (2, 2), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))



    # flatten the output and feed it into a dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    # softmax classifier
    model.add(keras.layers.Dense(NUM_KEYWORDS, activation="softmax")) #[output e.g = 0.1, 0.9]

    # compile
    optimiser = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimiser, loss=error, metrics=["accuracy"])

    # print model overview
    model.summary()

    return model

def main():

    # load train/validation/test data splits
    X_train, y_train, X_validation, y_validation, X_test, y_test = get_data_splits(DATA_PATH)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) # (segments, coefficients13, 1)

    model = build_model(input_shape, learning_rate=LEARNING_RATE)

    print("done")

    # train model
    model.fit(X_train, y_train, validation_data=(X_validation, y_validation),
              epochs=EPOCHS, batch_size=BATCH_SIZE) # epochs = pass trainData 40 times through the network

    # evaluate the model
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print("\nTest loss: {}, test accuracy: {}".format(test_error, 100*test_accuracy))


    # inputs:  ['dense_input'] (not sure about all of that yet)
    print('inputs: ', [input.op.name for input in model.inputs])

    # outputs:  ['dense_4/Sigmoid']
    print('outputs: ', [output.op.name for output in model.outputs])

    # save the model
    model.save(SAVED_MODEL_PATH)

    #save model to frozen protobuf
    frozen_graph = freeze_session(tf.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, './', 'xor.pbtxt', as_text=True)
    tf.train.write_graph(frozen_graph, './', 'xor.pb', as_text=False)

if __name__ == "__main__":
    main()