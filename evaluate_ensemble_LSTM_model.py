""" Evaluate the Ensemble Level LSTM Model """
from __future__ import print_function
import numpy as np
import tensorflow as tf
import pickle
        
## Int values for Gesture codes.
NUMBER_GESTURES = 9
GESTURE_CODES = {
    'N': 0,
    'FT': 1,
    'ST': 2,
    'FS': 3,
    'FSA': 4,
    'VSS': 5,
    'BS': 6,
    'SS': 7,
    'C': 8}

vocabulary_size = len(GESTURE_CODES)

def encode_ensemble_gestures(gestures):
    """Encode multiple natural numbers into one"""
    encoded = 0
    for i, g in enumerate(gestures):
        encoded += g * (vocabulary_size ** i)
    return encoded
        
def decode_ensemble_gestures(num_perfs,code):
    """Decodes ensemble gestures from a single int"""
    gestures = []
    for i in range(num_perfs):
        part = code % (vocabulary_size ** (i+1))
        gestures.append(part / (vocabulary_size ** i))
    return gestures


## Evaluating Network
MODEL_DIR = "/Users/charles/src/ensemble-performance-deep-models/"
MODEL_NAME = MODEL_DIR + "quartet-lstm-model-512-30-epochs.tfsave"

class EnsembleLSTMNetwork:

    def reset_graph(self):
        """Resets Tensorflow's computation graph."""
        if 'sess' in globals() and sess:
            sess.close()
        tf.reset_default_graph()


    def __init__(self):
        ## Training Network
        ## Hyperparameters for training
        num_nodes = 512 # tried could be 64--512
        self.num_input_performers = 4
        self.num_output_performers = 3

        self.num_classes = vocabulary_size
        self.num_input_classes = vocabulary_size ** self.num_input_performers
        self.num_output_classes = vocabulary_size ** self.num_output_performers
        batch_size = 64
        num_steps = 120
        num_layers = 3
        learning_rate = 1e-4
        batch_size = 1
        num_steps = 1
        self.state = None
    
        ## Reload the graph
        self.reset_graph()
        self.graph = tf.get_default_graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.int32,[batch_size,num_steps], name='input_placeholder')
            self.y = tf.placeholder(tf.int32,[batch_size,num_steps], name='labels_placeholder')
            self.embeddings = tf.get_variable('embedding_matrix', [self.num_input_classes, num_nodes])
            self.out_embedding = tf.get_variable('out_emedding_matrix',[self.num_output_classes,num_nodes])
            self.rnn_inputs = tf.nn.embedding_lookup(self.embeddings,self.x)    
            # Define the network
            self.cell = tf.contrib.rnn.LSTMCell(num_nodes,state_is_tuple=True)
            self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * num_layers, state_is_tuple=True)
            self.init_state = self.cell.zero_state(batch_size,tf.float32)
            self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(self.cell, self.rnn_inputs, initial_state=self.init_state)
            with tf.variable_scope('softmax'):
                self.W = tf.get_variable('W',[num_nodes,self.num_output_classes])
                self.b = tf.get_variable('b',[self.num_output_classes], initializer=tf.constant_initializer(0.0))
            self.rnn_outputs = tf.reshape(self.rnn_outputs,[-1,num_nodes])
            self.y_reshaped = tf.reshape(self.y,[-1])
            self.logits = tf.matmul(self.rnn_outputs, self.W) + self.b
            self.predictions = tf.nn.softmax(self.logits)
            self.total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_reshaped))
            self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.total_loss)
            self.saver = tf.train.Saver()

    def describe_graph():
        print("Network Specs:")
        print("Num Input Classes: " + str(self.num_input_classes))
        print("Num Output Classes: " + str(self.num_output_classes))
        print("Inputs: " + str(self.rnn_inputs))
        print("Outputs: " + str(self.rnn_outputs))
        print("Logits: " + str(self.logits))
        print("Predictions: " + str(self.predictions))

    def generate_gestures(self,lead_player,prev_ensemble):
        """
        Evaluates the network once for a lead player amd previous ensemble gestures, and network state.
        Returns the current ensemble gestures and network state.
        """
        with tf.Session() as sess:
            # Setup the model
            #print("Model:",MODEL_NAME)
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, MODEL_NAME)
            # Apply the inputs
            gesture_inputs = list(prev_ensemble)
            gesture_inputs.insert(0,lead_player)
            print("LSTM inputs are:",gesture_inputs)
            #print("State:",self.state)
            if self.state is not None:
                feed_dict = {self.x: [[encode_ensemble_gestures(gesture_inputs)]], self.init_state: self.state}
            else:
                feed_dict = {self.x: [[encode_ensemble_gestures(gesture_inputs)]]}
            preds,self.state = sess.run([self.predictions,self.final_state],feed_dict=feed_dict)
            output_step = np.random.choice(self.num_output_classes,1,p=np.squeeze(preds))[0] # choose the output step
            output_gestures = decode_ensemble_gestures(self.num_output_performers,output_step)
            return output_gestures    


##
# Testing code:
# network = EnsembleLSTMNetwork()
# output_gestures = [0,0,0]
# current_player_gesture = 5
# output_gestures = network.generate_gestures(current_player_gesture,output_gestures)
# print(output_gestures)
# output_gestures = network.generate_gestures(current_player_gesture,output_gestures)
# print(output_gestures)
