""" Ensemble Level LSTM Model """
from __future__ import print_function
import numpy as np
import tensorflow as tf

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

def encode_ensemble_gestures(gestures):
    """Encode multiple natural numbers into one"""
    encoded = 0
    for i, g in enumerate(gestures):
        encoded += g * (len(GESTURE_CODES) ** i)
    return encoded
        
def decode_ensemble_gestures(num_perfs,code):
    """Decodes ensemble gestures from a single int"""
    gestures = []
    for i in range(num_perfs):
        part = code % (len(GESTURE_CODES) ** (i+1))
        gestures.append(part / (len(GESTURE_CODES) ** i))
    return gestures

## Evaluating Network
MODEL_DIR = "/Users/charles/src/ensemble-performance-deep-models/"
MODEL_NAME = MODEL_DIR + "gesture-rnn-model-4to3"

class GestureRNNMeta:
    def __init__(self, sess):
        """Load the meta graph and the model."""
        self.num_input_performers = 4
        self.num_output_performers = 3
        self.num_input_classes = self.num_classes ** self.num_input_performers
        self.num_output_classes = self.num_classes ** self.num_output_performers
        self.sess = sess
        self.graph = tf.get_default_graph()
        self.saver = tf.train.import_meta_graph(MODEL_NAME + '.meta')
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, MODEL_NAME)
        self.state = None
        # Retrieve tensors from model.
        self.x = graph.get_operation_by_name('input/input_placeholder').outputs[0]
        self.predictions = graph.get_operation_by_name('predictions').outputs[0]
        self.init_state = graph.get_operation_by_name('init_state').outputs[0]
        self.final_state = graph.get_operation_by_name('final_state').outputs[0]

    def generate_gestures(self,lead_player,prev_ensemble,sess):
        """ 
        Evaluates the network once for a lead player and previous ensemble gestures.
        Returns the current ensemble gestures. The network state is preserved in between
        evaluations.
        """
        gesture_inputs = list(prev_ensemble)
        gesture_inputs.insert(0,lead_player)
        print("GestureRNN inputs are:",gesture_inputs)
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
