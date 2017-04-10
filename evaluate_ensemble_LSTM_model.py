""" Evaluate the Ensemble Level LSTM Model """
from __future__ import print_function
import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import os
import datetime
import pickle
from urllib import urlretrieve
import random
from itertools import permutations
        
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
num_input_performers = 4
num_output_performers = 3

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

def reset_graph():
    """Resets Tensorflow's computation graph."""
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

## Training Network
## Hyperparameters for training
num_nodes = 512 # tried could be 64--512
num_classes = vocabulary_size

num_input_classes = vocabulary_size ** num_input_performers
num_output_classes = vocabulary_size ** num_output_performers
batch_size = 64
num_steps = 120
num_layers = 3
learning_rate = 1e-4

# ### Generation Network
# 
# - Reload the model with new hyper-parameters to evaluate one step at a time.
# - Need to choose appropriate tfsave to load.

## Evaluating Network
model_dir = "/Users/charles/src/ensemble-performance-deep-models/"
model_name = model_dir + "quartet-lstm-model-512-5-epochs.tfsave"

batch_size = 1
num_steps = 1

## Reload the graph
reset_graph()
graph = tf.get_default_graph()
with graph.as_default():
    x = tf.placeholder(tf.int32,[batch_size,num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32,[batch_size,num_steps], name='labels_placeholder')
    embeddings = tf.get_variable('embedding_matrix', [num_input_classes, num_nodes])
    out_embedding = tf.get_variable('out_emedding_matrix',[num_output_classes,num_nodes])
    rnn_inputs = tf.nn.embedding_lookup(embeddings,x)
    
    # Define the network
    cell = tf.contrib.rnn.LSTMCell(num_nodes,state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    init_state = cell.zero_state(batch_size,tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W',[num_nodes,num_output_classes])
        b = tf.get_variable('b',[num_output_classes], initializer=tf.constant_initializer(0.0))
    
    rnn_outputs = tf.reshape(rnn_outputs,[-1,num_nodes])
    y_reshaped = tf.reshape(y,[-1])
    logits = tf.matmul(rnn_outputs, W) + b
    predictions = tf.nn.softmax(logits)

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    saver = tf.train.Saver()
    print("Graph Specs:")
    print("Num Input Classes: " + str(num_input_classes))
    print("Num Output Classes: " + str(num_output_classes))
    print("Inputs: " + str(rnn_inputs))
    print("Outputs: " + str(rnn_outputs))
    print("Logits: " + str(logits))
    print("Predictions: " + str(predictions))

##
# ## Code to test network on one input gesture at a time.
current_player_gesture = 0
output_gestures = decode_ensemble_gestures(num_output_performers,0)
state = None

## This def sort of works.
## Start Generating Output:
def generate_gesture_for_current_and_prev_ensemble_given_state(lead_player,prev_ensemble,state):
    """
    Evaluates the network once for a lead player amd previous ensemble gestures, and network state.
    Returns the current ensemble gestures and network state.
    """
    with tf.Session() as sess:
        # Setup the model
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./"+model_name)
        # Apply the 
        gesture_inputs = list(prev_ensemble)
        gesture_inputs.insert(0,lead_player)
        print("Inputs are:",gesture_inputs)
        if state is not None:
            feed_dict = {x: [[encode_ensemble_gestures(gesture_inputs)]], init_state: state}
        else:
            feed_dict = {x: [[encode_ensemble_gestures(gesture_inputs)]]}
        preds,state = sess.run([predictions,final_state],feed_dict=feed_dict)
        output_step = np.random.choice(num_output_classes,1,p=np.squeeze(preds))[0] # choose the output step
        output_gestures = decode_ensemble_gestures(num_output_performers,output_step)
        return output_gestures,state    

## Try Evaluating the model for one input gesture
current_player_gesture = 5
output_gestures, state = generate_gesture_for_current_and_prev_ensemble_given_state(current_player_gesture,output_gestures,state)
print(output_gestures)

