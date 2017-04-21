"""Gesture-RNN model for simulating ensemble interaction on touch-screens."""
from __future__ import print_function
import numpy as np
import tensorflow as tf
import pickle
import time
import os
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

## Evaluating Network
MODEL_DIR = "/Users/charles/src/ensemble-performance-deep-models/"
MODEL_NAME = MODEL_DIR + "quartet-lstm-model-512-30-epochs.tfsave"
LOG_PATH = "/tmp/tensorflow-logs/"

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
		part = code % (vocabulary_size ** (i+1))
		gestures.append(part / (len(GESTURE_CODES) ** i))
	return gestures

class QuartetDataManager(object):
	"""Manages data from metatone quartet performances and generates epochs"""

	def __init__(self, num_steps, batch_size):
		"""Load Metatone Corpus and Create Example Data"""
		self.num_steps = num_steps
		self.batch_size = batch_size
		self.examples_file = "MetatoneQuartetExamples-" + str(self.num_steps) + "steps-" + str(self.batch_size) + "batch" + ".pickle"

		## Make sure corpus is available.
		URL = "https://github.com/anucc/metatone-analysis/raw/master/metadata/"
		PICKLE_FILE = "metatone_performances_dataframe.pickle"
		if not os.path.exists(PICKLE_FILE):
			urlretrieve(URL + PICKLE_FILE, PICKLE_FILE)
		with open(PICKLE_FILE, 'rb') as f:
				metatone_dataset = pickle.load(f)

		### Load Quartet Improvisations.
		improvisations = metatone_dataset[
			(metatone_dataset["performance_type"] == "improvisation") &
			(metatone_dataset["performance_context"] != "demonstration") &
			(metatone_dataset["number_performers"] == 4)]
		gesture_data = improvisations['gestures']
		self.ensemble_improvisations = gesture_data.tolist()
		print("Number of performances in training data: ", len(self.ensemble_improvisations))
		print("Attempting to load",self.examples_file)
		if os.path.exists(self.examples_file):
			with open(self.examples_file, 'rb') as e:
				self.dataset = pickle.load(e)
		else:
			self.dataset = self.setup_training_examples()
			with open(self.examples_file, 'wb') as e:
				pickle.dump(self.dataset,e)
		print("Loaded", str(len(self.dataset)), "Training Examples.")



	def setup_test_data(self):
		"""Load individual parts of non-trained data for testing."""
		improvisations = metatone_dataset[
			(metatone_dataset["performance_type"] == "improvisation") &
			(metatone_dataset["performance_context"] != "demonstration") &
			(metatone_dataset["number_performers"] != 4)]
		gesture_data = improvisations['gestures']
		self.individual_improvisations = []
		for perf in gesture_data.tolist():
			for one_perf in perf.T:
				self.individual_improvisations.append(one_perf)

	def setup_training_examples(self):
		"""Setup training examples from corpus."""
		imp_xs = []
		imp_ys = []
		for imp in self.ensemble_improvisations:
			print("Processing performance data.")
			for i in range(len(imp)-self.num_steps-1):
				imp_slice = imp[i:i+self.num_steps+1]
				for j in range(len(imp_slice.T)):
					lead = imp_slice[1:].T[j] # lead gestures (post steps)
					ensemble = imp_slice.T[np.arange(len(imp_slice.T)) != j] # rest of the players indexed by player
					for ens_perm in permutations(ensemble): # consider all permutations of the players
						ens_pre = np.array(ens_perm).T[:-1] # indexed by time slice
						ens_post = np.array(ens_perm).T[1:] # indexed by time slice
						y = map(encode_ensemble_gestures,ens_post)
						#y = ens_post # test just show the gestures
						x = map(encode_ensemble_gestures,zip(lead,*(ens_pre.T))) # encode ensemble state
						#x = zip(lead,*(ens_pre.T)) # test just show the gestures
						imp_xs.append(x) # append the inputs
						imp_ys.append(y) # append the outputs
		print("Total Training Examples: " + str(len(imp_xs)))
		print("Total Training Labels: " + str(len(imp_ys)))
		return zip(imp_xs,imp_ys)

	def next_epoch(self):
		"""Return an epoch of batches of shuffled examples."""
		np.random.shuffle(self.dataset)
		dataset_size = len(self.dataset)
		batches = []
		for i in range(dataset_size / self.batch_size):
			batch = self.dataset[i*self.batch_size:(i+1)*self.batch_size]
			bx,by = zip(*batch)
			batches.append((np.array(bx),np.array(by)))
		return(batches)


class GestureRNN(object):
	def __init__(self):
		"""initialize GestureRNN model"""
		self.vocabulary_size = len(GESTURE_CODES)
		## Model Hyperparameters
		num_nodes = 512
		num_layers = 3

		## IO Hyperparameters
		self.num_input_performers = 4
		self.num_output_performers = 3
		self.num_classes = self.vocabulary_size
		self.num_input_classes = self.vocabulary_size ** self.num_input_performers
		self.num_output_classes = self.vocabulary_size ** self.num_output_performers
		
		# Training Hyperparamters
		self.batch_size = 64
		num_steps = 120
		learning_rate = 1e-4

		# Running Hyperparameters
		batch_size = 1
		num_steps = 1
		self.state = None
	
		## Load the graph
		tf.reset_default_graph()
		self.graph = tf.get_default_graph()
		with self.graph.as_default():
			with tf.name_scope('input'):
				self.x = tf.placeholder(tf.int32,[batch_size,num_steps], name='input_placeholder')
				self.y = tf.placeholder(tf.int32,[batch_size,num_steps], name='labels_placeholder')
			self.embeddings = tf.get_variable('embedding_matrix', [self.num_input_classes, num_nodes])
			self.out_embedding = tf.get_variable('out_emedding_matrix',[self.num_output_classes,num_nodes])
			self.rnn_inputs = tf.nn.embedding_lookup(self.embeddings,self.x, name="input_embedding")    
			# RNN section
			self.cell = tf.contrib.rnn.LSTMCell(num_nodes,state_is_tuple=True)
			self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * num_layers, state_is_tuple=True)
			self.init_state = self.cell.zero_state(batch_size,tf.float32)
			self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(self.cell, self.rnn_inputs, initial_state=self.init_state)
			# Output section
			with tf.variable_scope('softmax'):
				self.W = tf.get_variable('W',[num_nodes,self.num_output_classes])
				self.b = tf.get_variable('b',[self.num_output_classes], initializer=tf.constant_initializer(0.0))
			self.rnn_outputs = tf.reshape(self.rnn_outputs,[-1,num_nodes], name = "reshape_rnn_outputs")
			self.y_reshaped = tf.reshape(self.y,[-1], name = "reshape_labels")
			self.logits = tf.matmul(self.rnn_outputs, self.W, name = "logits_mul") + self.b
			self.saver = tf.train.Saver()
		self.writer = tf.summary.FileWriter(LOG_PATH, graph=self.graph)

	def predict(self, x):
		""" Return prediction operator"""
		return tf.nn.softmax(self.logits, name="prediction")


	def loss(self, batch_x):
		y_predict = self.predict(batch_x)
		return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_reshaped, name="cross_entropy"), name="loss")


	def optimize(self, batch_x, batch_y):
		return tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

	def model_name():
		"""Returns the name of the present model for saving to disk"""
		return "gesture-rnn-model-" + str(self.num_input_performers) + "to" + str(self.num_output_performers)

	def train_epoch(index, batches):
		"""Code for training one epoch of training data"""
		training_loss = 0
		steps = 0
		training_state = None
		print("Starting Epoch " + str(i) + " of " + str(num_epochs))
		for batch_data, batch_labels in epoch:
			self.train_batch(batch_data,batch_labels)
		print("Trained Epoch " + str(i) + " of " + str(num_epochs))
		training_losses.append(training_loss/steps)


	def train_batch(batch_x,batch_y):
		feed = {self.x: batch_x, self.y: batch_y}
		if training_state is not None:
			feed[init_state] = training_state
		training_loss_current, training_state, _ = sess.run([total_loss,final_state,train_step],feed_dict=feed)
		steps += 1
		training_loss += training_loss_current
		if (steps % 2000 == 0): 
			print("Trained batch: " + str(steps) + " of " + str(len(epoch)) + " loss was: " + str(training_loss_current))


	def train(epochs):
		"""Train the network for the a number of epochs."""
		num_epochs = 30
		tf.set_random_seed(2345) # should this be removed?
		print("Going to train: " + self.model_name())
		start_time = time.time()
		self.training_losses = []
		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())
			for i, epoch in enumerate(generate_epochs(num_epochs, num_steps, batch_size)):
				self.train_epoch(i,epoch)
			self.saver.save(sess,model_name)
		print("It took ", time.time() - start_time, " to train the network.")

