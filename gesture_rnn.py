"""
Gesture-RNN model for simulating ensemble interaction on touch-screens.
"""
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import tensorflow as tf
import h5py
import pickle
import time
import os
from urllib import urlretrieve
from itertools import permutations
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Int values for Gesture codes.
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

# Evaluating Network
MODEL_DIR = ""
LOG_PATH = "output-logs/"  # "/tmp/tensorflow/"
NP_RANDOM_STATE = 6789
TF_RANDOM_STATE = 2345
SPLIT_RANDOM_STATE = 2468

# Flags
tf.app.flags.DEFINE_boolean("train", False, "Train the network and save the model.")
tf.app.flags.DEFINE_integer("epochs", 30, "Number of epochs to train for (default 30).")
tf.app.flags.DEFINE_boolean("generate", False, "Generate some sample test output.")
tf.app.flags.DEFINE_integer("num_perfs", 10, "Number of sample performances to generate.")
tf.app.flags.DEFINE_boolean("test_eval", False, "Test generation of a few performance steps.")
tf.app.flags.DEFINE_boolean("test_train", False, "Test training of two epochs (without saving the model).")
FLAGS = tf.app.flags.FLAGS


def encode_ensemble_gestures(gestures):
    """Encode multiple natural numbers into one"""
    encoded = 0
    for i, g in enumerate(gestures):
        encoded += g * (len(GESTURE_CODES) ** i)
    return encoded


def decode_ensemble_gestures(num_perfs, code):
    """Decodes ensemble gestures from a single int"""
    # TODO: Check that this works correctly now.
    gestures = []
    for i in range(num_perfs):
        part = code % (len(GESTURE_CODES) ** (i + 1))
        gestures.append(part // (len(GESTURE_CODES) ** i))
    return gestures


class QuartetDataManager(object):
    """Manages data from metatone quartet performances and generates epochs"""

    def __init__(self, num_steps, batch_size, train_split=0.95):
        """Load Metatone Corpus and Create Example Data"""
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.train_split = train_split
        self.examples_file = "MetatoneQuartetExamples-" + str(self.num_steps) + "steps" + ".h5"

        # Make sure corpus is available.
        URL = "https://github.com/anucc/metatone-analysis/raw/master/metadata/"
        PICKLE_FILE = "metatone_performances_dataframe.pickle"
        if not os.path.exists(PICKLE_FILE):
            urlretrieve(URL + PICKLE_FILE, PICKLE_FILE)
        with open(PICKLE_FILE, 'rb') as f:
                self.metatone_dataset = pickle.load(f)

        # Load Quartet Improvisations.
        improvisations = self.metatone_dataset[
            (self.metatone_dataset["performance_type"] == "improvisation") &
            (self.metatone_dataset["performance_context"] != "demonstration") &
            (self.metatone_dataset["number_performers"] == 4)]
        gesture_data = improvisations['gestures']
        self.ensemble_improvisations = gesture_data.tolist()
        print("Number of performances in training data: ", len(self.ensemble_improvisations))
        print("Attempting to load", self.examples_file)
        if os.path.exists(self.examples_file):
            print("File exists, loading.")
            with h5py.File(self.examples_file, 'r') as data_file:
                self.dataset = data_file['examples'][:]
                self.validation_set = data_file['validation'][:]
        else:
            print("File doesn't exist, creating.")
            self.dataset, self.validation_set = self.setup_training_examples()
            print("Created Training Examples, now saving to h5 file.")
            self.dataset = np.array(self.dataset)
            self.validation_set = np.array(self.validation_set)
            with h5py.File(self.examples_file, 'w') as data_file:
                data_file.create_dataset('examples', data=self.dataset)
                data_file.create_dataset('validation', data= self.validation_set)
        print("Loaded", str(len(self.dataset)), "Training Examples.")

    def setup_test_data(self):
        """Load individual parts of non-trained data for testing."""
        improvisations = self.metatone_dataset[
            (self.metatone_dataset["performance_type"] == "improvisation") &
            (self.metatone_dataset["performance_context"] != "demonstration") &
            (self.metatone_dataset["number_performers"] != 4)]
        gesture_data = improvisations['gestures']
        self.individual_improvisations = []
        for perf in gesture_data.tolist():
            for one_perf in perf.T:
                self.individual_improvisations.append(one_perf)
        return self.individual_improvisations

    def setup_training_examples(self):
        """Setup training examples from corpus."""
        imp_xs = []
        imp_ys = []
        for imp in self.ensemble_improvisations:
            print("Processing performance data.")
            for i in range(len(imp) - self.num_steps - 1):
                imp_slice = imp[i:i + self.num_steps + 1]
                for j in range(len(imp_slice.T)):
                    lead = imp_slice[1:].T[j]  # lead gestures (post steps)
                    ensemble = imp_slice.T[np.arange(len(imp_slice.T)) != j]  # rest of the players indexed by player
                    for ens_perm in permutations(ensemble):  # consider all permutations of the players
                        ens_pre = np.array(ens_perm).T[:-1]  # indexed by time slice
                        ens_post = np.array(ens_perm).T[1:]  # indexed by time slice
                        y = map(encode_ensemble_gestures, ens_post)
                        # y = ens_post # test just show the gestures
                        x = map(encode_ensemble_gestures, zip(lead, *(ens_pre.T)))  # encode ensemble state
                        # x = zip(lead,*(ens_pre.T)) # test just show the gestures
                        imp_xs.append(x)  # append the inputs
                        imp_ys.append(y)  # append the outputs
        print("Total Examples:", len(imp_xs))
        print("Total Labels:", len(imp_ys))
        print("Splitting train set with prop:", self.train_split)
        X_train, X_test, y_train, y_test = train_test_split(imp_xs, imp_ys, train_size=self.train_split, random_state=SPLIT_RANDOM_STATE)
        ex_train = zip(X_train, y_train)
        ex_test = zip(X_test, y_test)
        print("Training Set:", str(len(ex_train)))
        print("Validation Set:", str(len(ex_test)))
        return ex_train, ex_test

    def next_epoch(self):
        """Return an epoch of batches of shuffled examples."""
        np.random.shuffle(self.dataset)
        dataset_size = len(self.dataset)
        batches = []
        for i in range(dataset_size // self.batch_size):
            batch = self.dataset[i * self.batch_size:(i + 1) * self.batch_size]
            bx, by = zip(*batch)
            batches.append((np.array(bx), np.array(by)))
        return(batches)


RNN_MODE_TRAIN = 'train'
RNN_MODE_RUN = 'run'
ENSEMBLE_SIZE_QUARTET = 4


class GestureRNN(object):

    def __init__(self, mode=RNN_MODE_TRAIN, ensemble_size=4, num_nodes=512, num_layers=3, testing=False):
        """
        Initialize GestureRNN model. Use "mode = 'run'" for evaluation graph
        and "mode = "train" for training graph.
        """
        # Model Hyperparameters
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.mode = mode
        self.testing = testing

        # IO Hyperparameters
        self.num_input_performers = ensemble_size
        self.num_output_performers = ensemble_size - 1
        self.vocabulary_size = len(GESTURE_CODES)
        self.num_classes = self.vocabulary_size
        self.num_input_classes = self.vocabulary_size ** self.num_input_performers
        self.num_output_classes = self.vocabulary_size ** self.num_output_performers

        # Training Hyperparamters
        self.learning_rate = 1e-4
        self.run_name = self.get_run_name()
        tf.logging.info("Loading %d to %d Gesture-RNN in %s mode with %d nodes in %d layers.", self.num_input_performers, self.num_output_performers, self.mode, self.num_nodes, self.num_layers)
        if self.mode is RNN_MODE_TRAIN:
            # Training Tensorsize
            self.batch_size = 64
            self.num_steps = 120
        else:
            # Running Hyperparameters
            self.batch_size = 1
            self.num_steps = 1

        # State Storage
        self.state = None
        self.training_state = None

        # Load the graph
        tf.reset_default_graph()
        self.graph = tf.get_default_graph()
        with self.graph.as_default():
            with tf.name_scope('input'):
                self.x = tf.placeholder(tf.int32, [self.batch_size, self.num_steps], name='input_placeholder')
                self.y = tf.placeholder(tf.int32, [self.batch_size, self.num_steps], name='labels_placeholder')
                # reshape labels to have shape (batch_size * num_steps, )
                self.y_reshaped = tf.reshape(self.y, [-1], name="reshape_labels")
            with tf.variable_scope('embedding'):
                self.embeddings = tf.get_variable('emb_matrix', [self.num_input_classes, self.num_nodes])
                self.rnn_inputs = tf.nn.embedding_lookup(self.embeddings, self.x, name="input_emb")
            # RNN section
            with tf.variable_scope('rnn'):
                rnn_cells = [tf.contrib.rnn.LSTMCell(self.num_nodes, state_is_tuple=True) for _ in range(self.num_layers)]
                self.cell = tf.contrib.rnn.MultiRNNCell(rnn_cells)
                self.init_state = self.cell.zero_state(self.batch_size, tf.float32)
                self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(self.cell, self.rnn_inputs, initial_state=self.init_state)
                self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.num_nodes], name="reshape_rnn_outputs")

            # Fully-Connected Softmax Section
            with tf.variable_scope('rnn_to_cat'):
                W = tf.get_variable('W', [self.num_nodes, self.num_output_classes])
                b = tf.get_variable('b', [self.num_output_classes], initializer=tf.constant_initializer(0.0))
                self.logits = tf.matmul(self.rnn_outputs, W, name="logits_mul") + b
            self.predictions = tf.nn.softmax(self.logits, name="softmax")
            tf.summary.histogram("out_weights", W)
            tf.summary.histogram("out_biases", b)
            tf.summary.histogram("out_logits", self.logits)

            self.saver = tf.train.Saver(name="saver", keep_checkpoint_every_n_hours=1)
            # Training Operations
            if self.mode is RNN_MODE_TRAIN:
                cost_function = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_reshaped, name="cross_entropy")
                self.loss = tf.reduce_mean(cost_function, name="loss")
                tf.summary.scalar("loss_summary", self.loss)
                with tf.name_scope('training'):
                    self.global_step = tf.Variable(0, name='global_step', trainable=False)
                    optimizer = tf.train.AdamOptimizer(self.learning_rate)
                    self.train_op = optimizer.minimize(self.loss, global_step=self.global_step, name="train_op")
            if self.testing:
                with tf.name_scope('accuracy'):
                    predicted_labels = tf.cast(tf.argmax(self.predictions, 1), tf.int32)
                    print(predicted_labels.shape)
                    print(self.y_reshaped.shape)
                    correct_predictions = tf.equal(predicted_labels, self.y_reshaped)
                    self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
                tf.summary.scalar("train_accuracy", self.accuracy)

            self.summaries = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(LOG_PATH + self.run_name + '/', graph=self.graph)
        train_vars_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        tf.logging.info("done initialising: %s vars: %d", self.model_name(), train_vars_count)

    def model_name(self):
        """Returns the name of the present model for saving to disk"""
        name = "gesture-rnn-%dto%d-%dn%dl" % (self.num_input_performers, self.num_output_performers, self.num_nodes, self.num_layers)
        return name

    def get_run_name(self):
        """Generates a time-stampted model name for marking runs"""
        out = self.model_name() + "-"
        out += time.strftime("%Y%m%d-%H%M%S")
        return out

    def train_batch(self, batch_x, batch_y, sess):
        """Train the network for just one batch."""
        if self.mode is not RNN_MODE_TRAIN:
            tf.logging.info("model not initialised to train.")
            return 0
        feed = {self.x: batch_x, self.y: batch_y}
        if self.training_state is not None:
            feed[self.init_state] = self.training_state
        # Training step evaluation:
        training_loss_current, self.training_state, _, summary, step = sess.run([
            self.loss, self.final_state, self.train_op, self.summaries, self.global_step], feed_dict=feed)
        self.writer.add_summary(summary, step)
        return training_loss_current, step

    def train_epoch(self, batches, sess):
        """Train the network on one epoch of training data."""
        total_training_loss = 0
        epoch_steps = 0
        total_steps = len(batches)
        step = 0
        for batch_x, batch_y in batches:
            training_loss, step = self.train_batch(batch_x, batch_y, sess)
            epoch_steps += 1
            total_training_loss += training_loss
            if (epoch_steps % 200 == 0):
                tf.logging.info("trained batch: %d of %d; loss was %f", epoch_steps, total_steps, training_loss)
        return (total_training_loss / epoch_steps), step

    def train(self, data_manager, num_epochs, saving=True):
        """Train the network for the a number of epochs."""
        self.num_epochs = num_epochs
        tf.logging.info("going to train: %s", self.model_name())
        start_time = time.time()
        training_losses = []
        step = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(num_epochs):
                batches = data_manager.next_epoch()
                epoch_average_loss, step = self.train_epoch(batches, sess)
                training_losses.append(epoch_average_loss)
                tf.logging.info("trained epoch %d of %d", i, self.num_epochs)
                if saving:
                    # Save a checkpoint
                    checkpoint_path = LOG_PATH + self.run_name + '/' + self.model_name()
                    tf.logging.info('saving model %s, global_step %d.', checkpoint_path, step)
                    self.saver.save(sess, checkpoint_path, global_step=step)
            if saving:
                # Save completed model.
                tf.logging.info('saving model %s.', self.model_name())
                self.saver.save(sess, self.model_name())
        print("It took ", time.time() - start_time, " to train the network.")

    def prepare_model_for_running(self, sess):
        """Prepare Model for Evaluation"""
        sess.run(tf.global_variables_initializer())
        self.saver.restore(sess, MODEL_DIR + self.model_name())
        self.state = None

    def generate_gestures(self, lead_player, prev_ensemble, sess):
        """ Evaluates the network once for a lead player and previous ensemble gestures.
        Returns the current ensemble gestures. The network state is preserved in between
        evaluations. """
        gesture_inputs = list(prev_ensemble)
        gesture_inputs.insert(0, lead_player)
        if self.state is not None:
            feed_dict = {self.x: [[encode_ensemble_gestures(gesture_inputs)]], self.init_state: self.state}
        else:
            feed_dict = {self.x: [[encode_ensemble_gestures(gesture_inputs)]]}
        preds, self.state = sess.run([self.predictions, self.final_state], feed_dict=feed_dict)
        output_step = np.random.choice(self.num_output_classes, 1, p=np.squeeze(preds))[0]  # choose the output step
        output_gestures = decode_ensemble_gestures(self.num_output_performers, output_step)
        return output_gestures

    def generate_performance(self, lead_performance, sess):
        """
        Generates ensemble responses to a complete performance by a lead player.
        lead_performance should be a list of gesture codes.
        """
        generated_performance = pd.DataFrame()
        generated_performance["lead"] = lead_performance
        output_perf = []
        previous_ensemble = decode_ensemble_gestures(self.num_output_performers, 0)
        self.prepare_model_for_running(sess)
        for gesture in lead_performance:
            previous_ensemble = self.generate_gestures(gesture, previous_ensemble, sess)
            output_perf.append(previous_ensemble)
        out = np.array(output_perf)
        for i, seq in enumerate(out.T):
            name = "rnn-player-" + str(i)
            generated_performance[name] = seq
        return generated_performance


def test_training(epochs=2):
    """ Test Training. """
    train_model(epochs, saving=False)


def test_evaluation(num_trials=100):
    """ Test evaluation of individual gestures.
    This is the template code for real-time use in Metatone Classifier.
    """
    print("Going to run an RNN generation test.")
    g = GestureRNN(mode=RNN_MODE_RUN)
    sess = tf.Session()
    g.prepare_model_for_running(sess)
    ens_gestures = [0, 0, 0]
    for i in range(num_trials):
        n = np.random.randint(len(GESTURE_CODES))
        ens_gestures = g.generate_gestures(n, ens_gestures, sess)
        print("in:", n, "out:", ens_gestures)
    sess.close()


def plot_gesture_only_score(plot_title, gestures):
    """ Plots a gesture score of gestures only """
    plt.style.use('ggplot')
    # ax = plt.figure(figsize=(35,10),frameon=False,tight_layout=True).add_subplot(111)
    ax = plt.figure(figsize=(14, 4), frameon=False, tight_layout=True).add_subplot(111)
    ax.yaxis.grid()
    plt.ylim(-0.5, 8.5)
    plt.yticks(np.arange(9), ['n', 'ft', 'st', 'fs', 'fsa', 'vss', 'bs', 'ss', 'c'])
    for n in gestures.columns:
        plt.plot(gestures.index, gestures[n], '-', label=n)
    plt.savefig(plot_title + '.pdf', dpi=150, format="pdf")


def generate_a_fake_performance(num_performances=1):
    q = QuartetDataManager(120, 64)
    individual_improvisations = q.setup_test_data()

    print("Number of performances for testing: ", len(individual_improvisations))
    # Do the math.
    g = GestureRNN(mode="run")
    for i in range(num_performances):
        player_one = np.random.choice(individual_improvisations)
        player_one = player_one.tolist()
        with tf.Session() as sess:
            perf = g.generate_performance(player_one, sess)
        plot_name = g.model_name() + "-perf-" + str(i)
        plot_gesture_only_score(plot_name, perf)


def train_model(epochs, saving=True, model='quartet', num_nodes=512):
    """ Train the model for a number of epochs. """
    # Presently, only the quartet model is working.
    if model is 'quartet':
        train_quartet(epochs=epochs, num_nodes=512)
    elif model is 'duo':
        train_duo(epochs=epochs, num_nodes=512)


def train_quartet(epochs=30, num_nodes=512):
    """ Train the model for a number of epochs. """
    np.random.seed(6789)
    tf.set_random_seed(2345)
    q = QuartetDataManager(120, 64)
    g = GestureRNN(mode="train", num_nodes=num_nodes)
    g.train(q, epochs)
    print("Done training phew.")


def train_duo(epochs=30, num_nodes=512):
    """ Train the model for a number of epochs. """
    tf.set_random_seed(2345)  # should this be removed?
    # d = DuetDataManager(120,64)
    g = GestureRNN(mode="train", ensemble_size=2)
    # g.train(d,epochs)
    print("Not implemented yet! Need to make the DuetDataManager as well!")


def test_duo_eval(num_trials=100):
    print("Duos not implemented yet, look in the notebook directory.")
    g = GestureRNN(mode="run", ensemble_size=2)
    sess = tf.Session()
    # g.prepare_model_for_running(sess) # this will fail as it's not trained.
    ens_gestures = [0]
    for i in range(num_trials):
        n = np.random.randint(len(GESTURE_CODES))
        # ens_gestures = g.generate_gestures(n,ens_gestures,sess)
        print("in:", n, "out:", ens_gestures)
    sess.close()


def main(_):
    """ Command line accessible functions. """
    if FLAGS.train:
        train_model(epochs=FLAGS.epochs, saving=True, model='quartet')
    if FLAGS.generate:
        generate_a_fake_performance(num_performances=FLAGS.num_perfs)
    if FLAGS.test_eval:
        test_evaluation()
    if FLAGS.test_train:
        test_training()


def training_experiment():
    """ 20170615: training experiment to create 3 models with different node sizes """
    train_quartet(epochs=50, num_nodes=64)
    train_quartet(epochs=50, num_nodes=128)
    train_quartet(epochs=50, num_nodes=256)
    train_quartet(epochs=50, num_nodes=512)


if __name__ == "__main__":
    tf.app.run(main=main)
