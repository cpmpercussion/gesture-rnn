"""
Loads data from Metatone Duet Performances and generates epochs, batches, and sequences.
"""
from __future__ import division
from __future__ import print_function
import numpy as np
import h5py
import pickle
import os
from urllib import urlretrieve
from itertools import permutations
from sklearn.model_selection import train_test_split
from metatone_gesture_encoding import encode_ensemble_gestures

# Evaluating Network
NP_RANDOM_STATE = 6789
SPLIT_RANDOM_STATE = 2468


class DuetDataManager(object):
    """Manages data from metatone duet performances and generates epochs"""

    def __init__(self, num_steps, batch_size, train_split=0.95):
        """Load Metatone Corpus and Create Example Data"""
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.train_split = train_split
        self.examples_file = "MetatoneDuetExamples-" + str(self.num_steps) + "steps" + ".h5"

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
                data_file.create_dataset('validation', data=self.validation_set)
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
