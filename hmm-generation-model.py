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
#from sklearn import datasets, metrics, cross_validation
import matplotlib.pyplot as plt
import random
from scipy.stats import entropy

URL = "https://github.com/anucc/metatone-analysis/raw/master/metadata/"
PICKLE_FILE = "metatone_performances_dataframe.pickle"

if not os.path.exists(PICKLE_FILE):
    urlretrieve(URL + PICKLE_FILE, PICKLE_FILE)

with open(PICKLE_FILE, 'rb') as f:
        metatone_dataset = pickle.load(f)
        
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


## Load in the Dataset
improvisations = metatone_dataset[
    (metatone_dataset["performance_type"] == "improvisation") &
    (metatone_dataset["performance_context"] != "demonstration")]

#print(str(improvisations["performance_type"].unique()))
#print(str(improvisations["performance_context"].unique()))

print("Number of records: " + str(improvisations["number_performers"].count()))
print("Min performers: " + str(improvisations["number_performers"].min()) + " Max performers: "+ str(improvisations["number_performers"].max()))
print("Total performer-records: " + str(improvisations["number_performers"].sum()))

#performers = improvisations["number_performers"].plot(kind="box")
#plt.show(performers)


gesture_data = improvisations['gestures']
individual_improvisations = []

for perf in gesture_data.tolist():
    for one_perf in perf.T:
        individual_improvisations.append(one_perf)

print("Dataset contains " + str(len(individual_improvisations)) + " individual improvisations.")
vocabulary_size = len(GESTURE_CODES)

def generate_epochs(num_epochs, num_steps, batch_size):
    ## Setup the inputs and label sets
    imp_xs = []
    imp_ys = []

    for imp in individual_improvisations:
        for i in range(len(imp)-num_steps-1):
            imp_x = imp[i:i+num_steps]
            imp_y = imp[i+1:i+num_steps+1]
            imp_xs.append(imp_x)
            imp_ys.append(imp_y)
    
    dataset = zip(imp_xs,imp_ys)
    print("Total Training Examples: " + str(len(imp_xs)))
    print("Total Training Labels: " + str(len(imp_ys)))
    epochs = []
    for j in range(num_epochs):
        # shutffle the big list
        np.random.shuffle(dataset)
        dataset_size = len(dataset)
        batches = []
        for i in range(dataset_size / batch_size):
            ## Setup the batches
            batch = dataset[i*batch_size:(i+1)*batch_size]
            bx,by = zip(*batch)
            batches.append((np.array(bx),np.array(by)))
        epochs.append(batches)
    return epochs

individual_improvisations[0]

seq = individual_improvisations[0]

def one_step(i):
    out = np.zeros([9,9])
    out[i[0]][i[1]] += 1
    return out

def trans_mat(seq):
    o_s = np.array([seq[:-1],seq[1:]])
    return np.sum(map(one_step, o_s.T),axis=0)

def flux_measure(mat):
    """
    Measure of a transition matrix's flux. Given a numpy matrix M with
    diagonal D, returns the ||M||_1 - ||D||_1 / ||M||_1 Maximised at 1
    when nothing on diagonal, Minimised at 0 when everything on
    diagonal.
    """
    mat = np.array(mat)
    d = np.trace(mat) # |d|_1 
    m = np.sum(mat) # |M|_1
    if m == 0:
        # Take care of case of empty matrix
        # returning 0 is wrong but more benign than NaN
        measure = 0
    else:
        measure = (m - d) / m # Flux.
    return measure

def entropy_measure(mat):
    """
    Measures a transition matrix's entropy in the information
    theoretic sense. H(P) = -\sum_{i,j}p_{ij}\log_2(p_{ij}) Uses
    scipy.stats.entropy
    """
    return entropy(np.reshape(mat,len(mat)**2), base=2)

def flux_seq(seq):
    return flux_measure(trans_mat(seq))

def entropy_seq(seq):
    return entropy_measure(trans_mat(seq))

print("Real Performance Statistics")
real_performance_stats = pd.DataFrame({"flux":map(flux_seq,individual_improvisations),"entropy":map(entropy_seq,individual_improvisations)})
#real_performance_stats.plot(kind="box")
print(real_performance_stats.describe())

print("Fake Performance Statistics")
fake_performances = pd.DataFrame.from_csv("100epoch-120step-performances.csv")
fake_performance_stats = pd.DataFrame()
fake_performance_stats["flux"] = fake_performances.apply(flux_seq,axis=0)
fake_performance_stats["entropy"] = fake_performances.apply(entropy_seq,axis=0)
print(fake_performance_stats.describe())

## Make a big 'ol one-step Markov model of all the real performance data
