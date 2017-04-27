"""
Module for Generating and Predicting Values from n-order Markov
Models

Charles Martin
2016
"""
import numpy as np

class MarkovModel:
    """
    Class containing the Markov model, must be initialised with an
    order and size of state-space.
    """
    
    def __init__(self, order, num_states):
        """
        Initialise the Markov model with an order and number of states.
        """
        self.order = order
        self.num_states = num_states
        self.transition_matrix = np.zeros([self.num_states for i in range(self.order+1)])
        self.transition_counts = np.zeros([self.num_states for i in range(self.order+1)], dtype=np.int)
        self.choice_failure = 0

    def fit(self,examples):
        """
        Create a new transition matrix given a list of examples each being
        a list of self.order+1 states in the set range(self.num_states). Will
        not use examples of any different length.
        """
        for example in examples:
            if len(example) == (self.order + 1):
                self.transition_counts[tuple(example)] += 1
            else:
                print("Skipped bad example with length: " + str(len(example)) + " " + str(example))
        self.transition_matrix = self.count_matrix_to_stochastic_matrix(self.transition_counts)
        
    def predict(self,input):
        """
        Given a sequence of self.order states, predict the next state in the sequence.
        """
        if np.shape(input) == (self.order,):
            stoch_vector = self.transition_matrix[tuple(input)]
            if stoch_vector.shape == (self.num_states,):
                return self.choose_from_stochastic_vector(stoch_vector)
            else:
                print("Error with stoch vector shape: " + str(np.shape(stoch_vector)))
                return None
        else:
            print("Error with input shape: " + str(np.shape(input)))
            return None

    def count_matrix_to_stochastic_matrix(self, mat):
        """
        Given an arbitrary matrix of transition counts, calculate the stochastic matrix
        """
        s = mat.shape
        m = mat.reshape((-1,self.num_states))
        m = np.true_divide(m,np.sum(m,axis=1).reshape(-1,1))
        m = np.nan_to_num(m)
        return(m.reshape(s))
        
    def choose_from_stochastic_vector(self,vec):
        """
        Given a stochastic vector vec, choose an index using the probabilities weighted by the vector.
        """
        try:
	    rnd = np.random.random() * np.sum(vec)
	    for i, w in enumerate(vec):
                rnd -= w
                if rnd < 0:
                    return i
            if rnd == 0:
                #print("Zero vector, making uniform random choice")
                self.choice_failure += 1
                return self.uniform_choice_of_state()
        except ValueError:
            print("Error on shape of Vec: " + str(vec.shape) + " and rnd: " + str(rnd.shape))
            return self.uniform_choice_of_state

    def uniform_choice_of_state(self):
        return np.random.choice(range(self.num_states))

    def generate_priming_sequence(self):
        output = []
        for i in range(self.order):
            output.append(self.uniform_choice_of_state())
        return output
    
    def generate_sequence(self, primer, length):
        """
        Generate a sequence of a certain length given a priming sequence of length order.
        """
        output = primer
        for i in range(length):
            new_primer = output[-1*self.order:]
            if np.shape(new_primer) == (self.order,):
                next = self.predict(new_primer)
                output.append(next)
            else:
                print("error in length of primer: " + str(np.shape(new_primer)))
        return output

