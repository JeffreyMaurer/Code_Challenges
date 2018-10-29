import random
import numpy as np

"""
In this section, there will be a few different nonlinear functions to choose from.
They all have two params: x, which is a scalar, vector, matrix or tensor; and a boolean
for whether we are finding the derivative at that point or not. The default for deriv is False.
"""

def sigmoid(x, deriv=False):
    if (deriv==True):
        return x * (1-x)
    return 1/(1+np.exp(-x))


def relu(x, deriv=False):
    z = np.array(x)
    z[z<0] = 0
    z = z/np.abs(np.max(z))
    if (deriv==True):
        z[z>0] = 1
    return z

def leaky_relu(x, deriv=False):
    z = np.array(x)
    z = z/np.abs(np.max(z))
    z[z<0] *= 0.01
    if (deriv==True):
        z[z>0] = 1
        z[z<=0] = 0.01
    return z


def softplus(x, deriv=False):
    z = np.array(x)
    z = z/np.abs(np.max(z))
    if (deriv==True):
        return z
    return np.log(1+np.exp(z))

# np.random.seed(1)

class NeuralNetwork:
    """
    This class describes a fully connected feed-forward neural network that is 
    capable of updating weights through backpropagation and genetic algorithms.
    """

    next_ID = 0
    learning_rate = 0.01

    def __init__(self, HyperParams, nonlin=sigmoid):
        """
        description: This initializes the network's architecture, weights and name.
        params:
            HyperParams: iterable of integers, signifying how many nodes in each layer
            nonlin: the activation function to use for all layers
        output: 
            self: the neural network
        modified:
            self.synapses: a list of numpy matrices, the weights of the network
            self.score: number, the networks's "fitness" (used for genetic algorithm)
            self.name: integer, the identity of the network
            self.nonlin: the network's activation function
        """
        self.synapses = []
        for synapse in range(len(HyperParams)-1):
            self.synapses.append(2*np.random.random((HyperParams[synapse], HyperParams[synapse+1]))-1)
        self.score = 0
        self.name = str(NeuralNetwork.next_ID)
        NeuralNetwork.next_ID += 1
        self.nonlin=nonlin
        
    def feed(self, state):
        """
        description: This is the main function of the neural network, it produces output from input
        params:
            state: a numpy vector or matrix of floats or integers, the input to the neural network
        output: 
            self.layers[-1]: a numpy vector or matrix of floats, the final output of the network
        modified:
            self.layers: a list of numpy vectors or matrices of numbers, the input and output of each layer
        """
        self.layers = []
        self.layers.append(state)
        for j in range(len(self.synapses)):
            self.layers.append(self.nonlin(np.dot(self.layers[-1], self.synapses[j])))
        return(self.layers[-1])

    def backprop(self, error):
        """
        description: This is the second main function of the neural network, for training of single examples at a time
        params:
            error: a numpy vector of floats or integers, the difference between the 
            actual output of feed and the desired output
        output: 
            None
        modified:
            self.synapses: a list of numpy vectors or matrices of numbers, the weights between each layer
        """
        for j in range(1,1+len(self.synapses)):
            delta = error * self.nonlin(self.layers[-j], True)
            error = delta.dot(self.synapses[-j].T)
            self.synapses[-j] += NeuralNetwork.learning_rate*np.outer(self.layers[-(j+1)].T, delta)

    def backprop_batch(self, error):
        """
        description: This is the second main function of the neural network, for training of batches
        params:
            error: a numpy matrix of floats or integers, the difference between the 
            actual output of feed and the desired output
        output: 
            None
        modified:
            self.synapses: a list of numpy vectors or matrices of numbers, the weights between each layer
        """
        for j in range(1,1+len(self.synapses)):
            delta = error * self.nonlin(self.layers[-j], True)
            error = delta.dot(self.synapses[-j].T)
            self.synapses[-j] += self.layers[-(j+1)].T.dot(delta)


    def train_batch(self, state, outcome, epoch = 10*1000):
        """
        description: The function to train the neural network using batches
        params:
            state: a numpy matrix of floats or integers, the input to the neural network
            outcome: a numpy matrix of floats or integers, the desired output
            epoch: integer, the number of iterations of training on this batch
        output: 
            Every 100 epoch, the absolute means error is printed to the command line
        modified:
            self.synapses: a list of numpy matrices of numbers, the weights between each layer
            self.layers: a list of numpy matrices of numbers, the input and output of each layer
        """
        for i in range(epoch):
            output = self.feed(state)

            error = outcome - output
            if (i % (epoch//100)) == 0: print(str(np.mean(np.abs(error))))

            self.backprop_batch(error)

    def train_simple(self, state, outcome, epoch=1):
        """
        description: The function to train the neural network using individual examples
        params:
            state: a numpy vector of floats or integers, the input to the neural network
            outcome: a numpy vector of floats or integers, the desired output
            epoch: integer, the number of iterations of training on this example
        output: 
            None
        modified:
            self.synapses: a list of numpy vectors of numbers, the weights between each layer
            self.layers: a list of numpy vectors of numbers, the input and output of each layer
        """
        for i in range(epoch):
            output = self.feed(state)
            error = outcome - output
            self.backprop(error)
        
    def next_gen(self):
        """
        description: The function to train the neural network using a genetic algorithm
        params:
            None
        output: 
            child: a neural network with weights that are similar to self
        modified:
            None
        """
        child = NeuralNetwork([1])
        for synapse in self.synapses:
            # add variation
            child.synapses.append(synapse + 0.10*np.random.random(synapse.shape)-0.05)
        child.name += "<-" + self.name
        return(child)

    def train_replay(self, replay, attack_freq): # s, a, r, s'
        state, action, reward, state_prime = replay
        Q = self.feed(state)
        Q[action] = reward * attack_freq[action] / sum(attack_freq) # + y * max(np.multiply(Q.feed(state), mask))
        self.backprop(Q)

    def train_replay_batch(self, replays, attack_freq): # s, a, r, s'
        #states = np.array([replay[0] for replay in replays]) # 4 x 50
        #vector_actions = np.array([replay[1] for replay in replays]) # 1 x 50
        #matrix_actions = np.zeros((len(attack_freq),len(replays))
        #matrix_actions[np.arange(len(replays)), vector_actions] = 1
        #rewards = np.array([replay[2] for replay in replays]) # 1 x 50
        #Q = self.feed(states)
        #Q[np.arange(len(replays)), vector_actions] = rewards * attack_freq[matrix_actions] / sum(attack_freq) # + y * max(np.multiply(Q.feed(state), mask))
        #self.backprop_batch(Q)
        pass

# As a show of how the neural network can be used, 
# here is the neural network solving the xor problem
    
# input data
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])


#output data
y = np.array([[0],
             [1],
             [1],
             [0]])

if __name__ == "__main__":
    NN = NeuralNetwork((3,10,10,1), relu)
    NN.train_batch(X, y, 10000)
    print(NN.feed(X))


