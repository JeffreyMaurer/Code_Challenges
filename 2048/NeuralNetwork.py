import random
import numpy as np

def nonlin(x, deriv=False):
    if(deriv==True):
        return x * (1-x)
    return 1/(1+np.exp(-x))


# np.random.seed(1)

# f = open("/usr/share/dict/words", "r")
# words = f.readlines()

class NeuralNetwork:
    
    next_ID = 0

    def __init__(self, HyperParams):
        self.synapses = []
        for synapse in range(len(HyperParams)-1):
            self.synapses.append(2*np.random.random((HyperParams[synapse], HyperParams[synapse+1]))-1)
        self.score = 0
        # self.name = words[random.randint(0, len(words))].strip()
        self.name = str(NeuralNetwork.next_ID)
        NeuralNetwork.next_ID += 1
        

    def load(self):
        with open("weights.txt", "r") as f:
            for line in f:
                self.synapses.append(exec(line))


    def save(self):
        with open("weights.txt", "w") as f:
            for layer in self.synapses:
                f.write(str(layer))


    def train_batch(self, state, correct_decision):
        for i in range(10*1000):
            self.layers = []
            self.layers.append(state)
            for j in range(len(self.synapses)):
                self.layers.append(nonlin(np.dot(self.layers[-1], self.synapses[j])))
                
            error = correct_decision - self.layers[-1]
            if (i % 1000) == 0: print(str(np.mean(np.abs(error))))

            for j in range(1,1+len(self.synapses)):
                delta = error * nonlin(self.layers[-j], True)
                error = delta.dot(self.synapses[-j].T)
                self.synapses[-j] += self.layers[-(j+1)].T.dot(delta)


    def train(self, state, decision):
        self.layers = []
        self.layers.append(state)
        for j in range(len(self.synapses)):
            self.layers.append(nonlin(np.dot(self.layers[-1], self.synapses[j])))
            
        error = correct_decision - self.layers[-1]
        for j in range(1,1+len(self.synapses)):
            delta = error * nonlin(self.layers[-j], True)
            error = delta.dot(self.synapses[-j].T)
            self.synapses[-j] += self.layers[-(j+1)].T.dot(delta)
                

    def next_gen(self):
        child = NeuralNetwork([1])
        for synapse in self.synapses:
            # add variation
            child.synapses.append(synapse + 0.1*np.random.random(synapse.shape)-0.05)
        # child.name += " son of " + self.name
        child.name += "<-" + self.name
        return child

    def feed(self, state):
        self.layers = []
        self.layers.append(state)
        for j in range(len(self.synapses)):
            self.layers.append(nonlin(np.dot(self.layers[-1], self.synapses[j])))
        # print(self.layers[-1])
        return self.layers[-1]

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

NN = NeuralNetwork((3,4,1))
if __name__ == "__main__":
    print(NN.synapses)
    print(NN.next_gen().synapses)
