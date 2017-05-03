from abc import ABCMeta
import numpy as np

class RecNet:
    #__metaclass__ = ABCMeta
    def __init__ (self, nodes, level):
        self.nodes = nodes

    def activate (self, x):
        raise NotImplementedError('Activate not impl')

# Abstract network
class ARN (RecNet):
    def __init__ (self, nodes, level):
        size = 3

        # Weight matrices
        self.W_i  = np.random.rand(size,size*size)
        self.W_ih = np.random.rand(size,size)
        self.W_ho = np.random.rand(size,size)

        if level <= 1: # Base case
            self.input_nodes  = [BRN(nodes) for i in range(nodes)]
            self.hidden_nodes = [BRN(nodes) for i in range(nodes)]
            self.output_nodes = [BRN(nodes) for i in range(nodes)]
        else:
            self.input_nodes  = [ARN(nodes, level-1) for i in range(nodes)]
            self.hidden_nodes = [ARN(nodes, level-1) for i in range(nodes)]
            self.output_nodes = [ARN(nodes, level-1) for i in range(nodes)]

    def sigmoid (self, x):
        return np.divide(1., np.add(1., np.exp(-x)))

    def activate (self, x):
        # Compute input layer
        #inp_inputs = np.dot(self.W_i, x)
        i = np.array([ n.activate(x) for n in self.input_nodes ])

        # Compute hidden layer
        hid_inputs = np.dot(self.W_ih, i)
        h = np.array([ n.activate(hid_inputs) for n in self.hidden_nodes ])

        # Compute output layer
        out_inputs = np.dot(self.W_ho, h)
        outs = np.array([ x.activate(out_inputs) for x in self.output_nodes ])

        return self.sigmoid( outs )

# Base node
class BRN (RecNet):
    def __init__ (self, nodes):
        size = 3

        # Weight matrices
        self.W_i  = np.random.rand(size,size*size)
        self.W_ih = np.random.rand(size,size)
        self.W_ho = np.random.rand(size,size)

    def sigmoid (self, x):
        return 1. / (1. + np.exp(-x))

    def activate (self, x):
        i = self.sigmoid( np.dot(self.W_i, x) )
        h = self.sigmoid( np.dot(self.W_ih, i) )
        o = self.sigmoid( np.dot(self.W_ho, h) )

        return o.flatten()

if __name__ == '__main__':
    nn = ARN(3, 2)
    nn.activate( np.random.rand(9,1) )
