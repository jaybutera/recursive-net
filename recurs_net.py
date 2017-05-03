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
        # Weight matrices
        self.W_i  = np.random.rand((3,3), dtype=np.float32)
        self.W_ih = np.random.rand((3,3), dtype=np.float32)
        self.W_ho = np.random.rand((3,3), dtype=np.float32)

        if level < 1: # Base case
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
        inp_inputs = np.dot(self.W_i, x)
        i = np.array([ x.activate(inp_inputs) for x in self.input_nodes ])

        # Compute hidden layer
        hid_inputs = np.dot(self.W_ih, inps)
        h = np.array([ x.activate(hid_inputs) for x in self.hidden_nodes ])

        # Compute output layer
        out_inputs = np.dot(self.W_ho, h)
        outs = np.array([ x.activate(out_inputs) for x in self.output_nodes ])

        return self.sigmoid( outs )

# Base node
class BRN (RecNet):
    #def __init__ (self, 
    def sigmoid (self, x):
        return 1. / (1. + np.exp(-x))

    def activate (self, x):
        i = np.sigmoid( np.dot(self.W_i, x) )
        h = np.sigmoid( np.dot(self.W_ih, i) )
        o = np.sigmoid( np.dot(self.W_ho, o) )

        return o

    def __rmul__ (self, o):
        return 

RecNet.register(ARN)
RecNet.register(BRN)

class ARN:
    def __init__ (self, nodes):
        self.input_nodes  = [ARN(nodes) for i in range(nodes)]
        self.hidden_nodes = [ARN(nodes) for i in range(nodes)]
        self.output_nodes = [ARN(nodes) for i in range(nodes)]

class BRN:
    def __init__ (self, nodes):
