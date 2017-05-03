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
    def __init__ (self, nodes, inputs, level):
        # Weight matrices
        self.W_i  = np.random.rand(nodes,inputs)
        self.W_ih = np.random.rand(nodes,nodes)
        self.W_ho = np.random.rand(nodes,nodes)
        self.W_oo = np.random.rand(1,nodes)

        if level <= 1: # Base case
            self.input_nodes  = [BRN(nodes, inputs) for i in range(nodes)]
            self.hidden_nodes = [BRN(nodes, nodes) for i in range(nodes)]
            self.output_nodes = [BRN(nodes, nodes) for i in range(nodes)]
        else:
            self.input_nodes  = [ARN(nodes, inputs, level-1) for i in range(nodes)]
            self.hidden_nodes = [ARN(nodes, nodes, level-1) for i in range(nodes)]
            self.output_nodes = [ARN(nodes, nodes, level-1) for i in range(nodes)]

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

        out_point = self.sigmoid( np.dot(self.W_oo, outs) )

        return out_point
        #return self.sigmoid( outs )

# Base node
class BRN (RecNet):
    def __init__ (self, nodes, inputs):
        # Weight matrices
        self.W_i  = np.random.rand(nodes,inputs)
        self.W_ih = np.random.rand(nodes,nodes)
        self.W_ho = np.random.rand(nodes,nodes)
        self.W_oo = np.random.rand(1,nodes)

    def sigmoid (self, x):
        return 1. / (1. + np.exp(-x))

    def activate (self, x):
        i = self.sigmoid( np.dot(self.W_i, x) )
        h = self.sigmoid( np.dot(self.W_ih, i) )
        o = self.sigmoid( np.dot(self.W_ho, h) )
        o_flat = self.sigmoid( np.dot(self.W_oo, o) )

        return o_flat.flatten()[0]

if __name__ == '__main__':
    nn = ARN(3,8, 2)
    print( nn.activate( np.random.rand(8,1) ) )
