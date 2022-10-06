'''
Authors: Katiana Kontolati, PhD Candidate, Johns Hopkins University
         Somdatta Goswami, Postdoctoral Researcher, Brown University
Tensorflow Version Required: TF1.15     
'''
import tensorflow as tf
import numpy as np
import sys

class FNN:
    def __init__(self):
        pass
    
    def hyper_initial(self, layers):
        L = len(layers)
        W = []
        b = []
        for l in range(1, L):
            in_dim = layers[l-1]
            out_dim = layers[l]
            std = np.sqrt(2./(in_dim + out_dim))
            weight = tf.Variable(tf.random_normal(shape=[in_dim, out_dim], stddev=std))
            bias = tf.Variable(tf.zeros(shape=[1, out_dim]))
            W.append(weight)
            b.append(bias)
        return W, b

    def fnn(self, W, b, X, Xmin, Xmax):
        A = 2.0*(X - Xmin)/(Xmax - Xmin) - 1.0  # [-1,1]
        L = len(W)
        for i in range(L-1):
            A = tf.nn.leaky_relu(tf.add(tf.matmul(A, W[i]), b[i]))
        Y = tf.add(tf.matmul(A, W[-1]), b[-1])
        
        return Y

    def hyper_initial_target(self, layers, weights, biases):
        L = len(layers)
        W = []
        b = []
        # Freeze trunk net
        for l in range(1, L):
            if l==L-1:
                weight = tf.Variable(weights[l-1])
                bias = tf.Variable(biases[l-1])
            else:
                weight = tf.constant(weights[l-1])
                bias = tf.constant(biases[l-1])
            W.append(weight)
            b.append(bias)
        return W, b

    
    def l2_regularizer(self, W):
        
        l2 = 0.0
        L = len(W)
        for i in range(L-1):
            l2 += tf.nn.l2_loss(W[i])
        
        return l2  
    
    def fnn_B(self, W, b, X):
        A = X
        L = len(W)
        for i in range(L-1):
            A = tf.tanh(tf.add(tf.matmul(A, W[i]), b[i]))
        Y = tf.add(tf.matmul(A, W[-1]), b[-1])
    
        return Y
