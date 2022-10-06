'''
Authors: Katiana Kontolati, PhD Candidate, Johns Hopkins University
         Somdatta Goswami, Postdoctoral Researcher, Brown University
Tensorflow Version Required: TF1.15     
'''
import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io as io
import sys
np.random.seed(1234)

class DataSet:
    def __init__(self, bs):

        self.bs = bs
        self.F_train, self.Ux_train, self.Uy_train, self.F_test, self.Ux_test, self.Uy_test, \
        self.X, self.ux_mean, self.ux_std, self.uy_mean, self.uy_std = self.load_data()
        self.F_train_t, self.Ux_train_t, self.Uy_train_t, self.F_test_t, self.Ux_test_t, self.Uy_test_t, \
        self.X_t, self.ux_mean_t, self.ux_std_t, self.uy_mean_t, self.uy_std_t = self.load_data_target()

    def decoder(self, x, y):

        x = (x-8.5)*(self.ux_std + 1.0e-9) + self.ux_mean 
        y = (y-8.5)*(self.uy_std + 1.0e-9) + self.uy_mean 
        x = x*1e-5
        y = y*1e-5
        return x, y
    
    def decoder_target(self, x, y):
        
        x = (x-3.0)*(self.ux_std_t + 1.0e-9) + self.ux_mean_t 
        y = (y-8.0)*(self.uy_std_t + 1.0e-9) + self.uy_mean_t 
        x = x*1e-2
        y = y*1e-2
        
        return x, y

    def load_data(self):
    
        # Source data
        file = io.loadmat('./Data/Dataset_1Circle')
        s_bc = 101
        s = 1024
        f_train = file['f_bc_train'] 
        ux_train = file['ux_train'] *1e5
        uy_train = file['uy_train'] *1e5

        f_test = file['f_bc_test']
        ux_test = file['ux_test'] *1e5
        uy_test = file['uy_test'] *1e5

        xx = file['xx']
        yy = file['yy']
        xx = np.reshape(xx, (-1, 1))
        yy = np.reshape(yy, (-1, 1))
        X = np.hstack((xx, yy))
        
        f_train_mean = np.mean(np.reshape(f_train, (-1, s_bc)), 0)
        f_train_std = np.std(np.reshape(f_train, (-1, s_bc)), 0)
        ux_train_mean = np.mean(np.reshape(ux_train, (-1, s)), 0)
        ux_train_std = np.std(np.reshape(ux_train, (-1, s)), 0)
        uy_train_mean = np.mean(np.reshape(uy_train, (-1, s)), 0)
        uy_train_std = np.std(np.reshape(uy_train, (-1, s)), 0)
        
        f_train_mean = np.reshape(f_train_mean, (-1, 1, s_bc))
        f_train_std = np.reshape(f_train_std, (-1, 1, s_bc))
        ux_train_mean = np.reshape(ux_train_mean, (-1, s, 1))
        ux_train_std = np.reshape(ux_train_std, (-1, s, 1))
        uy_train_mean = np.reshape(uy_train_mean, (-1, s, 1))
        uy_train_std = np.reshape(uy_train_std, (-1, s, 1))
        
        F_train = np.reshape(f_train, (-1, 1, s_bc))
        F_train = (F_train - f_train_mean)/(f_train_std + 1.0e-9) 
        Ux_train = np.reshape(ux_train, (-1, s, 1))
        Ux_train = (Ux_train - ux_train_mean)/(ux_train_std + 1.0e-9) + 8.5
        Uy_train = np.reshape(uy_train, (-1, s, 1))
        Uy_train = (Uy_train - uy_train_mean)/(uy_train_std + 1.0e-9) + 8.5

        F_test = np.reshape(f_test, (-1, 1, s_bc))
        F_test = (F_test - f_train_mean)/(f_train_std + 1.0e-9) 
        Ux_test = np.reshape(ux_test, (-1, s, 1))
        Ux_test = (Ux_test - ux_train_mean)/(ux_train_std + 1.0e-9) + 8.5
        Uy_test = np.reshape(uy_test, (-1, s, 1))
        Uy_test = (Uy_test - uy_train_mean)/(uy_train_std + 1.0e-9) + 8.5
        
        return F_train, Ux_train, Uy_train, F_test, Ux_test, Uy_test, X, ux_train_mean, \
            ux_train_std, uy_train_mean, uy_train_std


    def load_data_target(self):
        
        # Target data
        file = io.loadmat('./Data/Dataset_2Circle')
        s_bc = 101
        s = 1183
        
        num_train = 200
        num_test = 50
        
        f_train = file['f_bc_train'] 
        ux_train = file['ux_train'] *1e2
        uy_train = file['uy_train'] *1e2

        f_test = file['f_bc_test']
        ux_test = file['ux_test'] *1e2
        uy_test = file['uy_test'] *1e2
        
        xx = file['xx']
        yy = file['yy']
        xx = np.reshape(xx, (-1, 1))
        yy = np.reshape(yy, (-1, 1))
        X_t = np.hstack((xx, yy))

        f_train_mean = np.mean(np.reshape(f_train, (-1, s_bc)), 0)
        f_train_std = np.std(np.reshape(f_train, (-1, s_bc)), 0)
        ux_train_mean = np.mean(np.reshape(ux_train, (-1, s)), 0)
        ux_train_std = np.std(np.reshape(ux_train, (-1, s)), 0)
        uy_train_mean = np.mean(np.reshape(uy_train, (-1, s)), 0)
        uy_train_std = np.std(np.reshape(uy_train, (-1, s)), 0)
        
        f_train_mean = np.reshape(f_train_mean, (-1, 1, s_bc))
        f_train_std = np.reshape(f_train_std, (-1, 1, s_bc))
        ux_train_mean = np.reshape(ux_train_mean, (-1, s, 1))
        ux_train_std = np.reshape(ux_train_std, (-1, s, 1))
        uy_train_mean = np.reshape(uy_train_mean, (-1, s, 1))
        uy_train_std = np.reshape(uy_train_std, (-1, s, 1))
    
        F_train = np.reshape(f_train, (-1, 1, s_bc))
        F_train_t = (F_train - f_train_mean)/(f_train_std + 1.0e-9) 
        Ux_train = np.reshape(ux_train, (-1, s, 1))
        Ux_train_t = (Ux_train - ux_train_mean)/(ux_train_std + 1.0e-9) + 3.0
        Uy_train = np.reshape(uy_train, (-1, s, 1))
        Uy_train_t = (Uy_train - uy_train_mean)/(uy_train_std + 1.0e-9) + 8.0

        F_test = np.reshape(f_test, (-1, 1, s_bc))
        F_test_t = (F_test - f_train_mean)/(f_train_std + 1.0e-9) 
        Ux_test = np.reshape(ux_test, (-1, s, 1))
        Ux_test_t = (Ux_test - ux_train_mean)/(ux_train_std + 1.0e-9) + 3.0
        Uy_test = np.reshape(uy_test, (-1, s, 1))
        Uy_test_t = (Uy_test - uy_train_mean)/(uy_train_std + 1.0e-9) + 8.0

        return F_train_t, Ux_train_t, Uy_train_t, F_test_t, Ux_test_t, Uy_test_t, X_t, ux_train_mean, ux_train_std, uy_train_mean, uy_train_std

    # Source
    def minibatch(self):
        # choose random indices - replace=False to avoid sampling same data
        batch_id = np.random.choice(self.F_train.shape[0], self.bs, replace=False)
        f_train = self.F_train[batch_id]
        ux_train = self.Ux_train[batch_id]
        uy_train = self.Uy_train[batch_id]
        x_train = self.X

        Xmin = np.array([ 0., 0.]).reshape((-1, 2))
        Xmax = np.array([ 1., 1.]).reshape((-1, 2))

        return x_train, f_train, ux_train, uy_train, Xmin, Xmax

    def testbatch(self, num_test):
        
        batch_id = np.random.choice(self.F_test.shape[0], num_test, replace=False)
        f_test = self.F_test[batch_id]
        ux_test = self.Ux_test[batch_id]
        uy_test = self.Uy_test[batch_id]
        x_test = self.X

        batch_id = np.reshape(batch_id, (-1, 1))

        return batch_id, x_test, f_test, ux_test, uy_test
    
    # Target
    def minibatch_target(self):
        # choose random indices - replace=False to avoid sampling same data
        batch_id = np.random.choice(self.F_train_t.shape[0], self.bs, replace=False)
        f_train = self.F_train_t[batch_id]
        ux_train = self.Ux_train_t[batch_id]
        uy_train = self.Uy_train_t[batch_id]
        x_train = self.X_t

        Xmin = np.array([ 0., 0.]).reshape((-1, 2))
        Xmax = np.array([ 1., 1.]).reshape((-1, 2))

        return x_train, f_train, ux_train, uy_train, Xmin, Xmax


    def testbatch_target(self, num_test):
        batch_id = np.random.choice(self.F_test_t.shape[0], num_test, replace=False)
        f_test = self.F_test_t[batch_id]
        ux_test = self.Ux_test_t[batch_id]
        uy_test = self.Uy_test_t[batch_id]
        x_test = self.X_t

        batch_id = np.reshape(batch_id, (-1, 1))

        return batch_id, x_test, f_test, ux_test, uy_test

