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
        self.F_train, self.U_train, self.F_test, self.U_test, \
        self.X, self.u_mean, self.u_std = self.load_data()
        self.F_train_t, self.U_train_t, self.F_test_t, self.U_test_t, \
        self.X_t, self.u_mean_t, self.u_std_t = self.load_data_target()

    def decoder(self, x):
        x = x*(self.u_std + 1.0e-9) + self.u_mean
        return x
    
    def decoder_target(self, x):
        x = x*(self.u_std_t + 1.0e-9) + self.u_mean_t
        return x

    def load_data(self):
    
        # Source data
        #Dataset 1: Correlation length is 0.02
        #Dataset 2: Correlation length is 0.1
        #Dataset 3: Correlation length is 0.3

        file = io.loadmat('./Data/Dataset1/Dataset_square')

        s = 100
        r = 1541

        # Training target data from scratch
        #file = io.loadmat('./Data/Dataset1/Dataset_right_triangle')
        #file = io.loadmat('./Data/Dataset1/Dataset_triangle')
        #s = 100
        #r = 2295 # 1200

        f_train = file['k_train']
        u_train = file['u_train']
        
        f_test = file['k_test']
        u_test = file['u_test']
        
        f_test = np.log(f_test)
        f_train = np.log(f_train)
        
        xx = file['xx']
        yy = file['yy']
        xx = np.reshape(xx, (-1, 1))
        yy = np.reshape(yy, (-1, 1))
        X = np.hstack((xx, yy))
        
        f_train_mean = np.mean(np.reshape(f_train, (-1, s, s)), 0)
        f_train_std = np.std(np.reshape(f_train, (-1, s, s)), 0)
        f_train_mean = np.reshape(f_train_mean, (-1, s, s, 1))
        f_train_std = np.reshape(f_train_std, (-1, s, s, 1))
        F_train = np.reshape(f_train, (-1, s, s, 1))
        F_train = (F_train - f_train_mean)/(f_train_std) #+ 5.0
        F_test = np.reshape(f_test, (-1, s, s, 1))
        F_test = (F_test - f_train_mean)/(f_train_std) #+ 5.0

        u_train_mean = np.mean(np.reshape(u_train, (-1, r)), 0)
        u_train_std = np.std(np.reshape(u_train, (-1, r)), 0)
        u_train_mean = np.reshape(u_train_mean, (-1, r, 1))
        u_train_std = np.reshape(u_train_std, (-1, r, 1))
        U_train = np.reshape(u_train, (-1, r, 1))
        U_train = (U_train - u_train_mean)/(u_train_std + 1.0e-9)
        U_test = np.reshape(u_test, (-1, r, 1))
        U_test = (U_test - u_train_mean)/(u_train_std + 1.0e-9)

        return F_train, U_train, F_test, U_test, X, u_train_mean, u_train_std


    def load_data_target(self):
        
        # Target data
        #Dataset 1: Correlation length is 0.02
        #Dataset 2: Correlation length is 0.1
        #Dataset 3: Correlation length is 0.3

        file = io.loadmat('./Data/Dataset1/Dataset_right_triangle')
        #file = io.loadmat('./Data/Dataset1/Dataset_triangle')

        s = 100
        r = 1200
        ##### For Rightangled triangle r = 1200
        ##### For equilateral triangle r = 2295
        
        num_train = 200
        num_test = 50
        
        f_train = file['k_train'][:num_train,:,:]
        u_train = file['u_train'][:num_train,:]

        f_test = file['k_test'][:num_test,:,:]
        u_test = file['u_test'][:num_test,:]
        
        f_test = np.log(f_test)
        f_train = np.log(f_train)
        
        xx = file['xx']
        yy = file['yy']
        xx = np.reshape(xx, (-1, 1))
        yy = np.reshape(yy, (-1, 1))
        X_t = np.hstack((xx, yy))
        
        f_train_mean = np.mean(np.reshape(f_train, (-1, s, s)), 0)
        f_train_std = np.std(np.reshape(f_train, (-1, s, s)), 0)
        f_train_mean = np.reshape(f_train_mean, (-1, s, s, 1))
        f_train_std = np.reshape(f_train_std, (-1, s, s, 1))
        
        F_train_t = np.reshape(f_train, (-1, s, s, 1))
        F_train_t = (F_train_t - f_train_mean)/(f_train_std) #+ 5.0
        F_test_t = np.reshape(f_test, (-1, s, s, 1))
        F_test_t = (F_test_t - f_train_mean)/(f_train_std) #+ 5.0
       
        u_train_mean = np.mean(np.reshape(u_train, (-1, r)), 0)
        u_train_std = np.std(np.reshape(u_train, (-1, r)), 0)
        u_train_mean_t = np.reshape(u_train_mean, (-1, r, 1))
        u_train_std_t = np.reshape(u_train_std, (-1, r, 1))      
        U_train_t = np.reshape(u_train, (-1, r, 1))
        U_train_t = (U_train_t - u_train_mean_t)/(u_train_std_t + 1.0e-9)
        U_test_t = np.reshape(u_test, (-1, r, 1))
        U_test_t = (U_test_t - u_train_mean_t)/(u_train_std_t + 1.0e-9)

        return F_train_t, U_train_t, F_test_t, U_test_t, X_t, u_train_mean_t, u_train_std_t

    # Source
    def minibatch(self):
        # choose random indices - replace=False to avoid sampling same data
        batch_id = np.random.choice(self.F_train.shape[0], self.bs, replace=False)
        f_train = self.F_train[batch_id]
        u_train = self.U_train[batch_id]
        x_train = self.X

        Xmin = np.array([ 0., 0.]).reshape((-1, 2))
        Xmax = np.array([ 1., 1.]).reshape((-1, 2))

        return x_train, f_train, u_train, Xmin, Xmax

    def testbatch(self, num_test):
        batch_id = np.random.choice(self.F_test.shape[0], num_test, replace=False)
        f_test = self.F_test[batch_id]
        u_test = self.U_test[batch_id]
        x_test = self.X

        batch_id = np.reshape(batch_id, (-1, 1))

        return batch_id, x_test, f_test, u_test
    
    # Target
    def minibatch_target(self):
        # choose random indices - replace=False to avoid sampling same data
        batch_id = np.random.choice(self.F_train_t.shape[0], self.bs, replace=False)
        f_train = self.F_train_t[batch_id]
        u_train = self.U_train_t[batch_id]
        x_train = self.X_t

        Xmin = np.array([ 0., 0.]).reshape((-1, 2))
        Xmax = np.array([ 0.5, 1.]).reshape((-1, 2))

        return x_train, f_train, u_train, Xmin, Xmax


    def testbatch_target(self, num_test):
        batch_id = np.random.choice(self.F_test_t.shape[0], num_test, replace=False)
        f_test = self.F_test_t[batch_id]
        u_test = self.U_test_t[batch_id]
        x_test = self.X_t

        batch_id = np.reshape(batch_id, (-1, 1))

        return batch_id, x_test, f_test, u_test

