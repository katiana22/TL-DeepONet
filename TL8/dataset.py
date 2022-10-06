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
    def __init__(self, num, bs):
        self.num = num
        self.bs = bs
        self.F_train, self.U_train, self.F_test, self.U_test, \
        self.X, self.u_mean, self.u_std = self.load_data()
        self.F_train_t, self.U_train_t, self.F_test_t, self.U_test_t, \
        self.X_t, self.u_mean_t, self.u_std_t = self.load_data_target()

    def func(self, x_train):
        f = np.sin(np.pi*x_train[:, 0:1])*np.sin(np.pi*x_train[:, 1:2])
        u = np.cos(np.pi*x_train[:, 0:1])*np.cos(np.pi*x_train[:, 1:2])
        return f, u

    def samples(self):

        num_train = 1
        x = np.linspace(-1, 1, self.num)
        y = np.linspace(-1, 1, self.num)
        xx, yy = np.meshgrid(x, y)
        xx = np.reshape(xx, (-1, 1))
        yy = np.reshape(yy, (-1, 1))
        x_train = np.hstack((xx, yy))
        F, U = self.func(x_train)

        Num = self.num*self.num

        F = np.reshape(F, (-1, self.num, self.num, 1))
        U = np.reshape(U, (-1, Num, 1))
        F_train = F[:num_train, :, :]
        U_train = U[:num_train, :, :]
        F_test = F[:num_train, :, :]
        U_test = U[:num_train, :, :]
        return F_train, U_train, F_test, U_test

    def decoder(self, x):
        x = x*(self.u_std + 1.0e-9) + self.u_mean
        return x
    
    def decoder_target(self, x):
        x = x*(self.u_std_t + 1.0e-9) + self.u_mean_t
        return x

    def load_data(self):
    
        # Source data
        file = np.load('./Data/Brusselator_output_b_2.2.npz', allow_pickle=True)
        nt, nx, ny = 10, file['nx'], file['ny']
        n_samples = file['n_samples']

        inputs = file['inputs'].reshape(n_samples, nx, ny)
        outputs = np.array((file['outputs'])).reshape(n_samples, nt, nx, ny)
        
        num_train = 800
        num_test = 200
        
        s, t = 28, 10
        
        f_train = inputs[:num_train, :, :]
        u_train = outputs[:num_train, :, :, :]   

        f_test = inputs[num_train:num_train+num_test, :, :]
        u_test = outputs[num_train:num_train+num_test, :, :, :] 

        #f_test = np.log(f_test)
        #f_train = np.log(f_train)
        
        x = np.linspace(0, 1, s)
        y = np.linspace(0, 1, s)
        z = np.sin(np.linspace(0, 1, t))

        zz, xx, yy = np.meshgrid(z, x, y, indexing='ij')

        xx = np.reshape(xx, (-1, 1)) # flatten
        yy = np.reshape(yy, (-1, 1)) # flatten
        zz = np.reshape(zz, (-1, 1)) # flatten

        X = np.hstack((zz, xx, yy)) # shape=[t*s*s,3]
                                  
        # compute mean values
        f_train_mean = np.mean(f_train, 0)
        f_train_std = np.std(f_train, 0)
        u_train_mean = np.mean(u_train, 0)
        u_train_std = np.std(u_train, 0)
 
        num_res = t*s*s # total output dimension
        #num_res = s*s # for taking the mean in time
        
        # Reshape
        f_train_mean = np.reshape(f_train_mean, (-1, s, s, 1))
        f_train_std = np.reshape(f_train_std, (-1, s, s, 1))
        u_train_mean = np.reshape(u_train_mean, (-1, num_res, 1))
        u_train_std = np.reshape(u_train_std, (-1, num_res, 1))

        #  Mean normalization of train data
        F_train = np.reshape(f_train, (-1, s, s, 1))
        F_train = (F_train - f_train_mean)/(f_train_std + 1.0e-9)
        U_train = np.reshape(u_train, (-1, num_res, 1))
        U_train = (U_train - u_train_mean)/(u_train_std + 1.0e-9)

        #  Mean normalization of test data (using the mean and std of train)
        F_test = np.reshape(f_test, (-1, s, s, 1))
        F_test = (F_test - f_train_mean)/(f_train_std + 1.0e-9)
        U_test = np.reshape(u_test, (-1, num_res, 1))
        U_test = (U_test - u_train_mean)/(u_train_std + 1.0e-9)

        '''
        U_ref = np.reshape(U_test, (U_test.shape[0], U_test.shape[1]))
        np.savetxt('./Output/u_ref', U_ref, fmt='%e')
        '''

        return F_train, U_train, F_test, U_test, X, u_train_mean, u_train_std


    def load_data_target(self):
        
        # Target data
        file = np.load('./Data/Brusselator_output_sharp.npz', allow_pickle=True)
        nt, nx, ny = 10, file['nx'], file['ny']
        n_samples = file['n_samples']

        inputs = file['inputs'].reshape(n_samples, nx, ny)
        outputs = np.array((file['outputs'])).reshape(n_samples, nt, nx, ny)
        
        num_train = 200
        num_test = 200
        
        s, t = 28, 10
        
        f_train = inputs[:num_train, :, :]
        u_train = outputs[:num_train, :, :, :]   

        f_test = inputs[num_train:num_train+num_test, :, :]
        u_test = outputs[num_train:num_train+num_test, :, :, :] 
        
        x = np.linspace(0, 1, s)
        y = np.linspace(0, 1, s)
        z = np.sin(np.linspace(0, 1, t))

        zz, xx, yy = np.meshgrid(z, x, y, indexing='ij')

        xx = np.reshape(xx, (-1, 1)) # flatten
        yy = np.reshape(yy, (-1, 1)) # flatten
        zz = np.reshape(zz, (-1, 1)) # flatten

        X_t = np.hstack((zz, xx, yy)) # shape=[t*s*s,3]
                                  
        # compute mean values
        f_train_mean = np.mean(f_train, 0)
        f_train_std = np.std(f_train, 0)
        u_train_mean = np.mean(u_train, 0)
        u_train_std = np.std(u_train, 0)
 
        num_res = t*s*s # total output dimension
        #num_res = s*s # for taking the mean in time
        
        # Reshape
        f_train_mean_t = np.reshape(f_train_mean, (-1, s, s, 1))
        f_train_std_t = np.reshape(f_train_std, (-1, s, s, 1))
        u_train_mean_t = np.reshape(u_train_mean, (-1, num_res, 1))
        u_train_std_t = np.reshape(u_train_std, (-1, num_res, 1))

        #  Mean normalization of train data
        F_train = np.reshape(f_train, (-1, s, s, 1))
        F_train_t = (F_train - f_train_mean_t)/(f_train_std_t + 1.0e-9)
        U_train = np.reshape(u_train, (-1, num_res, 1))
        U_train_t = (U_train - u_train_mean_t)/(u_train_std_t + 1.0e-9)

        #  Mean normalization of test data (using the mean and std of train)
        F_test = np.reshape(f_test, (-1, s, s, 1))
        F_test_t = (F_test - f_train_mean_t)/(f_train_std_t + 1.0e-9)
        U_test = np.reshape(u_test, (-1, num_res, 1))
        U_test_t = (U_test - u_train_mean_t)/(u_train_std_t + 1.0e-9)

        '''
        U_ref = np.reshape(U_test, (U_test.shape[0], U_test.shape[1]))
        np.savetxt('./Output/u_ref', U_ref, fmt='%e')
        '''
        return F_train_t, U_train_t, F_test_t, U_test_t, X_t, u_train_mean_t, u_train_std_t

    # Source
    def minibatch(self):
        # choose random indices - replace=False to avoid sampling same data
        batch_id = np.random.choice(self.F_train.shape[0], self.bs, replace=False)
        f_train = self.F_train[batch_id]
        u_train = self.U_train[batch_id]
        x_train = self.X

        Xmin = np.array([ 0., 0., 0.]).reshape((-1, 3))
        Xmax = np.array([ 1., 1., 1.]).reshape((-1, 3))
        #x_train = np.linspace(-1, 1, self.N).reshape((-1, 1))

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

        Xmin = np.array([ 0., 0., 0.]).reshape((-1, 3))
        Xmax = np.array([ 1., 1., 1.]).reshape((-1, 3))
        #x_train = np.linspace(-1, 1, self.N).reshape((-1, 1))

        return x_train, f_train, u_train, Xmin, Xmax


    def testbatch_target(self, num_test):
        batch_id = np.random.choice(self.F_test_t.shape[0], num_test, replace=False)
        f_test = self.F_test_t[batch_id]
        u_test = self.U_test_t[batch_id]
        x_test = self.X_t

        batch_id = np.reshape(batch_id, (-1, 1))

        return batch_id, x_test, f_test, u_test

