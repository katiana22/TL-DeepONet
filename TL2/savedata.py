'''
Authors: Katiana Kontolati, PhD Candidate, Johns Hopkins University
         Somdatta Goswami, Postdoctoral Researcher, Brown University
Tensorflow Version Required: TF1.15     
'''
import tensorflow.compat.v1 as tf
import numpy as np
import sys
from fnn import FNN
from plotting import *
import os
import scipy

class SaveData:
    def __init__(self):
        pass

    def save(self, sess, x_pos, fnn_model, W, b, Xmin, Xmax, u_B, f_ph, u_ph, data, num_test, save_results_to, domain):
        
        domain = domain
        save_results_to = save_results_to +"/" + domain        
        if not os.path.exists(save_results_to):
            os.makedirs(save_results_to)
            
        if domain == 'source':
            test_id, x_test, f_test, u_test = data.testbatch(num_test)
        else:
            test_id, x_test, f_test, u_test = data.testbatch_target(num_test)
        
        x = tf.tile(x_pos[None, :, :], [num_test, 1, 1])
        u_T = fnn_model.fnn(W, b, x, Xmin, Xmax)
        test_dict = {f_ph: f_test, u_ph: u_test}
        u_nn = u_B*u_T        
        u_pred = tf.reduce_sum(u_nn, axis=-1, keepdims=True)

        u_pred_ = sess.run(u_pred, feed_dict=test_dict)
        if domain == 'source':
            u_test = data.decoder(u_test)
            u_pred_ = data.decoder(u_pred_)
        else:
            u_test = data.decoder_target(u_test)
            u_pred_ = data.decoder_target(u_pred_)
        
        f_test = np.reshape(f_test, (f_test.shape[0], -1))
        u_pred_ = np.reshape(u_pred_, (u_test.shape[0], u_test.shape[1]))

        u_ref = np.reshape(u_test, (u_test.shape[0], u_test.shape[1]))

        err_u = np.mean(np.linalg.norm(u_pred_ - u_ref, 2, axis=1)/np.linalg.norm(u_ref, 2, axis=1))            
        
        if domain == 'source':
            print('Relative L2 Error (Source): %.3f'%(err_u))
            err_u = np.reshape(err_u, (-1, 1))
            np.savetxt(save_results_to+'/err', err_u, fmt='%e')       
            scipy.io.savemat(save_results_to+'/Darcy_source.mat', 
                         mdict={'x_test': f_test,
                                'y_test': u_test, 
                                'y_pred': u_pred_})
            
            scipy.io.savemat('./Plot'+'/Darcy_source.mat', 
                         mdict={'x_test': f_test,
                                'y_test': u_test, 
                                'y_pred': u_pred_})
        else:
            print('Relative L2 Error (Target): %.3f'%(err_u))
            err_u = np.reshape(err_u, (-1, 1))
            np.savetxt(save_results_to+'/err', err_u, fmt='%e')       
            scipy.io.savemat(save_results_to+'/Darcy_target.mat', 
                         mdict={'x_test': f_test,
                                'y_test': u_test, 
                                'y_pred': u_pred_})
            
            scipy.io.savemat('./Plot'+'/Darcy_target.mat', 
                         mdict={'x_test': f_test,
                                'y_test': u_test, 
                                'y_pred': u_pred_})