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

        U_ref = np.reshape(u_test, (u_test.shape[0], u_test.shape[1]))
        np.savetxt(save_results_to +'/u_ref', U_ref, fmt='%e')
        np.savetxt(save_results_to +'/test_id', test_id, fmt='%e')
        np.savetxt(save_results_to +'/f_test', f_test, fmt='%e')
        np.savetxt(save_results_to +'/u_pred', u_pred_, fmt='%e')

        err = np.mean(np.linalg.norm(u_pred_ - U_ref, 2, axis=1)/np.linalg.norm(U_ref, 2, axis=1))
        err = np.reshape(err, (-1, 1))
        np.savetxt(save_results_to +'/err', err, fmt='%e')
        
        if domain == 'source':
            print('Relative L2 Error (Source): %.3f'%(err))
        else:
            print('Relative L2 Error (Target): %.3f'%(err))
        
        save_results_to = save_results_to +"/Plots/"
        os.makedirs(save_results_to)
        num_rows = 28
        num_cols = 28
        nt = 10
        for i in range(0, num_test, 20):

            k_print = f_test[i,:]
            k_print = k_print.reshape(num_rows, num_cols) 
            
            disp_pred = u_pred_[i,:].reshape(nt, num_rows, num_cols)                
            disp_true = U_ref[i,:].reshape(nt, num_rows, num_cols)   

            print(f"Plotting results for test sample: {i}", flush=True)
            dataSegment = "Test"
            plotField(k_print, disp_pred, disp_true, i, dataSegment, save_results_to) 
