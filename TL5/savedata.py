'''
Authors: Katiana Kontolati, PhD Candidate, Johns Hopkins University
         Somdatta Goswami, Postdoctoral Researcher, Brown University
Tensorflow Version Required: TF1.15     
'''
import tensorflow as tf
import numpy as np
import sys
from fnn import FNN
from plotting import *
import os
import scipy

class SaveData:
    def __init__(self):
        pass

    def save(self, sess, x_pos, fnn_model, W, b, Xmin, Xmax, u_B, f_ph, ux_ph, uy_ph, data, num_test, save_results_to, domain):
        
        domain = domain
        save_results_to = save_results_to +"/" + domain        
        if not os.path.exists(save_results_to):
            os.makedirs(save_results_to)
            
        if domain == 'source':
            test_id, x_test, f_test, ux_test, uy_test = data.testbatch(num_test)
        else:
            test_id, x_test, f_test, ux_test, uy_test = data.testbatch_target(num_test)

        x = tf.tile(x_pos[None, :, :], [num_test, 1, 1])
        u_T = fnn_model.fnn(W, b, x, Xmin, Xmax)
        test_dict = {f_ph: f_test, ux_ph: ux_test, uy_ph: uy_test}
        u_nn = u_B*u_T

        ux_pred = tf.reduce_sum(u_nn[:,:,0:100], axis=-1, keepdims=True)
        uy_pred = tf.reduce_sum(u_nn[:,:,100:], axis=-1, keepdims=True)
        
        ux_pred_, uy_pred_ = sess.run([ux_pred, uy_pred], feed_dict=test_dict)
        if domain == 'source':
            ux_test, uy_test = data.decoder(ux_test, uy_test)
            ux_pred_, uy_pred_ = data.decoder(ux_pred_, uy_pred_)
        else:   
            ux_test, uy_test = data.decoder_target(ux_test, uy_test)
            ux_pred_, uy_pred_ = data.decoder_target(ux_pred_, uy_pred_)
            
        err_x = np.mean(np.linalg.norm(ux_pred_ - ux_test, 2, axis=1)/np.linalg.norm(ux_test, 2, axis=1))
        err_y = np.mean(np.linalg.norm(uy_pred_ - uy_test, 2, axis=1)/np.linalg.norm(uy_test, 2, axis=1))

        f_test = np.reshape(f_test, (f_test.shape[0], -1))
        ux_pred_ = np.reshape(ux_pred_, (ux_test.shape[0], ux_test.shape[1]))
        uy_pred_ = np.reshape(uy_pred_, (uy_test.shape[0], uy_test.shape[1]))
        
        Ux_ref = np.reshape(ux_test, (ux_test.shape[0], ux_test.shape[1]))
        Uy_ref = np.reshape(uy_test, (uy_test.shape[0], uy_test.shape[1]))
        
        if domain == 'source':
            print('Relative L2 Error (Source): %.3f,  %.3f'%(err_x, err_y))
            err_x = np.reshape(err_x, (-1, 1))
            err_y = np.reshape(err_y, (-1, 1))
            np.savetxt(save_results_to+'/err_x', err_x, fmt='%e')  
            np.savetxt(save_results_to+'/err_y', err_y, fmt='%e') 
            scipy.io.savemat(save_results_to+'/Elastic_source.mat', 
                         mdict={'x_test': f_test,
                                'ux_test': Ux_ref,
                                'uy_test': Uy_ref,
                                'ux_pred': ux_pred_,
                                'uy_pred': uy_pred_})
            
            scipy.io.savemat('./Plot'+'/Elastic_source.mat', 
                         mdict={'x_test': f_test,
                                'ux_test': Ux_ref,
                                'uy_test': Uy_ref,
                                'ux_pred': ux_pred_,
                                'uy_pred': uy_pred_})
        else:
            print('Relative L2 Error (Target): %.3f,  %.3f'%(err_x, err_y))
            err_x = np.reshape(err_x, (-1, 1))
            err_y = np.reshape(err_y, (-1, 1))
            np.savetxt(save_results_to+'/err_x', err_x, fmt='%e')  
            np.savetxt(save_results_to+'/err_y', err_y, fmt='%e') 
            scipy.io.savemat(save_results_to+'/Elastic_target.mat', 
                         mdict={'x_test': f_test,
                                'ux_test': Ux_ref,
                                'uy_test': Uy_ref,
                                'ux_pred': ux_pred_,
                                'uy_pred': uy_pred_})
            
            scipy.io.savemat('./Plot'+'/Elastic_target.mat', 
                         mdict={'x_test': f_test,
                                'ux_test': Ux_ref,
                                'uy_test': Uy_ref,
                                'ux_pred': ux_pred_,
                                'uy_pred': uy_pred_})