'''
Manuscript Associated: Deep transfer operator learning for partial differential equations under conditional shift
Authors: Katiana Kontolati, PhD Candidate, Johns Hopkins University
         Somdatta Goswami, Postdoctoral Researcher, Brown University
Tensorflow Version Required: TF1.15     
This should be used for sharp data    

This is the target model. Run this after completing the simulation of the source model.
'''
import os
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as io
from dataset import DataSet
from fnn import FNN
from conv import CNN
from savedata import SaveData
from loss import CEOD_loss
import sys 

print("You are using TensorFlow version", tf.__version__)

save_index = 1
current_directory = os.getcwd()    
case = "Case_"
folder_index = str(save_index)
results_dir = "/" + case + folder_index +"/Results"
variable_dir = "/" + case + folder_index +"/Variables"
save_results_to = current_directory + results_dir
save_variables_to = current_directory + variable_dir

np.random.seed(1234)
#tf.set_random_seed(1234)

#output dimension of Branch/Trunk (latent dimension)
p = 150

#fnn in CNN
layer_B = [512, 256, p]
#trunk net
layer_T = [2, 128, 128, 128, p]
#resolution
h = 100
w = 100

#parameters in CNN
n_channels = 1
#n_out_channels = 16
filter_size_1 = 5
filter_size_2 = 5
filter_size_3 = 5
filter_size_4 = 5
stride = 1

#filter size for each convolutional layer
num_filters_1 = 16
num_filters_2 = 16
num_filters_3 = 16
num_filters_4 = 64
#batch_size
bs = 50

#size of input for Trunk net
x_num = 2295 
##### For Rightangled triangle x_num = 1200
##### For equilateral triangle x_num = 2295
beta = 0.001
def main():
    
    loss2 = CEOD_loss(x_num, bs)
    
    # Load data 
    data = DataSet(bs)
    x_train, f_train, u_train, Xmin, Xmax = data.minibatch_target()
    x_pos = tf.constant(x_train, dtype=tf.float32)
    x = tf.tile(x_pos[None, :, :], [bs, 1, 1]) #[bs, x_num, x_dim]
    
    # c1 = tf.Variable(tf.random_normal(shape = [1, 1], dtype = tf.float32), dtype = tf.float32)
    c2 = tf.Variable(tf.random_normal(shape = [1, 1], dtype = tf.float32), dtype = tf.float32)
    
    # Load pre-trained variables (from source network)
    cnn_vars = io.loadmat(save_variables_to+'/CNN_vars.mat')
    fnn_vars = io.loadmat(save_variables_to+'/FNN_vars.mat')
    
    # Placeholders
    fnn_layer_1_ph = tf.placeholder(shape=[None, layer_B[0]], dtype=tf.float32)
    u_pred_ph_s = tf.placeholder(shape=[None, x_num, 1], dtype=tf.float32)
    f_ph = tf.placeholder(shape=[None, h, w, n_channels], dtype=tf.float32) #[bs, 1, h, w, n_channels]
    u_ph = tf.placeholder(shape=[None, x_num, 1], dtype=tf.float32) #[bs, x_num, 1]
    learning_rate = tf.placeholder(tf.float32, shape=[])

    # Target branch net
    # CNN of branch net
    conv_model = CNN()

    #conv_linear = conv_model.linear_layer(f_ph, n_out_channels)
    conv_1 = conv_model.conv_layer_target(f_ph, cnn_vars['W1'], cnn_vars['b1'], stride, actn=tf.nn.relu)
    pool_1 = conv_model.avg_pool(conv_1, ksize=2, stride=2)   
    conv_2 = conv_model.conv_layer_target(pool_1, cnn_vars['W2'], cnn_vars['b2'], stride, actn=tf.nn.relu)
    pool_2 = conv_model.avg_pool(conv_2, ksize=2, stride=2) 
    conv_3 = conv_model.conv_layer_target(pool_2, cnn_vars['W3'], cnn_vars['b3'], stride, actn=tf.nn.relu)
    pool_3 = conv_model.avg_pool(conv_3, ksize=2, stride=2)
    conv_4 = conv_model.conv_layer_target(pool_3, cnn_vars['W4'], cnn_vars['b4'], stride, actn=tf.nn.relu)
    pool_4 = conv_model.avg_pool(conv_4, ksize=2, stride=2) 
    layer_flat = conv_model.flatten_layer(pool_4)

    # FNN of branch net
    fnn_layer_1, Wf1, bf1 = conv_model.fnn_layer_target(layer_flat, cnn_vars['Wf1'], cnn_vars['bf1'], actn=tf.tanh, use_actn=True)
    fnn_layer_2, Wf2, bf2 = conv_model.fnn_layer_target(fnn_layer_1, cnn_vars['Wf2'], cnn_vars['bf2'], actn=tf.nn.tanh, use_actn=True)
    out_B, Wf3, bf3 = conv_model.fnn_layer_target(fnn_layer_2, cnn_vars['Wf3'], cnn_vars['bf3'], actn=tf.tanh, use_actn=False) #[bs, p]
    u_B = tf.tile(out_B[:, None, :], [1, x_num, 1]) #[bs, x_num, p]
    
    # Trunk net
    Wt, bt = [fnn_vars['W1'], fnn_vars['W2'], fnn_vars['W3'], fnn_vars['W4']], [fnn_vars['b1'], fnn_vars['b2'], fnn_vars['b3'], fnn_vars['b4']]

    fnn_model = FNN()
    W, b = fnn_model.hyper_initial_target(layer_T, Wt, bt)
    u_T = fnn_model.fnn(W, b, x, Xmin, Xmax)

    u_nn = u_B*u_T
    u_pred = tf.reduce_sum(u_nn, axis=-1, keepdims=True)
    W_cnn_fnn = [Wf1] + [Wf2] + [Wf3] 
    regularizers = fnn_model.l2_regularizer(W_cnn_fnn)

    # c1_t = tf.math.sin(np.pi*c1*tf.exp(c1)) 
    #c2_t = tf.math.sin(np.pi*c2*tf.exp(c2))
    
    ############################
    # Loss function
    l2_loss = tf.reduce_sum(tf.norm(u_pred - u_ph, 2, axis=1)/tf.norm(u_ph, 2, axis=1)) +  beta*regularizers
    ceod_loss = loss2.CEOD([fnn_layer_1], u_ph, [fnn_layer_1_ph], u_pred_ph_s)   
    
    hybrid_loss = l2_loss + 10*ceod_loss  # total loss
    optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1 = 0.99)
    optimizer2 = tf.train.AdamOptimizer(learning_rate=5.0e-1, beta1 = 0.99, beta2 = 0.99)

    grads_W= optimizer1.compute_gradients(hybrid_loss, W[-1])
    grads_b = optimizer1.compute_gradients(hybrid_loss, b[-1])    
    grads_Wf1 = optimizer1.compute_gradients(hybrid_loss, Wf1)
    grads_bf1 = optimizer1.compute_gradients(hybrid_loss, bf1)
    grads_Wf2 = optimizer1.compute_gradients(hybrid_loss, Wf2)
    grads_bf2 = optimizer1.compute_gradients(hybrid_loss, bf2)
    grads_Wf3 = optimizer1.compute_gradients(hybrid_loss, Wf3)
    grads_bf3 = optimizer1.compute_gradients(hybrid_loss, bf3)
    # grads_c1 = optimizer2.compute_gradients(hybrid_loss, [c1])
    #grads_c2 = optimizer2.compute_gradients(hybrid_loss, [c2])
    
    # grads_c1_minus = [(-gv[0], gv[1]) for gv in grads_c1]
    #grads_c2_minus = [(-gv[0], gv[1]) for gv in grads_c2]
    
    op_W = optimizer1.apply_gradients(grads_W)
    op_b = optimizer1.apply_gradients(grads_b)
    op_Wf1 = optimizer1.apply_gradients(grads_Wf1)
    op_bf1 = optimizer1.apply_gradients(grads_bf1)
    op_Wf2 = optimizer1.apply_gradients(grads_Wf2)
    op_bf2 = optimizer1.apply_gradients(grads_bf2)
    op_Wf3 = optimizer1.apply_gradients(grads_Wf3)
    op_bf3 = optimizer1.apply_gradients(grads_bf3)    
    # op_c1 = optimizer2.apply_gradients(grads_c1_minus)
    # op_c2= optimizer2.apply_gradients(grads_c2_minus)

    # train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(hybrid_loss)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
    sess.run(tf.global_variables_initializer())
    
    n = 0
    nmax = 20000  # epochs
    time_step_0 = time.perf_counter()
    
    train_loss = np.zeros((nmax+1, 1))
    test_loss = np.zeros((nmax+1, 1))    
    while n <= nmax:
        
        if n <10000:
            lr = 0.0001
        elif (n < 20000):
            lr = 0.0005
        elif (n < 40000):
            lr = 0.0001
        else:
            lr = 0.00005
            
        x_train, f_train, u_train, _, _ = data.minibatch_target() # target data
        # If trained with CEOD uncomment the lines below
        x_train_source, f_train_source, u_train_source, _, _ = data.minibatch() # source data
        fnn_layer_1_s = sess.run(fnn_layer_1, feed_dict={f_ph:f_train_source}) 
        u_pred_s = sess.run(u_pred, feed_dict={f_ph:f_train_source}) 
        train_dict = {f_ph: f_train, fnn_layer_1_ph: fnn_layer_1_s, u_ph: u_train, u_pred_ph_s: u_pred_s, learning_rate: lr}
        sess.run([op_W,op_b,op_Wf1,op_bf1,op_Wf2,op_bf2,op_Wf3,op_bf3], train_dict)
        #sess.run([op_c2], train_dict)
        
        loss_ = sess.run(hybrid_loss, feed_dict=train_dict)
        
        if n%1 == 0:
            test_id, x_test, f_test, u_test = data.testbatch_target(bs)
            u_test_ = sess.run(u_pred, feed_dict={f_ph: f_test})
            u_test = data.decoder_target(u_test)
            u_test_ = data.decoder_target(u_test_)   
            err = np.mean(np.linalg.norm(u_test_ - u_test, 2, axis=1)/np.linalg.norm(u_test, 2, axis=1))
            time_step_1000 = time.perf_counter()
            T = time_step_1000 - time_step_0
            print('Step: %d, Loss: %.4e, Test L2 error: %.4f, Time (secs): %.4f'%(n, loss_, err, T))
            time_step_0 = time.perf_counter()
    
        train_loss[n,0] = loss_
        test_loss[n,0] = err
        n += 1
        
    stop_time = time.perf_counter()
    print('Total run time: ', stop_time)
    
    # Save data
    data_save = SaveData()
    num_test = 50
    data_save.save(sess, x_pos, fnn_model, W, b, Xmin, Xmax, u_B, f_ph, u_ph, data, num_test, save_results_to, domain='target')

    ## Plotting the loss history
    plt.rcParams.update({'font.size': 15})
    num_epoch = train_loss.shape[0]
    x = np.linspace(1, num_epoch, num_epoch)
    fig = plt.figure(constrained_layout=True, figsize=(7, 5))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(x, test_loss[:,0], color='red', label='Testing Loss')
    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend(loc='upper left')
    fig.savefig(save_results_to+'/target/loss_test.png')
   
    ## Save target test loss
    np.savetxt(save_results_to+'/target/loss_test', test_loss[:,0])
    np.savetxt(save_results_to+'/target/epochs', x)

    # Load source test loss
    test_loss_source = np.loadtxt(save_results_to+'/source/loss_test')
    epochs_source = np.loadtxt(save_results_to+'/source/epochs')
    
    ## Plotting both source and target loss
    plt.rcParams.update({'font.size': 15})
    num_epoch = train_loss.shape[0]
    x = np.linspace(1, num_epoch, num_epoch)
    fig = plt.figure(constrained_layout=True, figsize=(7, 5))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(x, test_loss[:,0], color='r', label='target')
    ax.plot(epochs_source, test_loss_source, color='b', label='source')
    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend(loc='upper left')
    fig.savefig(save_results_to+'/target/loss_test_comparison.png') 
    
########## NOT LOG PlOTS        
    plt.rcParams.update({'font.size': 15})
    num_epoch = train_loss.shape[0]
    x = np.linspace(1, num_epoch, num_epoch)
    fig = plt.figure(constrained_layout=True, figsize=(7, 5))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(x, test_loss[:,0], color='red', label='Testing Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend(loc='upper left')
    fig.savefig(save_results_to+'/target/loss_test_notlog.png')
    
    ## Plotting both source and target loss
    plt.rcParams.update({'font.size': 15})
    num_epoch = train_loss.shape[0]
    x = np.linspace(1, num_epoch, num_epoch)
    fig = plt.figure(constrained_layout=True, figsize=(7, 5))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(x, test_loss[:,0], color='r', label='target')
    ax.plot(epochs_source, test_loss_source, color='b', label='source')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend(loc='upper left')
    fig.savefig(save_results_to+'/target/loss_test_comparison_notlog.png')


if __name__ == "__main__":
    main()
