'''
Manuscript Associated: Deep transfer operator learning for partial differential equations under conditional shift
Authors: Katiana Kontolati, PhD Candidate, Johns Hopkins University
         Somdatta Goswami, Postdoctoral Researcher, Brown University
Tensorflow Version Required: TF1.15     
This should be used for sharp data    

This is the source model.
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
import shutil
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
# Remove existing results
if os.path.exists(save_results_to):
    shutil.rmtree(save_results_to)
    shutil.rmtree(save_variables_to)

os.makedirs(save_results_to) 
os.makedirs(save_variables_to) 

np.random.seed(1234)

#output dimension of Branch/Trunk (latent dimension)
p = 150

#fnn in CNN
layer_B = [512, 512, p]
#trunk net
layer_T = [3, 128, 128, 128, p]
#resolution
h = 28
w = 28
num = h*w
#parameters in CNN
n_channels = 1
#n_out_channels = 16
filter_size_1 = 8
filter_size_2 = 8
filter_size_3 = 8
filter_size_4 = 8
filter_size_5 = 8
stride = 1

#filter size for each convolutional layer
num_filters_1 = 16
num_filters_2 = 16
num_filters_3 = 16
num_filters_4 = 16
num_filters_5 = 32
#batch_size
bs = 100

#size of input for Trunk net
nx = h
nt = 10
x_num = nt*nx*nx
beta = 0.0009
def main():
    
    data = DataSet(nx, bs)
    x_train, f_train, u_train, Xmin, Xmax = data.minibatch()

    x_pos = tf.constant(x_train, dtype=tf.float32)
    x = tf.tile(x_pos[None, :, :], [bs, 1, 1]) #[bs, x_num, x_dim]

    f_ph = tf.placeholder(shape=[None, h, w, n_channels], dtype=tf.float32) #[bs, 1, h, w, n_channels]
    u_ph = tf.placeholder(shape=[None, x_num, 1], dtype=tf.float32) #[bs, x_num, 1]
    learning_rate = tf.placeholder(tf.float32, shape=[])
    
    lambda_u0 = tf.Variable(tf.random_normal(shape = [1, 1], dtype = tf.float32), dtype = tf.float32)
    lambda_u5 = tf.Variable(tf.random_normal(shape = [1, 1], dtype = tf.float32), dtype = tf.float32)
    lambda_u6 = tf.Variable(tf.random_normal(shape = [1, 1], dtype = tf.float32), dtype = tf.float32)
    
    # Branch net
    conv_model = CNN()

    #conv_linear = conv_model.linear_layer(f_ph, n_out_channels)
    conv_1, W1, b1 = conv_model.conv_layer(f_ph, filter_size_1, num_filters_1, stride, actn=tf.nn.relu)  
    pool_1 = conv_model.avg_pool(conv_1, ksize=2, stride=2)  
    conv_2, W2, b2 = conv_model.conv_layer(pool_1, filter_size_2, num_filters_2, stride, actn=tf.nn.relu)
    pool_2 = conv_model.avg_pool(conv_2, ksize=2, stride=2) 
    conv_3, W3, b3 = conv_model.conv_layer(pool_2, filter_size_3, num_filters_3, stride, actn=tf.nn.relu)
    pool_3 = conv_model.avg_pool(conv_3, ksize=2, stride=2)
    conv_4, W4, b4 = conv_model.conv_layer(pool_3, filter_size_4, num_filters_4, stride, actn=tf.nn.relu)
    pool_4 = conv_model.avg_pool(conv_4, ksize=2, stride=2) 
    layer_flat = conv_model.flatten_layer(pool_4)

    fnn_layer_1, Wf1, bf1 = conv_model.fnn_layer(layer_flat, layer_B[0], actn=tf.tanh, use_actn=True)
    fnn_layer_2, Wf2, bf2 = conv_model.fnn_layer(fnn_layer_1, layer_B[1], actn=tf.nn.tanh, use_actn=True)
    out_B, Wf3, bf3 = conv_model.fnn_layer(fnn_layer_2, layer_B[-1], actn=tf.tanh, use_actn=False) #[bs, p]
    u_B = tf.tile(out_B[:, None, :], [1, x_num, 1]) #[bs, x_num, p]
    
    # Trunk net
    fnn_model = FNN()
    W, b = fnn_model.hyper_initial(layer_T)
    u_T = fnn_model.fnn(W, b, x, Xmin, Xmax)
    u_nn = u_B*u_T

    u_pred = tf.reduce_sum(u_nn, axis=-1, keepdims=True)

    lambda_u0_t = tf.math.sin(np.pi*lambda_u0*tf.exp(lambda_u0))
    lambda_u5_t = tf.math.sin(np.pi*lambda_u5*tf.exp(lambda_u5))
    lambda_u6_t = tf.math.sin(np.pi*lambda_u6*tf.exp(lambda_u6))
    
    CNN_weights = [W1] + [W2] + [W3] + [W4] +[Wf1] + [Wf2] + [Wf3]
    regularizers = fnn_model.l2_regularizer(CNN_weights)

    loss = tf.reduce_mean(tf.square(u_ph - u_pred)) + tf.reduce_mean(tf.square(lambda_u5_t)*tf.square(u_ph[:,5*num:6*num,:] - u_pred[:,5*num:6*num,:])) + \
           tf.reduce_mean(tf.square(lambda_u6_t)*tf.square(u_ph[:,6*num:7*num,:] - u_pred[:,6*num:7*num,:])) + \
           tf.reduce_mean(tf.square(lambda_u0_t)*tf.square(u_ph[:,0:num,:] - u_pred[:,0:num,:])) + beta*regularizers
           
    optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1 = 0.99)
    optimizer2 = tf.train.AdamOptimizer(learning_rate=5.0e-1, beta1 = 0.99, beta2 = 0.99)
    
    grads_W= optimizer1.compute_gradients(loss, W)
    grads_b = optimizer1.compute_gradients(loss, b)   
    grads_W1 = optimizer1.compute_gradients(loss, W1)
    grads_b1 = optimizer1.compute_gradients(loss, b1)
    grads_W2 = optimizer1.compute_gradients(loss, W2)
    grads_b2 = optimizer1.compute_gradients(loss, b2)
    grads_W3 = optimizer1.compute_gradients(loss, W3)
    grads_b3 = optimizer1.compute_gradients(loss, b3)
    grads_W4 = optimizer1.compute_gradients(loss, W4)
    grads_b4 = optimizer1.compute_gradients(loss, b4)    
    grads_Wf1 = optimizer1.compute_gradients(loss, Wf1)
    grads_bf1 = optimizer1.compute_gradients(loss, bf1)
    grads_Wf2 = optimizer1.compute_gradients(loss, Wf2)
    grads_bf2 = optimizer1.compute_gradients(loss, bf2)
    grads_Wf3 = optimizer1.compute_gradients(loss, Wf3)
    grads_bf3 = optimizer1.compute_gradients(loss, bf3)
    grads_lambda_u0 = optimizer2.compute_gradients(loss, [lambda_u0])
    grads_lambda_u5 = optimizer2.compute_gradients(loss, [lambda_u5])
    grads_lambda_u6 = optimizer2.compute_gradients(loss, [lambda_u6])
    
    grads_lambda_u0_minus = [(-gv[0], gv[1]) for gv in grads_lambda_u0]
    grads_lambda_u5_minus = [(-gv[0], gv[1]) for gv in grads_lambda_u5]
    grads_lambda_u6_minus = [(-gv[0], gv[1]) for gv in grads_lambda_u6]
    
    op_W = optimizer1.apply_gradients(grads_W)
    op_b = optimizer1.apply_gradients(grads_b)
    op_W1 = optimizer1.apply_gradients(grads_W1)
    op_b1 = optimizer1.apply_gradients(grads_b1)
    op_W2 = optimizer1.apply_gradients(grads_W2)
    op_b2 = optimizer1.apply_gradients(grads_b2)
    op_W3 = optimizer1.apply_gradients(grads_W3)
    op_b3 = optimizer1.apply_gradients(grads_b3)
    op_W4 = optimizer1.apply_gradients(grads_W4)
    op_b4 = optimizer1.apply_gradients(grads_b4)
    op_Wf1 = optimizer1.apply_gradients(grads_Wf1)
    op_bf1 = optimizer1.apply_gradients(grads_bf1)
    op_Wf2 = optimizer1.apply_gradients(grads_Wf2)
    op_bf2 = optimizer1.apply_gradients(grads_bf2)
    op_Wf3 = optimizer1.apply_gradients(grads_Wf3)
    op_bf3 = optimizer1.apply_gradients(grads_bf3)   
    op_lamU0 = optimizer2.apply_gradients(grads_lambda_u0_minus)
    op_lamU5 = optimizer2.apply_gradients(grads_lambda_u5_minus)
    op_lamU6 = optimizer2.apply_gradients(grads_lambda_u6_minus)
    
    
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
    sess.run(tf.global_variables_initializer())
    
    n = 0
    nmax = 150000#35000  # epochs
    start_time = time.perf_counter()
    time_step_0 = time.perf_counter()
    
    train_loss = np.zeros((nmax+1, 1))
    test_loss = np.zeros((nmax+1, 1))    
    while n <= nmax:
        
        if n <10000:
            lr = 0.001
        elif (n < 20000):
            lr = 0.0005
        elif (n < 40000):
            lr = 0.0001
        else:
            lr = 0.00005
        x_train, f_train, u_train, _, _ = data.minibatch()
        train_dict = {f_ph: f_train, u_ph: u_train, learning_rate: lr}
        sess.run([op_W,op_b,op_W1,op_b1,op_W2,op_b2,op_W3,op_b3,op_W4,op_b4,op_Wf1,op_bf1,op_Wf2,op_bf2,op_Wf3,op_bf3], train_dict)
        sess.run([op_lamU0, op_lamU5, op_lamU6], train_dict)
        
        loss_, lambda_u0_, lambda_u5_, lambda_u6_ = sess.run([loss, lambda_u0, lambda_u5, lambda_u6], feed_dict=train_dict)
    
    
        if n%1 == 0:
            test_id, x_test, f_test, u_test = data.testbatch(bs)
            u_test_ = sess.run(u_pred, feed_dict={f_ph: f_test})
            u_test = data.decoder(u_test)
            u_test_ = data.decoder(u_test_)
            err = np.mean(np.linalg.norm(u_test_ - u_test, 2, axis=1)/np.linalg.norm(u_test, 2, axis=1))
            time_step_1000 = time.perf_counter()
            T = time_step_1000 - time_step_0
            print('Step: %d, Loss: %.4e, Test L2 error: %.4f, Time (secs): %.4f'%(n, loss_, err, T))
            time_step_0 = time.perf_counter()
    
        train_loss[n,0] = loss_
        test_loss[n,0] = err
        n += 1
    stop_time = time.perf_counter()
    print('Elapsed time (secs): %.3f'%(stop_time - start_time))

    # Save variables  
    W1_,b1_,W2_,b2_,W3_,b3_,W4_,b4_,Wf1_,bf1_,Wf2_,bf2_,Wf3_,bf3_ = \
        sess.run([W1,b1,W2,b2,W3,b3,W4,b4,Wf1,bf1,Wf2,bf2,Wf3,bf3])
    
    savedict_cnn = {'W1':W1_,'b1':b1_,'W2':W2_,'b2':b2_,'W3':W3_,'b3':b3_,'W4':W4_,'b4':b4_,\
                'Wf1':Wf1_,'bf1':bf1_,'Wf2':Wf2_,'bf2':bf2_,'Wf3':Wf3_,'bf3':bf3_}    
    
    Wt1, bt1, Wt2, bt2, Wt3, bt3, Wt4, bt4,= sess.run([W[0], b[0], W[1], b[1], W[2], b[2], W[3], b[3]])
    
    savedict_fnn = {'W1':Wt1,'b1':bt1,'W2':Wt2,'b2':bt2,'W3':Wt3,'b3':bt3,'W4':Wt4,'b4':bt4}    
    
    # Save variables (weights + biases)
    io.savemat(save_variables_to+'/CNN_vars.mat', mdict=savedict_cnn)
    io.savemat(save_variables_to+'/FNN_vars.mat', mdict=savedict_fnn)
    
    data_save = SaveData()
    num_test = 200
    data_save.save(sess, x_pos, fnn_model, W, b, Xmin, Xmax, u_B, f_ph, u_ph, data, num_test, save_results_to, domain='source')
    
    ## Plotting the loss history
    plt.rcParams.update({'font.size': 15})
    num_epoch = train_loss.shape[0]
    x = np.linspace(1, num_epoch, num_epoch)    
    fig = plt.figure(constrained_layout=True, figsize=(7, 5))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(x, train_loss[:,0], color='blue', label='Training Loss')
    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend(loc='upper left')
    fig.savefig(save_results_to+'/source/loss_train.png')
    
    ## Save test loss
    np.savetxt(save_results_to+'/source/loss_test', test_loss[:,0])
    np.savetxt(save_results_to+'/source/epochs', x)
    
    fig = plt.figure(constrained_layout=True, figsize=(7, 5))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(x, test_loss[:,0], color='red', label='Testing Loss')
    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend(loc='upper left')
    fig.savefig(save_results_to+'/source/loss_test.png')

########## NOT LOG PlOTS
    plt.rcParams.update({'font.size': 15})
    num_epoch = train_loss.shape[0]
    x = np.linspace(1, num_epoch, num_epoch)    
    fig = plt.figure(constrained_layout=True, figsize=(7, 5))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(x, train_loss[:,0], color='blue', label='Training Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend(loc='upper left')
    fig.savefig(save_results_to+'/source/loss_train_notlog.png')
    
    fig = plt.figure(constrained_layout=True, figsize=(7, 5))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(x, test_loss[:,0], color='red', label='Testing Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend(loc='upper left')
    fig.savefig(save_results_to+'/source/loss_test_notlog.png')


if __name__ == "__main__":
    main()
