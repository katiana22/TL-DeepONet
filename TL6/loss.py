'''
Authors: Katiana Kontolati, PhD Candidate, Johns Hopkins University
         Somdatta Goswami, Postdoctoral Researcher, Brown University
Tensorflow Version Required: TF1.15     
'''
import tensorflow.compat.v1 as tf
import sys

class CEOD_loss:
    def __init__(self, bs):
        self.bs = bs

    def kernel(self, X, X2, gamma=0.4):
        '''
        Input: X  Size1*n_feature (source inputs - output of first FNN layer, aftet the activation)
               X2 Size2*n_feature (target inputs)
        Output: Size1*Size2
        '''
        X = tf.transpose(X)
        X2 = tf.transpose(X2)

        n1, n2 = self.bs, self.bs

        X_sq = tf.math.square(X)
        n1sq = tf.math.reduce_sum(X_sq, axis=0)
        n1sq = tf.cast(n1sq, tf.float32)
        n2sq = tf.math.reduce_sum(X2**2, axis=0)

        D = tf.ones([n1, n2]) * n2sq + tf.transpose((tf.ones([n2, n1]) * n1sq)) +  - 2 * tf.linalg.matmul(tf.transpose(X), X2)
        K = tf.math.exp(-gamma * D)
        
        return K        
        
    def CEOD(self, X_p_list, Y_p, X_q_list, Y_q, lamda = 1):
        
        layer_num = 1
        out = 0
        for i in range(layer_num):
            
            X_p = X_p_list[i]
            X_q = X_q_list[i]  #[?,7840,1]
            
            Y_p = tf.reshape(Y_p, [-1, 7840])  #[?, 7840]
            Y_q = tf.reshape(Y_q, [-1, 7840])
            
            nps = self.bs #X_p.shape[0]
            nq = self.bs #X_q.shape[0]
            
            I1 = tf.eye(self.bs)
            I2 = tf.eye(self.bs)

            # Construct kernels 
            Kxpxp = self.kernel(X_p, X_p)
            Kxqxq = self.kernel(X_q, X_q)
            Kxqxp = self.kernel(X_q, X_p)
            Kypyq = self.kernel(Y_p, Y_q)
            Kyqyq = self.kernel(Y_q, Y_q)
            Kypyp = self.kernel(Y_p, Y_p)
            
            # Compute CEOD 
            a = tf.linalg.matmul((tf.linalg.inv(Kxpxp+nps*lamda*I1)),Kypyp)
            b = tf.linalg.matmul(a,(tf.linalg.inv(Kxpxp+nps*lamda*I1)))
            c = tf.linalg.matmul(b,Kxpxp)
            out1 = tf.linalg.trace(c)
            
            a1 = tf.linalg.matmul((tf.linalg.inv(Kxqxq+nq*lamda*I2)),Kyqyq)
            b1 = tf.linalg.matmul(a1,(tf.linalg.inv(Kxqxq+nq*lamda*I2)))
            c1 = tf.linalg.matmul(b1,Kxqxq)
            out2 = tf.linalg.trace(c1)
            
            a2 = tf.linalg.matmul((tf.linalg.inv(Kxpxp+nps*lamda*I1)),Kypyq)
            b2 = tf.linalg.matmul(a2,(tf.linalg.inv(Kxqxq+nq*lamda*I2)))
            c2 = tf.linalg.matmul(b2,Kxqxp)
            out3 = tf.linalg.trace(c2)       
            out += (out1 + out2 - 2*out3)     
        
        return out

