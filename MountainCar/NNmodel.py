
import numpy as np
import tensorflow as tf 

import pickle


class DenseLayer(object):
    def __init__(self, name, M1, M2, apply_batch_norm, f = tf.nn.relu):
        
        self.name = name 
        self.W = tf.Variable(initial_value= tf.random.normal(shape = [M1, M2], stddev= 0.02), 
                             name = "W_%s" % name)
        
        self.b = tf.Variable(initial_value= tf.zeros(shape = [M2,]), 
                             name = "b_%s" % name)
        
        self.apply_batch_norm = apply_batch_norm
        
        if self.apply_batch_norm:
            self.gamma = tf.Variable(initial_value= tf.ones(shape = [M2,]), name = "gamma_%s" % name)
            self.beta = tf.Variable(initial_value= tf.zeros(shape = [M2,]), name  = "beta_%s" % name)
            
            self.running_mean = tf.Variable(initial_value= tf.zeros(shape = [M2,]), 
                                                           name = "running_mean_%s" % name,
                                                           trainable = False)
            self.running_var = tf.Variable(initial_value= tf.zeros(shape = [M2,]), 
                                           name = "running_var_%s" % name, 
                                           trainable = False)
            
            self.normalization_parmas = [self.running_mean, self.running_var]
            
        
        self.f = f
        self.name = name 
        
        self.params = [self.W, self.b]
        
        if self.apply_batch_norm:
            self.params += [self.gamma, self.beta]
            
        
    @tf.function
    def forward(self, X, is_training, decay = 0.99):
        Z = tf.matmul(X, self.W) + self.b
        
        if self.apply_batch_norm:
            if is_training:
                
                batch_mean, batch_var = tf.nn.moments(Z, [0])
                self.running_mean.assign(self.running_mean * decay + batch_mean * (1-decay))
                self.running_var.assign(self.running_var * decay + batch_var * (1-decay))
                self.normalization_parmas = [self.running_mean, self.running_var]
                
                Z = tf.nn.batch_normalization(Z, mean = batch_mean,
                                              variance = batch_var,
                                              offset = self.beta,
                                              scale = self.gamma,
                                              variance_epsilon = 1e-3)
                
            else:
                Z = tf.nn.batch_normalization(Z,
                                              mean = self.running_mean,
                                              variance = self.running_var,
                                              offset = self.beta,
                                              scale = self.gamma,
                                              variance_epsilon = 1e-3)
                
                
        return self.f(Z)
    
    
    
    
class nnModel(object):
    def __init__(self, input_dims, n_actions, nnStructure, lr = 0.0001):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.nnStructure = nnStructure #[[20, activationFunc, apply_batch_norm]]
        self.optimizer = tf.keras.optimizers.SGD(learning_rate = lr)#Adam
        
        ## let's build the nn 
        self.layers = []
        self.losses = []
        self.sess = tf.compat.v1.Session()
        M1 = self.input_dims
        
        for i in range(len(nnStructure)):
            M2 = nnStructure[i][0]
            activation = nnStructure[i][1]
            apply_batch_norm = nnStructure[i][2]
            layer = DenseLayer(name = str(i), M1 = M1, M2 = M2  , f = activation, apply_batch_norm = apply_batch_norm)
            self.layers.append(layer)
            M1 = M2
            
            
        
        # let's build the last layer 
        layer = DenseLayer(name = "Dense_last_layer",
                           M1 = M1,
                           M2 = self.n_actions,
                           f = lambda x: x,
                           apply_batch_norm = False)
        
        
        self.layers.append(layer)
        
        self.trainable_params = []
        
        for layer in self.layers:
            self.trainable_params += layer.params
            
    @tf.function 
    def forward(self, X, is_training = False):
        
        Z = X
        # print(Z.shape)
        for layer in self.layers:
            
            Z = layer.forward(Z, is_training  = is_training)
            # print(Z.shape)
        return Z
    
    @tf.function
    def predict(self, X, is_training):
        return self.forward(X, is_training = is_training)
    
    
    def cost_fun(self, Y_hat, Y):
        return tf.reduce_mean(tf.math.square((Y_hat - Y)))
    
    @tf.function
    def train(self, X, Y, learning_rate = 0.001):
        
        # self.optimizer(learning_rate = learning_rate, beta_1 = 0.9, beta_2 = 0.999)
        # self.optimizer(learning_rate = 0.01)
        # Y_hat = self.predict(X, is_training = True)
        # print("Y_hat:", Y_hat)
        with tf.GradientTape(watch_accessed_variables= True) as tape:
            # print((Y_hat)**2)
            # cost = self.cost_fun(Y_hat, Y)
            cost = tf.reduce_mean( tf.math.square(self.forward(X, is_training = True) -  Y)) 
            # print("cost:", cost)
        
        # print(cost)
        self.losses.append(cost)#cost.numpy())
        gradients = tape.gradient(cost, self.trainable_params)
        # print("g:", gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_params))
        
        return cost 
    
    
    def load_weights(self, filepath):
        with open(filepath, 'rb') as model_params:
            params_loaded = pickle.load(model_params)
            
        count = 0
        for param in self.trainable_params:
            print(param)
            param.assign(params_loaded[count])
            count += 1
            print(param)
            
    def save_weights(self, filepath):
        with open(filepath, 'wb') as model_params:
            pickle.dump(self.trainable_params, model_params)
            
            
