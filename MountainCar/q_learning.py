
# Here we are going to implement Q-learning with RBF network to solve mountaincar

# Note: gym change from version 0.7.3 to 0.8.0
# MountainCar episode length is capped at 200 in later versions 
# THis means your agent can't learn as much in the earlier episodes 
# since they are no longer as long 

import pickle
import gym
import os
import sys
import time
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib 
from mpl_toolkits.mplot3d import Axes3D
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor


# SGDRegressor defaults:
    # loss = 'squred_loss', penalty = 'l2', alpha  = 1e-4
    # l1_ration = 0.15, fit_intercept = TRue, n_iter = 5, shuffle = True
    # verbose = 0, epsilon= 0.1, random_state = None, learning_rate = 'invscaling'
    # eta0 = 0.01, power_t = 0.25, warm_start = False, average = False
    # THE LEARNING RATE IS eta0
    

# Inspired by https://github.com/dennybritz/reinforcement-learning

class FeatureTransformer:
    def __init__(self, env, n_components = 500):
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        scaler = StandardScaler()
        scaler.fit(observation_examples) # just for scaling 
        
        # usedf to convert a state to a featurized representation.
        # We use RBF kernels with different variances to cover different parts of the space
        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma = 5.0, n_components = n_components)),
            ("rbf2", RBFSampler(gamma = 2.0, n_components = n_components)),
            ("rbf3", RBFSampler(gamma = 1.0, n_components = n_components)),
            ("rbf4", RBFSampler(gamma = 0.5, n_components = n_components)),
            ]) ## in general we have 2000 features, RBF kernel by Monte Carlo approximation of its Fourier transform.
        
        example_features = featurizer.fit_transform(scaler.transform(observation_examples))
        
        self.dimensions = example_features.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer
        
    def transform(self, observations):
        # print "observations:", observations 
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)
    
class Model:
    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate = learning_rate)
            model.partial_fit(feature_transformer.transform([env.reset()] ), [0] ) ## here we set it for the first time
            self.models.append(model) # here we have model for each action 
        
        self.params = [self.models, self.feature_transformer]
        
    def predict(self, s):
        X = self.feature_transformer.transform([s])
        result = np.stack([m.predict(X) for m in self.models]).T # now we have the linear prediction of each action
        assert(len(result.shape) == 2) # for checking the dimensionality 
        return result 
    
    def update(self, s, a, G):
        X = self.feature_transformer.transform([s])
        assert(len(X.shape) == 2)
        self.models[a].partial_fit(X, [G]) # we update only the model of this specific action 
        
    def sample_action(self, s, eps):
        # eps = 0
        # Technically, we don't need to do epsilon greedy 
        # because SGDRegressor predicts 0 for all states 
        # until they are updated. This works as the
        # "Optimistic initial Values" method, since all 
        # the rewards for the MountainCar are -1 
        
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))
    
    def save(self):
        with open(f'weights/trained.pickle', 'wb') as file:
            pickle.dump(self.params, file) 
            
    def load(self, file_name = 'weights/trained.pickle'):
        with open(file_name, 'rb') as file2:
            self.params = pickle.load(file2)
            
        
        self.models = self.params[0]
        self.feature_transformer = self.params[1]
        
            

# returns a list of states_and_rewards, and the total reward
def play_one(model, env, eps, gamma):
    observation = env.reset()
    done = False
    totalRewards = 0
    iters = 0
    
    while not done and iters < 10000:
        action = model.sample_action(observation, eps)
        prev_observation = observation 
        observation, reward, done, info = env.step(action)
        
        # update the model 
        nexti = model.predict(observation)
        # assert(nexti.shape == (1,env.action_space.n))
        G = reward + gamma * np.max(nexti[0])
        model.update(prev_observation, action, G)
        
        totalRewards += reward
        iters += 1
        
    return totalRewards

def watch_agent(model, env, eps = 0):
    
    observation = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        env.render()
        a = model.sample_action(observation, eps)
        observation, r, done, info = env.step(a)
        total_reward += 1
        time.sleep(0.02)
    return total_reward

def plot_cost_to_go(env, estimator, num_tiles = 20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num = num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num = num_tiles)
    X, Y = np.meshgrid(x, y)
    # both X and Y will be of shape (num_tiles, num_tiles)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
    # Z will also be of shape (num_tiles, num_tiles)
    
    fig = plt.figure(figsize = (10,5))
    ax = fig.add_subplot(111, projection = '3d')
    surf = ax.plot_surface(X, Y, Z, 
                           rstride = 1, cstride = 1, cmap = matplotlib.cm.coolwarm , vmin = -1.0, vmax = 1.0)
    plt.rcParams["font.family"] = "Times New Roman"
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-To-Go == -V(s)')
    ax.set_title("COst-To-Go Function")
    fig.colorbar(surf)
    plt.show()
    
def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0,t-100) : t+1].mean()
     
    
    plt.plot(np.arange(len(running_avg)), running_avg, label = "Running Avg 100")
    # plt.title("Running Average")
    plt.legend(frameon = False )
    plt.show()
    

def main(show_plots = True, watch_trained_agent = False, train = False, load_model = False):
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft, "constant")
    
    if train:
        gamma = 0.99
        
        if 'monitor' in sys.argv:
            filename = os.path.basename(__file__).split('.')[0]
            monitor_dir = './' + filename + str(datetime.now())
            env = wrappers.Monitor(env, monitor_dir)
            
        N = 300
        
        totalrewards = np.empty(N)
        
        for n in range(N):
            # eps = 1.0 / (0.1*n + 1)
            eps = 0.1*(0.97**n)
            
            if n == 199:
                print("eps:", eps)
                
            # eps = 1.0/np.sqrt(n+1)
            
            totalreward = play_one(model, env, eps, gamma)
            totalrewards[n] = totalreward
            
            if (n+1) % 100 == 0:
                print("episode:", n, "total reward:", totalreward)
                
        print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
        print("total steps:", -totalrewards.sum())
    
        plt.rcParams["font.family"] = "Times New Roman"
        if show_plots:
            plt.plot(np.arange(len(totalrewards)),  totalrewards, label = 'Rewards')
            plt.ylabel("Rewards")
            plt.xlabel("# iteration")
            plt.legend()
            plt.show()
            
            plot_running_avg(totalrewards)
            
            # plot the optimal state-value fucntion 
            plot_cost_to_go(env, estimator = model)
            
            model.save()
    else:
        if load_model:
            model.load()
          
    if watch_trained_agent:
        r = watch_agent(model, env)
        print("simulation reward:", r)
    
    
if __name__ == '__main__':
    # for i in range(10):
        # main(show_plots = False)
    main(train=False, load_model=True, watch_trained_agent= True)
    
    
    
    
    
        
    
    
    
        
        
        
        
        
            
            
    
    
    
    
        