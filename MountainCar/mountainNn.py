# based Q - learining

import gym 
import numpy as np 
import matplotlib.pyplot as plt 
from NNmodel import nnModel
import pickle
import tensorflow as tf 

def epsilon_greedy(model, s, eps = 0.1, is_training = False):
    p = np.random.random()
    if p < (1-eps):
        values = model.predict(s, is_training = is_training)
        return np.argmax(values)
    else:
        return model.env.action_space.sample()
        
    


class Agent(object):
    def __init__(self,nnStructure, env):
        self.env = env
        
        self.model = nnModel(input_dims = self.env.observation_space.shape[0],
                             n_actions = self.env.action_space.n, nnStructure = nnStructure)
        
        
    def act(self, state, epsilon , is_training = False):
        if np.random.rand() <= epsilon:
            return np.random.choice(self.env.action_space.n)
        act_values = self.model.predict(state, is_training = is_training)
        return np.argmax(act_values[0])
    
    def train(self, n_episodes = 5000, alpha = 0.1,
              gamma = 0.99, plotRewardsPerEpisodes = True):
        reward_per_episode = []
        
        # alpha =  # was 0.1 for the catpole-v0
        #repeat until convergence 
        
        for it in range(n_episodes):
            s = self.env.reset()
            
            s = np.array(s).astype('float32')
            s = np.expand_dims(s, axis = 0)
            
            episode_reward = 0
            done = False
            if (it+1) % 1000 == 0:
                alpha = alpha /2
            while not done :
                a = self.act(s, epsilon = 0.1, is_training= True )
                s2, r, done, info = self.env.step(a)
                s2 = np.array(s2).astype('float32')
                s2 = np.expand_dims(s2, axis = 0)
                # get the target
                if done:
                    target = r
                else:
                    value = np.amax(self.model.predict(X = s2, is_training = True), axis = 1)
                    # values = self.model.predict_all_actions(s2)
                    target = r + gamma * value


                target_full = self.model.predict(X = s, is_training = True)
                target_full = np.array(target_full.numpy(), dtype ='float32')
                target_full[0,a] = target
                
                
                self.model.train(X = s, Y =  target_full)
                # accumilate reward
                episode_reward += r
                
                # update state
                s = s2
                
            if (it + 1) % 50 ==0:
                print(f"Episode: {it+1}, reward: {episode_reward}")
            # early exit 
            if it > 60 and np.mean(reward_per_episode[-20:]) >= 300:
                print("Early exit")
                break
            
            reward_per_episode.append(episode_reward)
            
        # test trained agent 
        # test_reward = self.test_agent(self.model, env)
        # print(f"Average test reward: {test_reward}")
        
        if plotRewardsPerEpisodes:
            plt.plot(reward_per_episode)
            plt.title("Reward per episode")
            plt.show()
        
            
        
            
    def test_agent(self, model, env, n_episodes = 20):
        reward_per_episode = np.zeros(n_episodes)
        for it in range(n_episodes):
            done = False
            episode_reward = 0
            s = self.env.reset()
            s = np.array(s).astype('float32')
            s = np.expand_dims(s, axis = 0)
            while not done:
                a = self.act(s, epsilon = 0, is_training=False) # follow your best policy 
                s, r, done,info = env.step(a)
                episode_reward += r
                s = np.array(s).astype('float32')
                s = np.expand_dims(s, axis = 0)
                
            reward_per_episode[it] = episode_reward
        return np.mean(reward_per_episode)
    
    def watch_agent(self, env, eps, name = 'vid', saveVideo = False):
        done = False
        episode_reward = 0
        if saveVideo:
            env = gym.wrappers.Monitor(env, name ,force=True)
            
        s = env.reset()
        s = np.array(s).astype('float32')
        s = np.expand_dims(s, axis = 0)
        while not done:
            a = self.act( s, epsilon = eps ,  is_training=False)
            s, r, done, info = env.step(a)
            s = np.array(s).astype('float32')
            s = np.expand_dims(s, axis = 0)
            
            env.render()
            episode_reward += r
        print("Episode reward:", episode_reward)
        
    
if __name__ == "__main__":
    # instantiate environment 
    env = gym.make("MountainCar-v0")# can also be v0 atthe end 
    
    activationFunc = tf.nn.relu
    apply_batch_norm = False
    
    agent = Agent(nnStructure= [[20, activationFunc, apply_batch_norm],
                                [20, activationFunc, apply_batch_norm]], 
                  env = env)
    agent.train(n_episodes= 100, alpha= 0.1, gamma= 0.99)
    
    # # watch untrained agent (only if you wish)
    # watch_agent(model, env, eps = 0, name = 'untrained')
    
    # # model.train(alpha=0.1, n_episodes= 1000,plotRewardsPerEpisodes= True)
    
    # # save the model weights and features
    # # model.save_weights()
    
    # # load teh trained weights (trained agent)
    
    
    # # Watch trained aget 

    agent.watch_agent(env, eps = 0, name = 'trained', saveVideo=True )
    env.close()
    
            
            
            
            
        
            
    
    
    
    
    
    
    

        