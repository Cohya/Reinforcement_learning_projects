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
                             n_actions = self.env.action_space.n, nnStructure = nnStructure, lr = 0.0001)
        
        
    def act(self, state, epsilon , is_training = False):
        if np.random.rand() <= epsilon:
            return np.random.choice(self.env.action_space.n)
        act_values = self.model.predict(state, is_training = is_training)
        return np.argmax(act_values[0])
    
    
    def costumReward(self, r, s2,s):
        # print(old_state, "dsfg", s)
        # print("Delta position:", (s[0][0] - old_state[0][0]), "velocity:", s[0][1])
        # if s[0][0] >= 0.5:
        #     r = 10000
            
        #     print("Win")
        # # elif ((s[0][0] - old_state[0][0]) > 0 and s[0][1] > 0) or ((s[0][0] - old_state[0][0]) < 0 and s[0][1] < 0):
        # #         r = 15# * abs(s[0][0] - old_state[0][0]) 
        # #         # print("r234")
        # else:
        #     # print("r3",s[0][0] )
        #     r = -10 
            
        r =  100 * ((np.sin(3 * s2[0][0]) * 0.0025 + 0.5 * s2[0][1] * s2[0][1]) -
                    (np.sin(3 * s[0][0]) * 0.0025 + 0.5 * s[0][1] * s[0][1])) 
        
        if s2[0][0] >= 0.5:
            r += 1
        return r
                
            
        
    def train(self, n_episodes = 5000, alpha = 0.1,
              gamma = 0.99, plotRewardsPerEpisodes = True):
        reward_per_episode = []
        
        # alpha =  # was 0.1 for the catpole-v0
        #repeat until convergence 
        update_counter = 0
        epsilon = 1
        for it in range(n_episodes):
            s = self.env.reset()
            
            s = np.array(s).astype('float32')
            s = np.expand_dims(s, axis = 0)
            
            episode_reward = 0
            done = False
            if (it+1) % 1000 == 0:
                alpha = alpha /2
            while not done :
                a = self.act(s, epsilon = epsilon, is_training= True )
                s2, r, done, info = self.env.step(a)
                
                
                s2 = np.array(s2).astype('float32')
                s2 = np.expand_dims(s2, axis = 0)
                
                r = self.costumReward(r, s2, s)
                # print("s2:", s2, "s", s)
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
            epsilon = 0.01 + (1 - 0.01) * np.exp(- 0.001 * (update_counter))
            update_counter += 1
            if epsilon < 0.01:
                epsilon = 0.01
            if (it + 1) % 10 ==0:
                print(f"Episode: {it+1}, reward: {episode_reward}")
            # early exit 
            # if it > 60 and np.mean(reward_per_episode[-20:]) >= 300:
            #     print("Early exit")
            #     break
            
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
                s2, r, done,info = env.step(a)
                
                s2 = np.array(s).astype('float32')
                s2 = np.expand_dims(s, axis = 0)
                
                r = self.costumReward(r, s2, s)
                
                episode_reward += r
                
                s = s2

                
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
    
    agent = Agent(nnStructure= [[64, activationFunc, apply_batch_norm],
                                [128, activationFunc, apply_batch_norm],
                                ], 
                  env = env)
    agent.train(n_episodes= 15, alpha= 0.1, gamma= 0.99)
    
    # # watch untrained agent (only if you wish)
    # watch_agent(model, env, eps = 0, name = 'untrained')
    
    # # model.train(alpha=0.1, n_episodes= 1000,plotRewardsPerEpisodes= True)
    
    # # save the model weights and features
    # # model.save_weights()
    
    # # load teh trained weights (trained agent)
    
    
    # # Watch trained aget 

    agent.watch_agent(env, eps = 0, name = 'trained', saveVideo=True )

    
            
            
            
            
        
            
    
    
    
    
    
    
    

        