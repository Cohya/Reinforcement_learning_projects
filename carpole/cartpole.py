# based Q - learining

import gym 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.kernel_approximation import RBFSampler

GAMMA = 0.99
ALPHA = 0.01 # was 0.1 for the catpole-v0

def epsilon_greedy(model, s, eps = 0.1):
    p = np.random.random()
    if p < (1-eps):
        values = model.predict_all_actions(s)
        return np.argmax(values)
    else:
        return model.env.action_space.sample()
        
    
def gather_samples(env, n_episodes = 10000):
    samples = []
    
    for _ in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            a = env.action_space.sample()
            sa = np.concatenate((s, [a])) 
            samples.append(sa)
            
            s, r, done, info = env.step(a)
            
    return samples

class Model(object):
    def __init__(self, env):
        self.env = env
        samples = gather_samples(env) 
        self.featurizer = RBFSampler()
        self.featurizer.fit(samples)
        dims = self.featurizer.n_components
        
        # initialize linear model weights
        self.w = np.zeros(dims)
        
    def predict(self, s, a):
        sa = np.concatenate((s, [a]))
        x = self.featurizer.transform([sa])[0]
        return x @ self.w
    
    
    def predict_all_actions(self, s):
        return [self.predict(s,a) for a in range(self.env.action_space.n)]
    
    def grad(self, s,a, target):
        sa = np.concatenate((s,[a]))
        x = self.featurizer.transform([sa])[0]
        err = target - self.predict(s, a)
        g = err * x
        return g
    
    def train(self, n_episodes = 5000, plotRewardsPerEpisodes = True):
        reward_per_episode = []
        
        
        #repeat until convergence 
        
        for it in range(n_episodes):
            s = env.reset()
            episode_reward = 0
            done = False
            # if it % 1000:
            #     ALPHA = ALPHA /10
            while not done :
                a = epsilon_greedy(model, s)
                s2, r, done, info = env.step(a)
                
                # get the target
                if done:
                    target = r
                else:
                    values = model.predict_all_actions(s2)
                    target = r + GAMMA * np.max(values)
                    
                # update the model 
                g = model.grad(s,a, target)
                model.w += ALPHA * g
                
                # accumilate reward
                episode_reward += r
                
                # update state
                s = s2
                
            if (it + 1) % 50 ==0:
                print(f"Episode: {it+1}, reward: {episode_reward}")
                
            # early exit 
            if it > 60 and np.mean(reward_per_episode[-60:]) == 200:
                print("Early exit")
                break
            
            reward_per_episode.append(episode_reward)
            
        # test trained agent 
        test_reward = test_agent(model, env)
        print(f"Average test reward: {test_reward}")
        
        if plotRewardsPerEpisodes:
            plt.plot(reward_per_episode)
            plt.title("Reward per episode")
            plt.show()
        
    
    
def test_agent(model, env, n_episodes = 20):
    reward_per_episode = np.zeros(n_episodes)
    for it in range(n_episodes):
        done = False
        episode_reward = 0
        s = env.reset()
        
        while not done:
            a = epsilon_greedy(model, s, eps = 0) # follow your best policy 
            s, r, done,info = env.step(a)
            episode_reward += r
            
        reward_per_episode[it] = episode_reward
    return np.mean(reward_per_episode)

def watch_agent(model, env, eps, name = 'vid', saveVideo = False):
    done = False
    episode_reward = 0
    if saveVideo:
        env = gym.wrappers.Monitor(env, name ,force=True)
        
    s = env.reset()
    
    while not done:
        a = epsilon_greedy(model, s, eps = eps)
        s, r, done, info = env.step(a)
        env.render()
        episode_reward += r
    print("Episode reward:", episode_reward)
    
    
if __name__ == "__main__":
    # instantiate environment 
    env = gym.make("CartPole-v1")# can also be v0 atthe end 
    
    
    model = Model(env)
    # # watch untrained agent (only if you wish)
    # # watch_agent(model, env, eps = 0, name = 'untrained')
    
    
    model.train(plotRewardsPerEpisodes= True)
    
    # Watch trained aget 
    
    watch_agent(model, env, eps = 0, name = 'trained')
    
    
            
            
            
            
        
            
    
    
    
    
    
    
    

        