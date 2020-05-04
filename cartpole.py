import gym
import random
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

GAMMA = 0.95
EPSILON_MAX = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
BATCH_SIZE = 20
ALPHA = 0.001
N_EPS = 1000

class DQN:
        def __init__(self,state_space,action_space):
                self.epsilon = EPSILON_MAX
                self.action_space = action_space
                self.history = []
                self.model = Sequential()
                self.model.add(Dense(24, input_shape=(state_space,), activation="relu"))
                self.model.add(Dense(24, activation="relu"))
                self.model.add(Dense(action_space, activation="linear"))
                self.model.compile(loss="mse", optimizer=Adam(lr=ALPHA))    
                
        def remember(self,state, action, reward, next_state, terminal):
                self.history.append([state, action, reward, next_state, terminal])

        def act(self,state):
                if random.random() <= self.epsilon:
                        return random.randrange(self.action_space)
                q_vals = self.model.predict(state)
                return np.argmax(q_vals[0])


        def experience_replay(self):
                if len(self.history) < BATCH_SIZE:
                        return
                batches = random.sample(self.history, BATCH_SIZE)
                for state, action, reward, next_state, terminal in batches:
                        y = reward
                        if not terminal:
                                y = (reward + GAMMA * np.amax(self.model.predict(next_state)[0]))
                        y_target = self.model.predict(state)
                        y_target[0][action] = y
                        
                        self.model.fit(x=state,y=y_target,verbose=0)
                if self.epsilon > EPSILON_MIN:
                        self.epsilon *= EPSILON_DECAY



def run():
        env = gym.make('CartPole-v0')
        dqn = DQN(env.observation_space.shape[0],env.action_space.n)
        scores = []
        for i_episode in range(N_EPS):
                state = env.reset()
                t = 0
                tot_reward = 0
                while True:
                        #env.render()
                        state = np.reshape(state, [1, env.observation_space.shape[0]])
                        action = dqn.act(state)
                        #action = env.action_space.sample()
                        next_state, reward, terminal, info = env.step(action)
                        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
                        dqn.remember(state, action, reward, next_state, terminal)
                        reward = reward if not terminal else -reward
                        state = next_state
                        if terminal:
                                print("Episode {} finished after {} timesteps with reward {}".format(i_episode+1,t+1,tot_reward))
                                break  
                        dqn.experience_replay()
                        tot_reward += reward
        
                        t+=1
                scores.append(tot_reward)
                if i_episode%100 == 0:
                        print("The mean reward after {} episodes is {}".format(i_episode,np.mean(scores)))
                

        env.close()

