
import tensorflow as tf

import gym
from keras.api._v2.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import cv2

env = gym.make("LunarLanderContinuous-v2", render_mode='rgb_array')

class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.f1 = tf.keras.layers.Dense(512, activation='relu')
        self.f2 = tf.keras.layers.Dense(512, activation='relu')
        self.v =  tf.keras.layers.Dense(1, activation=None)

    def call(self, inputstate, action):
        x = self.f1(tf.concat([inputstate, action], axis=1))
        x = self.f2(x)
        x = self.v(x)
        return x


class Actor(tf.keras.Model):
    def __init__(self, no_action):
        super(Actor, self).__init__()    
        self.f1 = tf.keras.layers.Dense(512, activation='relu')
        self.f2 = tf.keras.layers.Dense(512, activation='relu')
        self.mu =  tf.keras.layers.Dense(no_action, activation='tanh')

    def call(self, state):
        x = self.f1(state)
        x = self.f2(x)
        x = self.mu(x)  
        return x

class Agent():
    def __init__(self, n_action= len(env.action_space.high)):
        self.actor_main = Actor(n_action)
        self.actor_target = Actor(n_action)
        self.critic_main = Critic()
        self.critic_target = Critic()
        self.batch_size = 64
        self.n_actions = len(env.action_space.high)
        self.a_opt = tf.keras.optimizers.Adam(1e-4)
        # self.actor_target = tf.keras.optimizers.Adam(.001)
        self.c_opt = tf.keras.optimizers.Adam(1e-4)
        # self.critic_target = tf.keras.optimizers.Adam(.002)
        self.memory = RBuffer(1_00_000, env.observation_space.shape, len(env.action_space.high))
        self.trainstep = 0
        self.replace = 5
        self.gamma = 0.99
        self.min_action = env.action_space.low[0]
        self.max_action = env.action_space.high[0]
        self.tau = 0.005
        self.actor_target.compile(optimizer=self.a_opt)
        self.critic_target.compile(optimizer=self.c_opt)

    def act(self, state, evaluate=False):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        actions = self.actor_main(state)
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=0.1)

        actions = self.max_action * (tf.clip_by_value(actions, self.min_action, self.max_action))
        #print(actions)
        return actions[0]

    def update_target(self, tau=None):
        if tau is None:
            tau = self.tau

        weights1 = []
        targets1 = self.actor_target.weights
        for i, weight in enumerate(self.actor_main.weights):
            weights1.append(weight * tau + targets1[i]*(1-tau))
        self.actor_target.set_weights(weights1)

        weights2 = []
        targets2 = self.critic_target.weights
        for i, weight in enumerate(self.critic_main.weights):
            weights2.append(weight * tau + targets2[i]*(1-tau))
        self.critic_target.set_weights(weights2)


    def train(self):
        if self.memory.cnt < self.batch_size:
            return 

        states, next_states, rewards, actions, dones = self.memory.sample(self.batch_size)

        states = tf.convert_to_tensor(states, dtype= tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype= tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype= tf.float32)
        actions = tf.convert_to_tensor(actions, dtype= tf.float32)
        #dones = tf.convert_to_tensor(dones, dtype= tf.bool)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:

            target_actions = self.actor_target(next_states)
            target_next_state_values = tf.squeeze(self.critic_target(next_states, target_actions), 1)
            critic_value = tf.squeeze(self.critic_main(states, actions), 1)
            target_values = rewards + self.gamma * target_next_state_values * dones
            critic_loss = tf.keras.losses.MSE(target_values, critic_value)

            new_policy_actions = self.actor_main(states)
            actor_loss = -self.critic_main(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        grads1 = tape1.gradient(actor_loss, self.actor_main.trainable_variables)
        grads2 = tape2.gradient(critic_loss, self.critic_main.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor_main.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic_main.trainable_variables))

        #if self.trainstep % self.replace == 0:
        self.update_target()

        #self.trainstep +=1

with tf.device('GPU:0'):
    tf.random.set_seed(336699)
    agent = Agent(2)
    episods = 2000
    ep_reward = []
    total_avgr = []
    target = False

    for s in range(episods):
        if target == True:
            break
        total_reward = 0.
        state = env.reset()
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.savexp(state, next_state, action, done, reward)
            agent.train()
            state = next_state
            total_reward += reward
            if done:
                ep_reward.append(total_reward)
                avg_reward = np.mean(ep_reward[-100:])
                total_avgr.append(avg_reward)
                print("total reward after {} steps is {} and avg reward is {}".format(s, total_reward, avg_reward))
                if int(avg_reward) == 200:
                    target = True