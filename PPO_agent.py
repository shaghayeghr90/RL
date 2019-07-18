import os
import gym
import itertools
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import sys
import tensorflow as tf
import collections
import random
import cv2
#Some parts of the code are inspired by Unity mlagents

class Model():
    def __init__(self,summaries_dir, obs_type, action_type):
        if summaries_dir:
            if not os.path.exists(summaries_dir):
                os.makedirs(summaries_dir)
            self.summary_writer = tf.summary.FileWriter(summaries_dir)

        self.max_steps = 5e+5
        self.a_size = env.action_space.n
        self.state_size = env.observation_space.low.shape[0]
        self.visual_observations = 4
        self.action = tf.placeholder(shape=[None],dtype=tf.int32, name="action")
        self.obs_type = obs_type
        self.action_type= action_type
        if obs_type == 'visual': 
            encoded_state, encoded_next_state=self.VisualStateProcessor()
        else:
            encoded_state, encoded_next_state=self.NonVisualStateProcessor()
        self.PolicyEstimator(encoded_state)
        self.ValueEstimator(encoded_state)
        if obs_type == 'visual':
            self.ForwardDynamicEstimator(encoded_state, encoded_next_state)
            self.InverseDynamicEstimator(encoded_state, encoded_next_state)
        self.define_loss()
        self.set_graph=False
        
    @staticmethod
    def create_visual_input(name):
        state = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name=name)
        return tf.to_float(state)/255.0
    
    def VisualStateProcessor(self):
        with tf.variable_scope("state_processor"):
            """hiddens = []
            next_hiddens = []
            self.state = self.create_visual_input("state")
            self.next_state = self.create_visual_input("next_state")
            for i in range(self.visual_observations):
                if i == 0:
                    reuse = False
                else:
                    reuse = True
                conv1 = tf.layers.conv2d(self.state[:, :, : , :, i], 16, kernel_size=[8, 8], strides=[4, 4],
                                     activation=tf.nn.elu, reuse=reuse, name="conv_1")
                conv2 = tf.layers.conv2d(conv1, 32, kernel_size=[4, 4], strides=[2, 2],
                                             activation=tf.nn.elu, reuse=reuse, name="conv_2")
                hidden = tf.layers.flatten(conv2)
                hidden = tf.layers.dense(hidden, 128, activation=tf.nn.relu, reuse=reuse, name="hidden_1",
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(1.0))
                hiddens.append(hidden)
                
                conv1 = tf.layers.conv2d(self.next_state[:, :, : , :, i], 16, kernel_size=[8, 8], strides=[4, 4],
                                     activation=tf.nn.elu, reuse=True, name="conv_1")
                conv2 = tf.layers.conv2d(conv1, 32, kernel_size=[4, 4], strides=[2, 2],
                                             activation=tf.nn.elu, reuse=True, name="conv_2")
                hidden = tf.layers.flatten(conv2)
                hidden = tf.layers.dense(hidden, 128, activation=tf.nn.relu, reuse=True, name="hidden_1",
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(1.0))
                next_hiddens.append(hidden)
                
            hidden = tf.concat(hiddens, axis=1)
            hidden_1 = tf.concat(next_hiddens, axis=1)"""
            
            self.state=self.create_visual_input("state")
            conv1 = tf.layers.conv2d(self.state, 16, kernel_size=[8, 8], strides=[4, 4],
                                     activation=tf.nn.elu, reuse=False, name="conv_1")
            conv2 = tf.layers.conv2d(conv1, 32, kernel_size=[4, 4], strides=[2, 2],
                                     activation=tf.nn.elu, reuse=False, name="conv_2")
            hidden = tf.layers.flatten(conv2)
            hidden = tf.layers.dense(hidden, 128, activation=tf.nn.relu, reuse=False, name="hidden_1",
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(1.0))
            
            self.next_state=self.create_visual_input("next_state")
            conv1_1 = tf.layers.conv2d(self.next_state, 16, kernel_size=[8, 8], strides=[4, 4],
                                     activation=tf.nn.elu, reuse=True, name="conv_1")
            conv2_1 = tf.layers.conv2d(conv1_1, 32, kernel_size=[4, 4], strides=[2, 2],
                                     activation=tf.nn.elu, reuse=True, name="conv_2")
            hidden_1 = tf.layers.flatten(conv2_1)
            hidden_1 = tf.layers.dense(hidden_1, 128, activation=tf.nn.relu, reuse=True, name="hidden_1",
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(1.0))
            
        return hidden,hidden_1
    
    @staticmethod
    def create_non_visual_input(name, state_size):
        state = tf.placeholder(shape=[None, state_size], dtype=tf.float32, name=name)
        return state
   
    def NonVisualStateProcessor(self):
        with tf.variable_scope("state_processor"):
            self.state = self.create_non_visual_input("state", self.state_size)
            self.next_state = self.create_non_visual_input("next_state", self.state_size)
        
            hidden = tf.layers.dense(self.state, 32, tf.nn.elu, tf.contrib.layers.xavier_initializer())
            hidden = tf.layers.dense(hidden, 32, tf.nn.elu, tf.contrib.layers.xavier_initializer())
            
            hidden_1 = tf.layers.dense(self.next_state, 32, tf.nn.elu, tf.contrib.layers.xavier_initializer())
            hidden_1 = tf.layers.dense(hidden_1, 32, tf.nn.elu, tf.contrib.layers.xavier_initializer())
            
        return hidden,hidden_1
    
    def PolicyEstimator(self,encoded_state):
        with tf.variable_scope("policy_estimator"):
            decay_epsilon = tf.train.polynomial_decay(0.2, tf.train.get_global_step(), self.max_steps, 0.1, power=1.0)
            self.advantage = tf.placeholder(shape=[None],dtype=tf.float32, name="advantage")
            self.old_action_probs = tf.placeholder(shape=[None, self.a_size],dtype=tf.float32)
            if self.action_type == 'continuous':
                mu = tf.layers.dense(encoded_state, self.a_size, None, tf.contrib.layers.xavier_initializer())
                sigma = tf.layers.dense(encoded_state, self.a_size, None, tf.contrib.layers.xavier_initializer())
                sigma = tf.nn.softplus(sigma) + 1e-5
                norm_dist = tf.contrib.distributions.Normal(mu, sigma)

                #action_tf_var can be backpropagated
                self.action_tf_var = tf.squeeze(norm_dist.sample(1), axis=0)
                self.action_tf_var = tf.clip_by_value(self.action_tf_var, env_action_space_low, env_action_space_high)
                self.action_prob = norm_dist.prob(self.action_tf_var)
                self.entropy = norm_dist.entropy()
                # policy gradient
                self.policy_loss= -tf.log(norm_dist.prob(self.action) + 1e-5) * self.advantage
                #Clipped Surrogate Objective
                # ratio = self.action_prob / (self.old_action_probs  + 1e-10)
                # a = ratio * self.advantage
                # b = tf.clip_by_value(ratio, 1.0 - decay_epsilon, 1.0 + decay_epsilon) * self.advantage
                # self.policy_loss=-tf.reduce_mean(tf.minimum(a,b))
            else:
                self.action_probs = tf.layers.dense(encoded_state, self.a_size, activation=tf.nn.softmax,
                                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.01))

                self.picked_action_prob=tf.reduce_sum(self.action_probs * tf.one_hot(self.action, self.a_size), axis=1)
                self.picked_old_action_prob=tf.reduce_sum(self.old_action_probs * tf.one_hot(self.action, self.a_size),axis=1)
                self.entropy=-tf.reduce_sum(self.action_probs * tf.log(self.action_probs + 1e-10))
                self.mean_entropy = -tf.reduce_mean(self.action_probs * tf.log(self.action_probs + 1e-10))
                #Policy gradient
                #self.policy_loss =tf.reduce_mean( -tf.log(self.picked_action_prob+ 1e-10) * self.advantage)
                #Clipped Surrogate Objective
                ratio = self.picked_action_prob / (self.picked_old_action_prob  + 1e-10)
                a = ratio * self.advantage
                b = tf.clip_by_value(ratio, 1.0 - decay_epsilon, 1.0 + decay_epsilon) * self.advantage
                self.policy_loss=-tf.reduce_mean(tf.minimum(a,b))

            # Summaries for Tensorboard
            self.policy_summaries = tf.summary.merge([
                tf.summary.scalar("policy_loss", self.policy_loss),
                tf.summary.scalar("entropy", self.mean_entropy)
                # tf.summary.histografm("entropy", self.entropy)
            ])
            
    def policy_predict(self,sess, state):
        if not self.set_graph:
            self.summary_writer.add_graph(sess.graph)
            self.set_graph=True
        if self.action_type == 'continuous':
            action, action_prob  = sess.run([self.action_tf_var, self.action_prob], feed_dict={self.state: state})
            return action, action_prob
        else:
            return sess.run(self.action_probs, { self.state: state })
    
    def ValueEstimator(self,encoded_state):
        with tf.variable_scope("value_estimator"):
            self.target = tf.placeholder(shape=[None],dtype=tf.float32, name="target")

            # self.value_estimate = tf.layers.dense(encoded_state,self.a_size,activation=None)
            # value_selected_action=tf.reduce_sum(self.value_estimate*tf.one_hot(self.action,self.a_size),axis=1)
            # self.value_loss = tf.reduce_mean(tf.squared_difference(self.value_selected_action, self.target))
            
            self.value_estimate = tf.layers.dense(encoded_state, 1, activation=None)
            self.value_loss = tf.reduce_mean(tf.squared_difference(self.value_estimate, self.target))

            # Summaries for Tensorboard
            self.value_summaries = tf.summary.merge([
                tf.summary.scalar("value_loss", self.value_loss)
            ])

    def value_predict(self,sess, state):
        return sess.run(self.value_estimate, { self.state: state })

    def InverseDynamicEstimator(self, encoded_state, encoded_next_state):
        with tf.variable_scope("inverse_dynamic_estimator"):
            merge1 = tf.concat([encoded_state, encoded_next_state], axis=1)
            q_fc1 = tf.layers.dense(merge1, units=256, activation=tf.nn.relu, name='q_fc1')
            p1 = tf.layers.dense(q_fc1, units=self.a_size, activation=tf.nn.softmax, name='q_prob1')
            self.picked_action_prob1 = tf.reduce_sum(p1 * tf.one_hot(self.action, self.a_size), axis=1)
            self.q_losses = -tf.reduce_sum(tf.log(p1 + 1e-10) * tf.one_hot(self.action, self.a_size), axis=1)
            self.q_loss=tf.reduce_mean(self.q_losses)

            # Summaries for Tensorboard
            self.inverse_dynamic_summaries = tf.summary.merge([
                tf.summary.scalar("q_loss", self.q_loss)
                # tf.summary.histogram("q_loss_hist", self.q_losses),
                # tf.summary.histogram("p1", self.picked_action_prob1)
                ])
            
    def ForwardDynamicEstimator(self,encoded_state, encoded_next_state):
        with tf.variable_scope("forward_dynamic_estimator"):
            merge1 = tf.concat([encoded_state, tf.one_hot(self.action,self.a_size)], axis=1)
            q_fc1 = tf.layers.dense(merge1, units=256, activation=tf.nn.relu, name='fw_fc1')
            predicted_encoded_next_state = tf.layers.dense(q_fc1, units=128, activation=tf.nn.relu, name='fw_fc2')
            self.fw_loss = tf.reduce_mean(tf.squared_difference(predicted_encoded_next_state, encoded_next_state))
            
            # Summaries for Tensorboard
            self.forward_dynamic_summaries = tf.summary.merge([
                tf.summary.scalar("fw_loss", self.fw_loss)
                ])
            
    def define_loss(self,learning_rate=1e-3):
        decay_learning_rate= tf.train.polynomial_decay(learning_rate, tf.train.get_global_step(), self.max_steps, 1e-10, power=1.0)
        decay_beta = tf.train.polynomial_decay(5e-4, tf.train.get_global_step(), self.max_steps, 1e-5, power=1.0)
        if obs_type == 'visual':
            self.loss = self.policy_loss + 0.5 * self.value_loss - decay_beta * tf.reduce_mean(self.entropy)+\
                2.0 * self.fw_loss + 0.8 * self.q_loss
        else:
            self.loss = self.policy_loss + 0.5 * self.value_loss - decay_beta * tf.reduce_mean(self.entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=decay_learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
        self.lr_summaries = tf.summary.merge([
            tf.summary.scalar("learning_rate", decay_learning_rate),
            tf.summary.scalar("beta", decay_beta)
            ])
        
    def update_model(self,sess,state,next_state,action,old_action_probs,target,advantage):
        feed_dict = { self.state: state, self.next_state: next_state, 
                     self.action: action, self.old_action_probs:old_action_probs, 
                     self.target: target, self.advantage:advantage}
        if obs_type == 'visual':
            summaries1, summaries2, summaries3, summaries4, summaries5, global_step, _, loss = \
            sess.run([self.lr_summaries, self.policy_summaries, 
                      self.value_summaries, self.inverse_dynamic_summaries, 
                      self.forward_dynamic_summaries, tf.train.get_global_step(), 
                      self.train_op, self.loss],feed_dict)
        else:
            summaries1, summaries2, summaries3, global_step, _, loss = sess.run([self.lr_summaries, 
                                                                                 self.policy_summaries, 
                                                                                 self.value_summaries,                                                                                                                      tf.train.get_global_step(), 
                                                                                 self.train_op, self.loss],feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries1, global_step)
            self.summary_writer.add_summary(summaries2, global_step)
            self.summary_writer.add_summary(summaries3, global_step)
            if self.obs_type == 'visual':
                self.summary_writer.add_summary(summaries4, global_step)
                self.summary_writer.add_summary(summaries5, global_step)
                 
class Agent(object):
    def __init__(self, sess, obs_type, action_type, env_name, total_episodes, discount_factor=0.99):
        self.env = gym.envs.make(env_name)
        self.render = False
        self.sess=sess
        self.model= Model(summaries_dir='./experiments_ppo_'+env_name+'/summaries', obs_type=obs_type, action_type=action_type)
        #first build the model, then initialize variables
        self.sess.run(tf.global_variables_initializer())
        self.total_episodes = total_episodes
        self.update_target_estimator_every=5000
        self.max_replay_memory_size = 10240
        self.batch_size = 1024
        self.epochs = 3
        self.discount_factor = discount_factor
        self.lambda_ = 0.95
        self.env_name = env_name
        self.obs_type = obs_type
        self.action_type = action_type
        self.checkpoint_dir = os.path.join('./experiments_ppo_'+env_name, 'checkpoints')
        self.checkpoint_path = os.path.join(self.checkpoint_dir, 'model')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver = tf.train.Saver()
        self.load_model(self.checkpoint_dir)        
        
    def load_model(self,checkpoint_dir):
        self.latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if self.latest_checkpoint:
            print("Loading checkpoint")
            self.saver.restore(self.sess, self.latest_checkpoint)
    
    def train(self):
        total_t = self.sess.run(tf.train.get_global_step())
        Transition = collections.namedtuple("Transition", ["state", "action","action_probs", "reward","td_target","td_error", "next_state", "done"])
        replay_memory = []
        targets = []
        advantages = []
        reward_history, timestep_history = [], []
        average_reward_history = []
        num_updates = 0
        for episode_number in range(self.total_episodes):
            states = []
            actions = []
            action_probs = []
            rewards = []
            next_states = []
            dones = []
            if episode_number % 1000 == 0:
                self.saver.save(self.sess, self.checkpoint_path)
            state = self.env.reset()
            if self.obs_type == 'visual':
                state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
                state = cv2.resize(state, (84, 84))
                state = np.stack([state] * 4, axis=2)
            else:
                state = normalize_state(state)
            #fire
            # state, _, _, _ = self.env.step(1)
            if self.render:
                self.env.render()
            reward_sum, timesteps = 0, 0
            ii = 0
            while ii < 1000:
                if self.action_type == 'continuous':
                    action, action_prob = self.model.policy_predict(self.sess, np.expand_dims(state, axis=0))
                else:
                    action_prob = self.model.policy_predict(self.sess, np.expand_dims(state, axis=0))[0]
                    action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
                next_state, reward, done, info = self.env.step(action)
                if self.obs_type == 'visual':
                    next_state = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
                    next_state = cv2.resize(next_state, (84, 84))
                    next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
                else:
                    next_state = normalize_state(next_state)
                if self.render:
                    self.env.render()

                states.append(state)
                actions.append(action)
                action_probs.append(action_prob)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                
                reward_sum = reward_sum + reward
                timesteps = timesteps + 1
                if done:
                    break
                state = next_state
                ii = ii + 1
            # when ,episode finiches, we should process the collected experiences and calculate advantage
            value_states =  self.model.value_predict(self.sess, np.array(states))
            rewards = np.array(rewards)
            dones = np.array(dones)
            if dones[-1] == True:
                value_next = [[0.0]]
            else: 
                value_next = self.model.value_predict(self.sess, np.expand_dims(next_states[-1], axis=0))
                print('value_next: ', value_next.shape)
            value_next = np.array(value_next)
            value_states = np.concatenate((value_states, value_next), axis=0).ravel()
           
            discounted_rewards = [0] * len(rewards)
            running_reward = value_next[0][0]
            for t in reversed(range(0, len(rewards))):
                running_reward = rewards[t] +  self.discount_factor * running_reward
                discounted_rewards[t] =  running_reward
            deltas =  rewards + self.discount_factor * value_states[1:] - value_states[:-1]
            gae = [0] * len(deltas)
            running_delta = 0
            for t in reversed(range(0, len(deltas))):
                running_delta = deltas[t] + (self.lambda_ * self.discount_factor) * running_delta
                gae[t] =  running_delta

            targets = targets + discounted_rewards
            advantages = advantages + gae
            for i in range(len(states)): 
                replay_memory.append(Transition(state=states[i], action=actions[i], 
                                                action_probs=action_probs[i], reward=rewards[i],
                                                td_target=discounted_rewards[i], td_error=gae[i], 
                                                next_state=next_states[i], done=dones[i]))
            total_t += 1
            
            if len(replay_memory) >= self.max_replay_memory_size:
                print('updating model', num_updates + 1)
                mean_td_target_batch = np.mean(np.array(targets))
                std_td_target_batch = np.std(np.array(targets))
                mean_td_error_batch = np.mean(np.array(advantages))
                std_td_error_batch = np.std(np.array(advantages))
                targets = []
                advantages = []
                for i in range(self.epochs):
                    random.shuffle(replay_memory)
                    start=0
                    while start + self.batch_size < len(replay_memory):
                        samples = replay_memory[start: start + self.batch_size]
                        start = start + self.batch_size
                        states_batch, action_batch, action_probs_batch, reward_batch, td_target_batch , td_error_batch , next_states_batch, done_batch = map(np.array, zip(*samples))
                        td_target_batch = (td_target_batch - mean_td_target_batch) / (std_td_target_batch + 1e-12)
                        td_error_batch = (td_error_batch-mean_td_error_batch) / (std_td_error_batch + 1e-12)
                        self.model.update_model(self.sess, states_batch, next_states_batch,
                                                action_batch, action_probs_batch,
                                                td_target_batch, td_error_batch)
                replay_memory=[]
                if num_updates % 100 == 0:
                    self.test(num_updates)
                num_updates = num_updates + 1
            # print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  # .format(episode_number, reward_sum, timesteps))
            summary = tf.Summary()
            reward_history.append(reward_sum)
            timestep_history.append(timesteps)
            if episode_number > 100:
                avg = np.mean(reward_history[-100:])
                avg_timesteps = np.mean(timestep_history[-100:])
            else:
                avg = np.mean(reward_history)
                avg_timesteps = np.mean(timestep_history)
            average_reward_history.append(avg)
            summary.value.add(tag="episode_reward", simple_value=reward_sum),
            summary.value.add(tag="episode_length", simple_value=timesteps),
            summary.value.add(tag="100-episode average of rewards", simple_value=avg)
            summary.value.add(tag="100-episode average of timesteps", simple_value=avg_timesteps)
            self.model.summary_writer.add_summary(summary, total_t)
            self.model.summary_writer.flush()
        plt.plot(reward_history)
        plt.plot(average_reward_history)
        plt.legend(["Reward", "100-episode average"])
        plt.title("Reward history")
        plt.show()
        print("Training finished.")
        self.env.close()
        
    def test(self, id_):
        # import time
        env_test = gym.envs.make(self.env_name)
        env_test = gym.wrappers.Monitor(env_test, './experiments_ppo_'+env_name+'/recording'+str(id_)+'/')
        state = env_test.reset()
        if self.obs_type == 'visual':
            state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            state = cv2.resize(state, (84, 84))
            state = np.stack([state] * 4, axis=2)
        else:
            state = normalize_state(state)
        #fire
        # state, _, _, _ = self.env.step(1)
        self.env.render()
        done=False
        while not done:
            if self.action_type == 'continuous':
                    action, action_prob = self.model.policy_predict(self.sess, np.expand_dims(state, axis=0))
                else:
                    action_prob = self.model.policy_predict(self.sess, np.expand_dims(state, axis=0))[0]
                    action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
            next_state, reward, done, info = env_test.step(action)
            if self.obs_type == 'visual':
                next_state = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
                next_state = cv2.resize(next_state, (84, 84))
                next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
            else:
                next_state = normalize_state(next_state)
            # time.sleep(1)
            state = next_state
        env_test.close()

env_name = 'Breakout-v0'  # 'CartPole-v0'
obs_type = 'visual'  # 'non_visual'
env = gym.envs.make(env_name)
state_space_samples = np.array([env.observation_space.sample() for x in range(10000)])
# mean_states=np.mean(state_space_samples, axis=0)
# std_states=np.std(state_space_samples, axis=0)
# print(mean_states,std_states)
def normalize_state(state):
    # return (state-mean_states)/(std_states + 1e-12)
    return state
VALID_ACTIONS = np.arange(env.action_space.n).tolist()
tf.reset_default_graph()
global_step_tensor  = tf.Variable(0, name="global_step", trainable=False)

train=True
with tf.Session() as sess:
    sess.run(global_step_tensor.initializer)
    if train:
        agent=Agent(sess, obs_type=obs_type, env_name=env_name, total_episodes=500000)
        agent.train()
    else:
        agent=Agent(sess, obs_type=obs_type, env_name=env_name, total_episodes=5)
        agent.test(0)
