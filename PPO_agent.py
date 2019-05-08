import os
import gym
import gym_vgdl
import itertools
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import sys
import tensorflow as tf
import collections
import random
import cv2
from pong import Pong
from simple_ai import PongAi
#Some parts of the code are inspired by Unity mlagents

class Model():
    def __init__(self,summaries_dir=None):
        if summaries_dir:
            if not os.path.exists(summaries_dir):
                os.makedirs(summaries_dir)
            self.summary_writer = tf.summary.FileWriter(summaries_dir)

        self.max_steps=1e+6
        self.a_size=3
        self.action = tf.placeholder(shape=[None],dtype=tf.int32, name="action")
    
        encoded_state,encoded_next_state=self.StateProcessor()
        self.autoencoder(encoded_state,encoded_next_state)
        self.PolicyEstimator(encoded_state)
        self.ValueEstimator(encoded_state)
        #self.TargetValueEstimator(encoded_state)
        #self.InverseLossEstimator(encoded_state,encoded_next_state)
        self.define_loss()
        self.set_graph=False
        
    @staticmethod
    def create_visual_input(name):
        state = tf.placeholder(shape=[None, 200, 210, 3], dtype=tf.float32, name=name)
        return state
    
    def StateProcessor(self):
        with tf.variable_scope("state_processor"):
            self.state=self.create_visual_input("state")
            conv1 = tf.layers.conv2d(self.state, 16, kernel_size=[8, 8], strides=[4, 4],
                                     activation=tf.nn.elu, reuse=False, name="conv_1")
            conv2 = tf.layers.conv2d(conv1, 32, kernel_size=[4, 4], strides=[2, 2],
                                     activation=tf.nn.elu, reuse=False, name="conv_2")
            hidden = tf.contrib.layers.flatten(conv2)
            hidden = tf.layers.dense(hidden, 128, activation=tf.nn.relu, reuse=False, name="hidden_1",
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(1.0))
            
            self.next_state=self.create_visual_input("next_state")
            conv1_1 = tf.layers.conv2d(self.next_state, 16, kernel_size=[8, 8], strides=[4, 4],
                                     activation=tf.nn.elu, reuse=True, name="conv_1")
            conv2_1 = tf.layers.conv2d(conv1_1, 32, kernel_size=[4, 4], strides=[2, 2],
                                     activation=tf.nn.elu, reuse=True, name="conv_2")
            hidden_1 = tf.contrib.layers.flatten(conv2_1)
            hidden_1 = tf.layers.dense(hidden_1, 128, activation=tf.nn.relu, reuse=True, name="hidden_1",
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(1.0))
        return hidden,hidden_1

    def autoencoder(self,encoded_state,encoded_next_state):
        with tf.variable_scope("autoencoder"):
            hidden = tf.layers.dense(encoded_state, 17664, activation=tf.nn.relu, reuse=False, name="hidden_2",
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(1.0))
            hidden=tf.reshape(hidden,[-1,23, 24, 32])
            conv_transpose1=tf.layers.conv2d_transpose(hidden, 16, kernel_size=[5, 5], strides=[2, 2],
                                         activation=tf.nn.elu, reuse=False, name="conv_transpose_2") 
            self.conv_transpose2 = tf.layers.conv2d_transpose(conv_transpose1, 3, kernel_size=[8, 10], strides=[4, 4],
                                         activation=tf.nn.elu, reuse=False, name="conv_transpose_1")
            self.conv_transpose2=tf.contrib.layers.flatten(self.conv_transpose2)
            
            next_hidden = tf.layers.dense(encoded_next_state, 17664, activation=tf.nn.relu, reuse=True, name="hidden_2",
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(1.0))      
            next_hidden=tf.reshape(next_hidden,[-1,23, 24, 32])
            next_conv_transpose1=tf.layers.conv2d_transpose(next_hidden, 16, kernel_size=[5, 5], strides=[2, 2],
                                         activation=tf.nn.elu, reuse=True, name="conv_transpose_2")
            self.next_conv_transpose2 = tf.layers.conv2d_transpose(next_conv_transpose1, 3, kernel_size=[8, 10], strides=[4, 4],
                                         activation=tf.nn.elu, reuse=True, name="conv_transpose_1")
            self.next_conv_transpose2=tf.contrib.layers.flatten(self.next_conv_transpose2)
            
            self.autoencoder_loss=0.001*tf.add(tf.reduce_mean(tf.squared_difference(tf.contrib.layers.flatten(self.state),self.conv_transpose2)),
                                         tf.reduce_mean(tf.squared_difference(tf.contrib.layers.flatten(self.next_state), self.next_conv_transpose2)))
            self.autoencoder_summaries = tf.summary.merge([
                    tf.summary.scalar("autoencoder_loss", self.autoencoder_loss)
                ])

    def PolicyEstimator(self,encoded_state,learning_rate=0.0001):
        with tf.variable_scope("policy_estimator"):
            decay_epsilon = tf.train.polynomial_decay(0.2, tf.train.get_global_step(), self.max_steps, 0.1, power=1.0)
            decay_learning_rate= tf.train.polynomial_decay(learning_rate, tf.train.get_global_step(), self.max_steps, 1e-10, power=1.0)
            self.advantage = tf.placeholder(shape=[None],dtype=tf.float32, name="advantage")
            self.old_action_probs = tf.placeholder(shape=[None,self.a_size],dtype=tf.float32)
            self.action_probs = tf.layers.dense(encoded_state,self.a_size,activation=tf.nn.softmax,
                                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.01))

            self.picked_action_prob=tf.reduce_sum(self.action_probs*tf.one_hot(self.action,self.a_size),axis=1)
            self.picked_old_action_prob=tf.reduce_sum(self.old_action_probs*tf.one_hot(self.action,self.a_size),axis=1)
            self.entropy=-tf.reduce_sum(self.action_probs * tf.log(self.action_probs + 1e-10))
            self.mean_entropy = -tf.reduce_mean(self.action_probs * tf.log(self.action_probs + 1e-10))
            #Policy gradient
            #self.policy_loss =tf.reduce_mean( -tf.log(self.picked_action_prob+ 1e-10) * self.advantage)
            #Clipped Surrogate Objective
            ratio = self.picked_action_prob / (self.picked_old_action_prob  + 1e-10)
            a = ratio * self.advantage
            b = tf.clip_by_value(ratio, 1.0 - decay_epsilon, 1.0 + decay_epsilon) * self.advantage
            self.policy_loss=-tf.reduce_mean(tf.minimum(a,b))
            
            optimizer = tf.train.AdamOptimizer(learning_rate=decay_learning_rate)
            self.policy_train_op = optimizer.minimize(
                self.policy_loss, global_step=tf.train.get_global_step())
            # Summaries for Tensorboard
            self.policy_summaries = tf.summary.merge([
                tf.summary.scalar("policy_loss", self.policy_loss),
                tf.summary.scalar("mean_entropy", self.mean_entropy),
                tf.summary.histogram("entropy", self.entropy)
            ])
            
    def policy_predict(self,sess, state):
##        if not self.set_graph:
##            self.summary_writer.add_graph(sess.graph)
##            self.set_graph=True
        return sess.run(self.action_probs, { self.state: state })

    def policy_update(self,sess, state, advantage, action,old_action_probs):
        feed_dict = { self.state: state, self.advantage: advantage, self.action: action ,self.old_action_probs:old_action_probs }
        summaries,global_step,_, loss = sess.run([self.policy_summaries,tf.train.get_global_step(),self.policy_train_op, self.policy_loss], feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss
    
    def ValueEstimator(self,encoded_state,learning_rate=0.0001):
        with tf.variable_scope("value_estimator"):
            self.target = tf.placeholder(shape=[None],dtype=tf.float32, name="target")
            decay_learning_rate= tf.train.polynomial_decay(learning_rate, tf.train.get_global_step(), self.max_steps, 1e-10, power=1.0)
            
            self.value_estimate = tf.layers.dense(encoded_state,self.a_size,activation=None)
            value_selected_action=tf.reduce_sum(self.value_estimate*tf.one_hot(self.action,self.a_size),axis=1)
            self.value_loss = tf.reduce_mean(tf.squared_difference(value_selected_action, self.target))
            
            optimizer = tf.train.AdamOptimizer(learning_rate=decay_learning_rate)
            self.value_train_op = optimizer.minimize(
                self.value_loss, tf.train.get_global_step())
            # Summaries for Tensorboard
            self.value_summaries = tf.summary.merge([
                tf.summary.scalar("value_loss", self.value_loss)
            ])

    def TargetValueEstimator(self,encoded_state):
        with tf.variable_scope("target_value_estimator"):
            
            self.target_value_estimate = tf.layers.dense(encoded_state,self.a_size,activation=None)
            
    def target_value_predict(self,sess, state):
        return sess.run(self.target_value_estimate, { self.state: state })
    
    def value_predict(self,sess, state):
        return sess.run(self.value_estimate, { self.state: state })
    
    def value_update(self,sess, state, action,target):
        feed_dict = { self.state: state, self.action:action,self.target: target }
        summaries,global_step,_, loss = sess.run([self.value_summaries,tf.train.get_global_step(),self.value_train_op, self.value_loss], feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss
    
    def InverseLossEstimator(self,encoded_state,encoded_next_state,learning_rate=0.0001):
        with tf.variable_scope("inverse_loss_estimator"):
            decay_learning_rate= tf.train.polynomial_decay(learning_rate, tf.train.get_global_step(), self.max_steps, 1e-10, power=1.0)
            merge1 = tf.concat([encoded_state,encoded_next_state],axis=1)
            q_fc1=tf.layers.dense(merge1,units=256,activation=tf.nn.relu,name='q_fc1')
            p1=tf.layers.dense(q_fc1,units=self.a_size,activation=tf.nn.softmax,name='q_prob1')
            self.picked_action_prob1=tf.reduce_sum(p1*tf.one_hot(self.action,self.a_size),axis=1)
            self.q_losses=-tf.log(self.picked_action_prob1+1e-10)
            self.qloss=tf.reduce_mean(self.q_losses)
            optimizer = tf.train.AdamOptimizer(learning_rate=decay_learning_rate)
            self.q_train_op = optimizer.minimize(self.qloss, global_step=tf.train.get_global_step())

            # Summaries for Tensorboard
            self.inverse_loss_summaries = tf.summary.merge([
                tf.summary.scalar("q_loss", self.qloss),
                tf.summary.histogram("q_loss_hist", self.q_losses),
                tf.summary.histogram("p1", self.picked_action_prob1)
                ])
            
    def state_processor_update(self, sess, state, action, next_state):
      feed_dict = { self.state: state, self.next_state: next_state, self.action: action}
      summaries, global_step, _, loss = sess.run([self.inverse_loss_summaries, tf.train.get_global_step(), self.q_train_op, self.qloss],
          feed_dict)
      if self.summary_writer:
          self.summary_writer.add_summary(summaries, global_step)
      return loss
    
    def define_loss(self,learning_rate=1e-3):
        self.decay_learning_rate= tf.train.polynomial_decay(learning_rate, tf.train.get_global_step(), self.max_steps, 1e-10, power=1.0)
        self.loss=self.policy_loss+self.value_loss+self.autoencoder_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=self.decay_learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
        self.lr_summaries = tf.summary.merge([
            tf.summary.scalar("learning_rate", self.decay_learning_rate)
            ])
        
    def update_model(self,sess,state,next_state,action,old_action_probs,target,advantage):
        feed_dict = { self.state: state, self.next_state: next_state, self.action: action,self.old_action_probs:old_action_probs,self.target: target,self.advantage:advantage}
        summaries1,summaries2,summaries3,summaries4, global_step, _, loss = sess.run([self.autoencoder_summaries,self.lr_summaries,
                                                                                      self.policy_summaries,self.value_summaries,
                                                                                      tf.train.get_global_step(), self.train_op, self.loss],feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries1, global_step)
            self.summary_writer.add_summary(summaries2, global_step)
            self.summary_writer.add_summary(summaries3, global_step)
            self.summary_writer.add_summary(summaries4, global_step)
                 
class Agent(object):
    def __init__(self,sess,env,opponent, train_episodes, discount_factor=0.99,player_id=1,headless=True):
        if type(env) is not Pong:
            raise TypeError("I'm not a very smart AI. All I can play is Pong.")
        self.env = env
        self.opponent=opponent
        self.player_id = player_id
        self.name = "PPO_agent"
        self.sess=sess
        self.model= Model(summaries_dir="./experiments4/pong/summaries")
        #first build the model, then initialize variables
        self.sess.run(tf.global_variables_initializer())
        self.train_episodes=train_episodes
        self.update_target_estimator_every=5000
        self.max_replay_memory_size=10000
        self.batch_size=1024
        self.epochs=3
        self.discount_factor=discount_factor
        self.lambda_=0.95
        self.headless=headless
        self.checkpoint_dir = os.path.join("./experiments4/pong", "checkpoints")
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "model")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver = tf.train.Saver()
        self.load_model(self.checkpoint_dir)        
        
    def load_model(self,checkpoint_dir):
        self.latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if self.latest_checkpoint:
            print("Loading checkpoint")
            self.saver.restore(self.sess, self.latest_checkpoint)
            
    def get_name(self):
        return self.name
    
    def get_action(self, state=None):
        action_probs = self.model.policy_predict(self.sess,np.expand_dims(state,axis=0))[0]
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action,action_probs
    
    def get_action2(self, state=None):
        action_probs = self.model.policy_predict(self.sess,np.expand_dims(state,axis=0))[0]
        action = np.argmax(action_probs)
        return action,action_probs
    
    def reset(self):
        return
    
    def train(self):
        total_t = self.sess.run(tf.train.get_global_step())
        Transition = collections.namedtuple("Transition", ["state", "action","action_probs", "reward","td_target","td_error", "next_state", "done"])
        replay_memory = []     
        reward_history, timestep_history = [], []
        average_reward_history = []
        for episode_number in range(self.train_episodes):
            self.saver.save(self.sess, self.checkpoint_path)
            state,state2 = self.env.reset()
            reward_sum, timesteps = 0, 0
            done=False
            advantages=[]
            while not done:
                action1,action_probs=self.get_action(state)
                action2 = self.opponent.get_action()
                (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
                next_state=ob1
                reward=rew1
                if not done:
                    reward=reward+0.1
                done=done
                if not self.headless:
                    self.env.render()
                value_next = self.model.value_predict(self.sess,np.expand_dims(next_state,axis=0))[0]
                value=(self.model.value_predict(self.sess,np.expand_dims(state,axis=0))[0])[action1]
                td_target = reward + self.discount_factor * np.max(value_next)#discounted reward
                td_error=td_target-value
                advantages.append(td_error)
                
                replay_memory.append(Transition(state=state, action=action1,action_probs=action_probs, reward=reward,td_target=td_target,td_error=td_error,next_state=next_state, done=done))
                
                reward_sum=reward_sum+reward
                timesteps=timesteps+1
                
                state = next_state
                
            total_t += 1
            #gae: generalized advantage
            gae=[0]*len(advantages)
            for t in range(len(advantages)):
                count=0
                for tt in range(t,len(advantages)):
                    gae[t]=gae[t]+np.power(self.lambda_*self.discount_factor,count)*advantages[tt]
                    count=count+1
            replay_memory_corrected=[]
            c=0
            for i in range(len(replay_memory)-len(advantages),len(replay_memory)):
                replay_memory_corrected.append(Transition(state=replay_memory[i].state, action=replay_memory[i].action,action_probs=replay_memory[i].action_probs, reward=replay_memory[i].reward,
                                                         td_target=replay_memory[i].td_target,td_error=gae[c],next_state=replay_memory[i].next_state, done=replay_memory[i].done))
                c=c+1

            if len(replay_memory_corrected)==self.max_replay_memory_size:
                print('updating model')
                random.shuffle(replay_memory_corrected)
                for i in range(self.epochs):
                    start=0
                    while start+self.batch_size<len(replay_memory_corrected):
                        samples = random.sample(replay_memory_corrected, self.batch_size)

                        samples=replay_memory_corrected[start:start+self.batch_size]
                        start=start+self.batch_size
                        states_batch, action_batch,action_probs_batch, reward_batch,
                        td_target_batch ,td_error_batch , next_states_batch,
                        done_batch = map(np.array, zip(*samples))
                        td_target_batch=(td_target_batch-np.mean(td_target_batch))/np.std(td_target_batch)
                        
                        self.model.update_model(self.sess,states_batch,next_states_batch,
                                                action_batch,action_probs_batch,
                                                td_target_batch ,td_error_batch)
                replay_memory=[]
                replay_memory_corrected=[]
            print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(episode_number, reward_sum, timesteps))
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
            self.model.summary_writer.add_summary(summary,episode_number)
            self.model.summary_writer.flush()
        plt.plot(reward_history)
        plt.plot(average_reward_history)
        plt.legend(["Reward", "100-episode average"])
        plt.title("Reward history")
        plt.show()
        print("Training finished.")
        env.end()
        
    def test(self):
        for episode_number in range(self.train_episodes):
            print("episode: ",episode_number)
            state,state2 = self.env.reset()
            done=False
            while not done:
                action1,action_probs=self.get_action2(state)
                action2 = self.opponent.get_action()
                (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
                next_state=ob1
                reward=rew1
                done=done
                if not self.headless:
                    self.env.render()   
                state = next_state
        env.end()
        
env = Pong(headless=False)
episodes = 1

player_id = 1
opponent_id = 3 - player_id
opponent = PongAi(env, opponent_id)

env.set_names('PPO_agent', opponent.get_name())
tf.reset_default_graph()
global_step_tensor  = tf.Variable(0, name="global_step", trainable=False)

train=True
with tf.Session() as sess:
    sess.run(global_step_tensor.initializer)
    if train:
        agent=Agent(sess,env,opponent, train_episodes=1000000,headless=True)
        agent.train()
    else:
        agent=Agent(sess,env,opponent, train_episodes=100,headless=False)
        agent.test()
