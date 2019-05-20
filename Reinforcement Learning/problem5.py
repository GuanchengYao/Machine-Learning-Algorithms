import math
import numpy as np
import torch as th
from torch.optim import Adam 
import gym
from problem4 import Game, QNet
from torch.distributions import Categorical

#-------------------------------------------------------------------------
'''
    Problem 5: Policy-gradient Method for Deep Reinforcement Learning 
    In this problem, you will implement an AI player for the frozen lake game, using a neural network.
    Instead of using the neural network to approximate the Q function, we use the neural network to directly output the action. 
    The input (game state) is represented as the one-hot encoding. 
    The neural network has one fully connected layer (without biases) with softmax activation. 
    The outputs of the network are the probabilities of taking each action for the input state. 
    We will use backpropagation to train the neural network.
    You could test the correctness of your code by typing `nosetests test5.py` in the terminal.
    ------------------------------
    Action code 
        0 : "LEFT",
        1 : "DOWN",
        2 : "RIGHT",
        3 : "UP"
'''

#-------------------------------------------------------
class PolicyNet(QNet):
    ''' 
        The agent is trying to maximize the sum of rewards (payoff) in the game using Policy-Gradient Method. 
        PolicyNet is a subclass of the agent in problem 4.
        We will use the weight matrix W as the policy network.
        The agent will use the output probabilities (after softmax) to randomly sample actions. 
    '''
    # ----------------------------------------------
    def __init__(self, n=4, d=16):
        ''' Initialize the agent. 
            Inputs:
                n: the number of actions in the game, an integer scalar. 
                d: the number of dimensions of the states of the game, an integer scalar. 
        '''
        super(PolicyNet, self).__init__(n,d,0.)

    # ----------------------------------------------
    def compute_z(self, s):
        '''
          Given a state of the game, compute the linear logits of neural netowrk for all actions. 
          Inputs:
                s: the current state of the machine, a pytorch vector of length d. 
          Output:
                z: the linear logits of the network, a pytorch tensor of length n.  n is the number of actions in the game.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        z = th.mv(self.W,s)
        #########################################
        return z


    #-----------------------------------------------------------------
    @staticmethod
    def compute_a(z):
        '''
            Compute probabilities of the agent taking each action. 
            Input:
                z: the linear logit of the neural network, a float variable of length n.
                    Here n is the number of actions. 
            Output:
                a: the probability of the agent taking different actions, a float variable of length n. 
            Hint: you could solve this problem using one line of code. You could use any function provided by pytorch.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        a = th.nn.functional.softmax(z, dim = 0)

        #########################################
        return a

    # ----------------------------------------------
    def forward(self, s):
        '''
          The policy function of the agent. 
          Inputs:
                s: the current state of the machine, a pytorch vector of length n_s. 
          Output:
                a: the probability of the agent taking different actions, a float variable of length n. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        z = self.compute_z(s)
        a = PolicyNet.compute_a(z)

        #########################################
        return a


    #--------------------------
    @staticmethod
    def sample_action(a):
        '''
            Given a vector of activations (probabilities of taking each action), randomly sample an action according to the probabilities. 
            Input:
                a: the probabilities of different actions, a pytorch variable of length n.
            Output:
                m: the sampled action (move), an integer scalar of value, 0, 1, ..., n-1 
                logp: the log probability of the sampled action, a float tensor of value between 0 and 1.
            Hint: you could use the th.distributions.Categorical class.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        c = Categorical(a)
        m = c.sample().item()
        logp = th.log(a[m])

        #########################################
        return m, logp 


    #--------------------------
    def play_episode(self, env, render =False):
        '''
            Play one episode of the game and collect data of actions and rewards, while fixing the model parameters.
            This process is also called roll-out or trial.
            Note: please don't update the parameter of the agent in the episode.
            At each step, sample an action randomly from the output of the network and collect the reward and new state from the game.
            An episode is over when the returned value for "done"= True.
            Input:
                env: the envirement of the game 
                render: whether or not to render the game (it's slower to render the game)
            Output:
                S: the game states, a python list of game states in each step. S[i] is the game state at the i-th step.
                M: the sampled actions in the game, a python list of sampled actions.
                    M[i] is the sampled action at the i-th step.
                logP: the log probability of sampled action at each step, a python list.
                    logP[i] is the log probability tensor in the i-th step.
                R: the raw rewards in the game, a python list of collected rewards.
                    R[i] is the collected reward at the i-th step.
        '''
        S,M,logP,R = [],[],[],[]
        s = env.reset() # initial state of the game 
        done = False
        # play the game until the episode is done
        while not done:
            if render:
                env.render() # render the game
            #########################################
            ## INSERT YOUR CODE HERE

            # compute the probability of taking each action
            a = self.forward(s)
            # sample an action based upon the probabilities
            m,logp = PolicyNet.sample_action(a)
            # play one step in the game
            s_new, r, done, info = env.step(m)
            S.append(s)
            M.append(m)
            logP.append(logp)
            R.append(r)
            s = s_new
            #########################################
        return S,M,logP,R


    #--------------------------
    @staticmethod
    def discount_rewards(R,gamma=0.98):
        '''
            Given a time sequence of raw rewards in a game episode, compute discounted rewards (non-sparse) 
            Input:
                R: the raw rewards collected at each step of the game, a float python list of length h.
                gamma: discount factor, a float scalar between 0 and 1.
            Output:
                dR: discounted future rewards for each step, a float numpy array of length h.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        dR= [sum([R[i:][j] * np.power(gamma,j) for j in range(len(R)-i)]) for i in range(len(R))]

        #########################################
        return dR 
 
    #-----------------------------------------------------------------
    @staticmethod
    def compute_L(logP,dR):
        '''
            Compute policy loss of a game episode: the sum of (- log_probability * discounted_reward) at each step
            Input:
                logP: the log probability of sampled action at each step, a python list of length n.
                    Here n is the number of steps in the game.
                    logP[i] is the log probability tensor of the i-th step in the game.
                dR: discounted future rewards for each step, a float python list of length n.
            Output:
                L: the squared error of step, a float scalar tensor. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        L = sum(-logP[i] * dR[i] for i in range(len(dR)))

        #########################################
        return L 

 
    #--------------------------
    def play(self, env, n_episodes, render =False,gamma=.95, lr=.1):
        '''
            Given a game environment of gym package, play multiple episodes of the game.
            An episode is over when the returned value for "done"= True.
            At each step, pick an action and collect the reward and new state from the game.
            After an episode is done, compute the discounted reward and update the parameters of the model using gradient descent.
            Input:
                env: the envirement of the game of openai gym package 
                n_episodes: the number of episodes to play in the game. 
                render: whether or not to render the game (it's slower to render the game)
                gamma: the discount factor, a float scalar between 0 and 1.
                lr: learning rate, a float scalar, between 0 and 1.
            Outputs:
                total_rewards: the total number of rewards achieved in the game 
        '''
        optimizer = Adam([self.W], lr=lr)
        total_rewards = 0.
        # play multiple episodes
        for _ in range(n_episodes):
            #########################################
            ## INSERT YOUR CODE HERE
            S,M,logP,R = self.play_episode(env, render)
            dR = PolicyNet.discount_rewards(R,gamma)
            L = PolicyNet.compute_L(logP,dR)

            # compute gradients
            L.backward()
            # update model parameters
            optimizer.step()
            # reset the gradients of W to zero
            optimizer.zero_grad()
            #########################################
            total_rewards += sum(R) # assuming the list of rewards of the episode is R
        return total_rewards

