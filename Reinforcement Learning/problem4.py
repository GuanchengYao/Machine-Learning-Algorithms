import math
import numpy as np
import torch as th
from torch.optim import SGD
import gym
#-------------------------------------------------------------------------
'''
    Problem 4: Deep Q-Learning
    In this problem, you will implement an AI player for the frozen lake game, using a neural network.
    Instead of storing the Q values in a table, we approximate the Q values with the output of a neural network. The input (game state) is represented as the one-hot encoding. The neural network has one fully connected layer (without biases, without non-linear activation). The outputs of the network are the Q values for the input state. 
    We will use backpropagation to train the neural network.
    You could test the correctness of your code by typing `nosetests test4.py` in the terminal.
    
    ------------------------------
    Action code 
        0 : "LEFT",
        1 : "DOWN",
        2 : "RIGHT",
        3 : "UP"
'''

#-------------------------------------------------------
class Game:
    '''Game is the frozen lake game with one-hot encoding of states. '''
    def __init__(self):
        self.env = gym.make("FrozenLake-v0")
    def reset(self):
        s = self.env.reset()        
        s = th.Tensor(np.identity(16)[s])
        return s 
    def step(self,action):
        '''convert the state into one-hot encoding'''
        s, r, done, info = self.env.step(action) 
        s = th.Tensor(np.identity(16)[s])
        return s,r, done, info
    def render(self):
        self.env.render()


#-------------------------------------------------------
class QNet(object):
    '''The agent is trying to maximize the sum of rewards (payoff) in the game using Q-Learning neural network. 
       The agent will 
                (1) with a small probability (epsilon or e), randomly choose an action with a uniform distribution on all actions (Exploration); 
                (2) with a big probability (1-e) to choose the action with the largest expected reward (Exploitation). If there is a tie, pick the one with the smallest index.'''
    # ----------------------------------------------
    def __init__(self, n=4, d=16, e=0.1):
        ''' Initialize the agent. 
            Inputs:
                n: the number of actions in the game, an integer scalar. 
                d: the number of dimensions of the states of the game, an integer scalar. 
                e: (epsilon) the probability of the agent randomly choosing an action with uniform probability. e is a float scalar between 0. and 1. 
            Outputs:
                self.n: the number of actions, an integer scalar. 
                self.e: the probability of the agent randomly choosing an action with uniform probability. e is a float scalar between 0. and 1. 
                self.W: the weight matrix connecting the input (state) to the output Q values on each action, a torch matrix (tensor) of shape n by d. We initialize the matrix with all-zeros.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        self.n = n
        self.e = e
        self.W = th.zeros((n,d), requires_grad = True)
        #########################################


    # ----------------------------------------------
    def compute_Q(self, s):
        '''
          Given a state of the game, compute the Q values for all actions. 
          Inputs:
                s: the current state of the machine, a pytorch vector of length d. 
          Output:
                Q: the Q values of the state with different actions, a pytorch tensor of length n.  n is the number of actions in the game.
            
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        Q = th.mv(self.W,s)
        #########################################
        return Q




    # ----------------------------------------------
    def forward(self, s):
        '''
          The policy function of the agent. 
          Inputs:
                s: the current state of the machine, a pytorch vector of length n_s. 
          Output:
                a: the index of the lever to pull. a is an integer scalar between 0 and n-1. 
            
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        Q = self.compute_Q(s)
        act = np.random.choice(2,p = [self.e, 1-self.e])
        if act:
            a = np.argmax(Q.data).item()
            # q = list(Q.data.numpy())
            # a = q.index(max(q))
        else:
            a = np.random.randint(len(Q))

        #########################################
        return a

    #-----------------------------------------------------------------
    def compute_L(self,s,a,r,s_new, gamma=.95):
        '''
            Compute squared error of the Q function. (target_Q_value - current_Q)^2
            Input:
                s: the current state of the game, an integer scalar. 
                a: the index of the action being chosen. a is an integer scalar between 0 and n-1. 
                r: the reward returned by the game, a float scalar. 
                s_new: the next state of the game, an integer scalar. 
                gamma: the discount factor, a float scalar between 0 and 1.
            Output:
                L: the squared error of step, a float scalar. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        # target Q
        target_Q = self.compute_Q(s_new)
        # current Q
        Q = self.compute_Q(s) 
        # loss
        L = th.pow(r + gamma * th.max(target_Q.data) - Q[a], 2)
        # lose = th.nn.MSELoss()
        # L = lose(Q[a], r + gamma * max(target_Q.data))
        
        #########################################
        return L 

 
    #--------------------------
    def play(self, env, n_episodes, render =False,gamma=.95, lr=.1):
        '''
            Given a game environment of gym package, play multiple episodes of the game.
            An episode is over when the returned value for "done"= True.
            (1) at each step, pick an action and collect the reward and new state from the game.
            (2) update the parameters of the model using gradient descent
            Input:
                env: the envirement of the game of openai gym package 
                n_episodes: the number of episodes to play in the game. 
                render: whether or not to render the game (it's slower to render the game)
                gamma: the discount factor, a float scalar between 0 and 1.
                lr: learning rate, a float scalar, between 0 and 1.
            Outputs:
                total_rewards: the total number of rewards achieved in the game 
        '''
        optimizer = SGD([self.W], lr=lr)
        total_rewards = 0.
        # play multiple episodes
        for _ in range(n_episodes):
            s = env.reset() # initialize the episode 
            done = False
            # play the game until the episode is done
            while not done:
                if render:
                    env.render() # render the game
                #########################################
                ## INSERT YOUR CODE HERE

                # agent selects an action
                a = self.forward(s)
                # game return a reward and new state
                s_new, r, done, info = env.step(a)
                # agent update the parameters
                L = self.compute_L(s,a,r,s_new, gamma)
                # compute gradients
                L.backward()
                # update model parameters
                optimizer.step()
                # reset the gradients of W to zero
                optimizer.zero_grad()
                s = s_new
                #########################################
                total_rewards += r # assuming the reward of the step is r
        return total_rewards




