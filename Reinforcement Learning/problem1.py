import math
import numpy as np

#-------------------------------------------------------------------------
'''
    Problem 1: Multi-armed bandit problem 
    In this problem, you will implement an AI player for Multi-armed bandit problem epsilon-greedy method.
    The main goal of this problem is to get familiar with a simplified problem in reinforcement learning, and how to train the model parameters on the data from a game.
    You could test the correctness of your code by typing `nosetests test1.py` in the terminal.
'''

#--------------------------
def Terms_and_Conditions():
    ''' 
        By submiting this homework or changing this function, you agree with the following terms:
       (1) Not sharing your code/solution with any student before and after the homework due. For example, sending your code segment to another student, putting your solution online or lending your laptop (if your laptop contains your solution or your dropbox automatically sychronize your solution between your home computer and your laptop) to another student to work on this homework will violate this term.
       (2) Not using anyone's code in this homework, build your own solution. For example, using some code segments from another student or online resources due to any reason (like too busy recently) will violate this term. Changing other people's code as your solution (such as changing the variable names) will also violate this term.
       (3) When discussing with any other student about this homework, only discuss high-level ideas or using psudo-code. Don't discuss about the solution at the code level. For example, discussing with another student about the solution of a function (which needs 5 lines of code to solve), and then working on the solution "independently", however the code of the two solutions are exactly the same, or only with minor differences  (like changing variable names) will violate this term.
      All violations of (1),(2) or (3) will be handled in accordance with the WPI Academic Honesty Policy.  For more details, please visit: https://www.wpi.edu/about/policies/academic-integrity/dishonesty
    '''
    #########################################
    ## CHANGE CODE HERE
    Read_and_Agree = True
    #########################################
    return Read_and_Agree
 
#-------------------------------------------------------
class Bandit:
    '''Bandit is the Multi-armed bandit machine. Instead of one  slot machine lever, you have a number of them, say three. Each lever/arm corresponds to a probability of winning. However these odds/probabilities are hidden from the players. '''
    # ----------------------------------------------
    def __init__(self, p):
        ''' Initialize the game. 
            Inputs:
                p: the vector of winning probabilities, a numpy vector of length n. 
                    Here n is the number of arms of the bandit. 
            Outputs:
                self.p: the vector of winning probabilities, a numpy vector of length n. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        self.p = p
        #########################################

    # ----------------------------------------------
    def step(self, a):
        '''
           Given an action (the id of the arm being pulled), return the reward based upon the winning probability of the arm. 
            Input:
                a: the index of the lever being pulled by the agent. a is an integer scalar between 0 and n-1. 
                    n is the number of arms in the bandit.
            Output:
                r: the reward returned to the agent, a float scalar. The "win" return 1., if "lose", return 0. as the reward.
                   The winning probabilty of this step should be the same as that of the lever being pulled by the agent.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        p = self.p[a]
        r = np.random.choice(2, p=[1-p,p])
        #########################################
        return r


#-------------------------------------------------------
class Agent:
    '''The agent is trying to maximize the sum of rewards (payoff) in the game using epsilon-greedy method.
       The agent will 
                (1) with a small probability (epsilon or e), randomly pull a lever with uniform distribution on all levers (Exploration); 
                (2) with a big probability (1-e) to pull the arm with the largest expected reward (Exploitation). If there is a tie, pick the one with the smallest index.'''
    # ----------------------------------------------
    def __init__(self, n, e=0.1):
        ''' Initialize the agent. 
            Inputs:
                n: the number of arms of the bandit, an integer scalar. 
                e: (epsilon) the probability of the agent randomly pulling a lever with uniform probability. e is a float scalar between 0. and 1. 
            Outputs:
                self.n: the number of levers, an integer scalar. 
                self.e: the probability of the agent randomly pulling a lever with uniform probability. e is a float scalar between 0. and 1. 
                self.Q: the expected ratio of rewards for pulling each lever, a numpy vector of length n. We initialize the vector with all-zeros.
                self.c: the counts of the number of times that each lever being pulled. a numpy vector of length n, initialized a all-zeros.
                
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        self.n = n
        self.e = e
        self.Q = np.zeros(n)
        self.c = np.zeros(n)
        #########################################

   # ----------------------------------------------
    def forward(self):
        '''
            The policy function of the agent.
            The agent will 
                (1) with a small probability (epsilon or e), randomly pull a lever with uniform distribution on all levers (Exploration); 
                (2) with a big probability (1-e) to pull the arm with the largest expected reward (Exploitation). If there is a tie, pick the one with the smallest index.
            Output:
                a: the index of the lever to pull. a is an integer scalar between 0 and n-1. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        act = np.random.choice(2, p=[self.e,1-self.e])
        if act:
        	a = np.argmax(self.Q)
        else:
        	a = np.random.randint(len(self.Q))

        #########################################
        return a


    #-----------------------------------------------------------------
    def update(self, a,r):
        '''
            Update the parameters of the agent.
            (1) increase the count of lever
            (2) update the expected reward based upon the received reward r.
            Input:
                a: the index of the arm being pulled. a is an integer scalar between 0 and n-1. 
                r: the reward returned, a float scalar. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        self.c[a] += 1
        self.Q[a] = (r + self.Q[a] * (self.c[a]-1)) / self.c[a]

        #########################################


    #-----------------------------------------------------------------
    def play(self, g, n_steps=1000):
        '''
            Play the game for n_steps steps. In each step,
            (1) pull a lever and receive the reward from the game
            (2) update the parameters 
            Input:
                g: the game machine, a multi-armed bandit object. 
                n_steps: number of steps to play in the game, an integer scalar. 
            Note: please do NOT use g.p in the agent. The agent can only call the g.step() function.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        for i in range(n_steps):
        	a = self.forward()
        	r = g.step(a)
        	self.update(a,r)

        #########################################



