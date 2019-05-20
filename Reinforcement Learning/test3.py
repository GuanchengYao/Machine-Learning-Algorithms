from problem3 import *
import gym
import sys
import numpy as np

'''
    Unit test 3:
    This file includes unit tests for problem3.py.
    You could test the correctness of your code by typing `nosetests -v test3.py` in the terminal.
'''
#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 3 (15 points in total)--------------'''
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)

 
#-------------------------------------------------------------------------
def test_agent_init():
    '''agent_init (2 point)'''
    n = np.random.randint(2,100)
    n_s = np.random.randint(2,100)
    m = QLearner(n,n_s) 
    assert type(m.Q) == np.ndarray 
    assert np.allclose(m.Q.shape,(n_s,n))
    assert np.allclose(m.Q, np.zeros((n_s,n))) 
    assert type(m.e) == float 
    assert np.allclose(m.e, 0.1) 
    assert type(m.n) == int 
    assert np.allclose(m.n, n) 

#-------------------------------------------------------------------------
def test_update():
    '''agent_update (7 point)'''
    # 2 actions, 2 states
    m = QLearner(2,2) 
    m.update(s=0,a=0,r=1.,s_new=1,gamma = 1., lr=0.)
    assert np.allclose(m.Q, [[0.,0.],[0.,0.]])

    m.update(s=0,a=0,r=1.,s_new=1,gamma = 0., lr=1.)
    assert np.allclose(m.Q, [[1.,0.],[0.,0.]])

    m.update(s=1,a=0,r=0.,s_new=0,gamma = 1., lr=1.)
    assert np.allclose(m.Q, [[1.,0.],[1.,0.]])
    
    m.update(s=1,a=1,r=0.,s_new=0,gamma = .5, lr=1.)
    assert np.allclose(m.Q, [[1.,0.],[1.,.5]])
    
    m.update(s=1,a=1,r=1.,s_new=0,gamma = .5, lr=1.)
    assert np.allclose(m.Q, [[1.,0.],[1.,1.5]])
    
    m.update(s=1,a=1,r=0.,s_new=0,gamma = .5, lr=.5)
    assert np.allclose(m.Q, [[1.,0.],[1.,1.0]])

#-------------------------------------------------------------------------
# '''You could use the following code to play the game and get an idea of how to use gym package. '''
#def test_env():
#    env = gym.make("FrozenLake-v0")
#    done = False
#    s = env.reset() # initialize the episode 
#    print 's:',s
#    while not done:
#        env.render() # render the game
#        s, r, done,_ = env.step(2) # play one step in the game
#        print 's,r:',s,r
#    assert False
   

#-------------------------------------------------------------------------
def test_play():
    '''agent_play (6 point)'''
    env = gym.make("FrozenLake-v0")
    m = QLearner(e=1.) 
    r=m.play(env,1000)
    assert r > 5 
    assert np.allclose(m.Q[5],np.zeros(4))
    assert np.allclose(m.Q[7],np.zeros(4))
    assert np.allclose(m.Q[11],np.zeros(4))
    assert np.allclose(m.Q[12],np.zeros(4))
    assert ((m.Q[-2]-m.Q[0])>0).all()
    assert ((m.Q[-2]-m.Q[6])>0).all()
    m.e = 0.1
    r=m.play(env,1000)
    assert r >= 200
    assert np.allclose(m.Q[5],np.zeros(4))
    assert np.allclose(m.Q[7],np.zeros(4))
    assert np.allclose(m.Q[11],np.zeros(4))
    assert np.allclose(m.Q[12],np.zeros(4))
    assert ((m.Q[-2]-m.Q[0])>0).all()
    assert ((m.Q[-2]-m.Q[6])>0).all()



 
