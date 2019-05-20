from problem2 import *
import sys
import numpy as np

'''
    Unit test 2:
    This file includes unit tests for problem2.py.
    You could test the correctness of your code by typing `nosetests -v test2.py` in the terminal.
'''
#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 2 (15 points in total)--------------'''
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)


#-------------------------------------------------------------------------
def test_bandit_init():
    '''bandit_init (1 point)'''
    p = np.mat('.2,.4,.3;.8,.1,.6')
    b = CBandit(p)
    assert np.allclose(p,b.p) 
    assert np.allclose(b.n_s,2) 
    assert np.allclose(b.s,0) 

#-------------------------------------------------------------------------
def test_bandit_step():
    '''bandit_step (1 point)'''
    # 3-armed bandit with two possible states
    p = np.mat('.2,.4,.3;.8,.1,.6')
    b = CBandit(p)
    # counts of winning times for each arm 
    c = np.zeros((2,3))
    for s in range(2):
        for a in range(3):
            for _ in range(1000):
                r,_ = b.step(a)
                b.s=s # fixing the state
                if r == 1.:
                    c[s,a]+=1
                elif r == 0.:
                    pass
                else:
                    assert False # the returned rewards should be either 1. or 0.
            assert np.allclose(c[s,a]/1000.,p[s,a],atol=1e-1) # the winning probability should match with p[0,i]
 
    c = np.zeros(2)
    c2 = np.zeros(2)
    for _ in range(1000):
        _,s = b.step(0)
        c[s]+=1
        c2[b.s]+=1
    assert np.allclose(c/1000., [.5,.5],atol=1e-1)
    assert np.allclose(c2/1000., [.5,.5],atol=1e-1)

 
#-------------------------------------------------------------------------
def test_agent_init():
    '''agent_init (1 point)'''
    n = np.random.randint(2,100)
    n_s = np.random.randint(2,100)
    m = Agent(n,n_s) 
    assert type(m.Q) == np.ndarray 
    assert np.allclose(m.Q.shape,(n_s,n))
    assert np.allclose(m.Q, np.zeros((n_s,n))) 
    assert type(m.c) == np.ndarray 
    assert np.allclose(m.c.shape,(n_s,n))
    assert np.allclose(m.c, np.zeros((n_s,n))) 
    assert type(m.e) == float 
    assert np.allclose(m.e, 0.1) 
    assert type(m.n) == int 
    assert np.allclose(m.n, n) 

 
#-------------------------------------------------------------------------
def test_agent_forward():
    '''agent_forward (4 point)'''
    m = Agent(3,2,0.) 
    a = m.forward(0)
    for i in range(20):
        assert m.forward(0)==0
        assert m.forward(1)==0

    m.Q[0,0] += -.1
    for i in range(20):
        assert m.forward(0)==1
        assert m.forward(1)==0

    m.Q[1,0] += -.01
    for i in range(20):
        assert m.forward(0)==1
        assert m.forward(1)==1

    m.Q[1,1] += -.01
    for i in range(20):
        assert m.forward(0)==1
        assert m.forward(1)==2

    m.e =1.0
    # counts of the times that each arm has been pulled
    c = np.zeros((2,3))
    for i in range(1000):
        c[0,m.forward(0)] +=1
        c[1,m.forward(1)] +=1
    assert np.allclose(c/1000., np.ones((2,3))/3., atol = 1e-1)


    m.e =.6
    c = np.zeros((2,3))
    for i in range(1000):
        c[0,m.forward(0)] +=1
        c[1,m.forward(1)] +=1
    assert np.allclose(c/1000., [[.2,.6,.2],[.2,.2,.6]], atol = 1e-1)

#-------------------------------------------------------------------------
def test_update():
    '''agent_update (4 point)'''
    m = Agent(3,2) 
    m.update(0,1,1.)
    assert np.allclose(m.c, [[0,1,0],[0,0,0]])
    assert np.allclose(m.Q, [[0.,1.,0.],[0,0,0]])
    m.update(0,1,0.)
    assert np.allclose(m.c, [[0,2,0],[0,0,0]])
    assert np.allclose(m.Q, [[0.,.5,0.],[0,0,0]])
    m.update(0,1,1.)
    assert np.allclose(m.c, [[0,3,0],[0,0,0]])
    assert np.allclose(m.Q, [[0.,.6667,0.],[0,0,0]],atol = 1e-2)

    m.update(0,2,0.)
    assert np.allclose(m.c, [[0,3,1],[0,0,0]])
    assert np.allclose(m.Q, [[0.,.6667,0.],[0,0,0]],atol = 1e-2)

    m.update(0,2,1.)
    assert np.allclose(m.c, [[0,3,2],[0,0,0]])
    assert np.allclose(m.Q, [[0.,.6667,.5],[0,0,0]],atol = 1e-2)

    m = Agent(3,2) 
    m.update(1,1,1.)
    assert np.allclose(m.c, [[0,0,0],[0,1,0]])
    assert np.allclose(m.Q, [[0.,0.,0.],[0,1,0]])


#-------------------------------------------------------------------------
def test_play():
    '''agent_play (4 point)'''
    p = np.mat('.2,.6,.8;.4,.7,.1')
    g = CBandit(p)
    m = Agent(3,2,e=1.) 
    m.play(g)
    assert np.allclose(m.Q,p,atol=1e-1)

    m = Agent(3,2,e=0.) 
    m.play(g)
    assert np.argmax(m.Q[0])==0
    assert np.argmax(m.Q[1])==0

    m = Agent(3,2,e=0.1) 
    m.play(g)
    assert np.argmax(m.Q[0])==2
    assert np.argmax(m.Q[1])==1


    for _ in range(10):
        g = CBandit(p)
        m = Agent(3,2,e=0.) 
        m.play(g,1)
        assert m.c[0,0] == 1
            
             
