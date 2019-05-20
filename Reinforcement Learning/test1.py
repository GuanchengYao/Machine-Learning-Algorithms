from problem1 import *
import sys
import numpy as np

'''
    Unit test 1:
    This file includes unit tests for problem1.py.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''
#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 1 (20 points in total)--------------'''
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)


#-------------------------------------------------------------------------
def test_bandit_init():
    '''bandit_init (1 point)'''
    p = [.2,.4,.8,.1,.5]
    b = Bandit(p)
    assert np.allclose(p,b.p) 

#-------------------------------------------------------------------------
def test_bandit_step():
    '''bandit_step (2 point)'''
    # 5-armed bandit
    p = [.2,.4,.8,.1,.5]
    b = Bandit(p)
    # counts of winning times for each arm 
    c = np.zeros(5)
    for a in range(5):
        for _ in range(1000):
            r = b.step(a)
            if r == 1.:
                c[a]+=1
            elif r == 0.:
                pass
            else:
                assert False # the returned rewards should be either 1. or 0.
        assert np.allclose(c[a]/1000.,p[a],atol=1e-1) # the winning probability should match with p[a]
 
#-------------------------------------------------------------------------
def test_agent_init():
    '''agent_init (2 point)'''
    n = np.random.randint(2,100)
    m = Agent(n) 
    assert type(m.Q) == np.ndarray 
    assert len(m.Q)==n
    assert np.allclose(m.Q, np.zeros(n)) 
    assert type(m.c) == np.ndarray 
    assert len(m.c)==n
    assert np.allclose(m.c, np.zeros(n)) 
    assert type(m.e) == float 
    assert np.allclose(m.e, 0.1) 
    assert type(m.n) == int 
    assert np.allclose(m.n, n) 

 
#-------------------------------------------------------------------------
def test_agent_forward():
    '''agent_forward (5 point)'''
    m = Agent(3,0.) 
    a = m.forward()
    for i in range(20):
        assert m.forward()==0

    m.Q[0] += -.1
    for i in range(20):
        assert m.forward()==1

    m.Q[1] += -.01
    for i in range(20):
        assert m.forward()==2

    m.e =1.0
    # counts of the times that each arm has been pulled
    c = np.zeros(3)
    for i in range(1000):
        c[m.forward()] +=1
    assert np.allclose(c/1000., np.ones(3)/3., atol = 1e-1)


    m.e =.6
    c = np.zeros(3)
    for i in range(1000):
        c[m.forward()] +=1
    assert np.allclose(c/1000., [.2,.2,.6], atol = 1e-1)

 
#-------------------------------------------------------------------------
def test_update():
    '''agent_update (5 point)'''
    m = Agent(3) 
    m.update(1,1.)
    assert np.allclose(m.c, [0,1,0])
    assert np.allclose(m.Q, [0.,1.,0.])
    m.update(1,0.)
    assert np.allclose(m.c, [0,2,0])
    assert np.allclose(m.Q, [0.,.5,0.])
    m.update(1,1.)
    assert np.allclose(m.c, [0,3,0])
    assert np.allclose(m.Q, [0.,.6667,0.],atol = 1e-2)

    m.update(2,0.)
    assert np.allclose(m.c, [0,3,1])
    assert np.allclose(m.Q, [0.,.6667,0.],atol = 1e-2)

    m.update(2,1.)
    assert np.allclose(m.c, [0,3,2])
    assert np.allclose(m.Q, [0.,.6667,.5],atol = 1e-2)


#-------------------------------------------------------------------------
def test_play():
    '''agent_play (5 point)'''
    p = [.2,.6,.8]
    g = Bandit(p)
    m = Agent(3,e=1.) 
    m.play(g)
    assert np.allclose(m.Q,p,atol=1e-1)

    m = Agent(3,e=0.) 
    m.play(g)
    assert np.argmax(m.Q)==0

    m = Agent(3,e=0.1) 
    m.play(g)
    assert np.argmax(m.Q)==2




