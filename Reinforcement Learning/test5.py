from problem5 import *
import gym
import sys
import numpy as np

'''
    Unit test 5:
    This file includes unit tests for problem5.py.
    You could test the correctness of your code by typing `nosetests -v test5.py` in the terminal.
'''
#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 5 (30 points in total)--------------'''
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)

#-------------------------------------------------------------------------
def test_compute_z():
    '''compute_z(2 point)'''
    # 2 actions, 3 dimensional state
    m = PolicyNet(2,3) 
    m.W.data[1,1] +=1.
    s = th.Tensor([0.,1.,0.])
    z = m.compute_z(s)
    assert type(z) == th.Tensor 
    assert np.allclose(z.data.size(),(2))
    assert np.allclose(z.data, [0,1]) 

    m.W.data[1,0] +=2.
    s = th.Tensor([1.,0.,0.])
    z = m.compute_z(s)
    assert np.allclose(z.data, [0,2]) 


#-------------------------------------------------------------------------
def test_compute_a():
    '''compute_a(2 point)'''
    # 2 actions, 3 dimensional state
    z = th.Tensor([0.,0.])
    a = PolicyNet.compute_a(z)
    assert type(a) == th.Tensor 
    assert np.allclose(a.data.size(),(2))
    assert np.allclose(a.data, [.5,.5],atol=1e-2) 

    z = th.Tensor([0.,0.,0.,0.])
    a = PolicyNet.compute_a(z)
    assert np.allclose(a.data, [.25,.25,.25,.25],atol=1e-2) 

    z = th.Tensor([-1000.,-1100.])
    a = PolicyNet.compute_a(z)
    assert np.allclose(a.data, [1.,0.],atol=1e-2) 

#-------------------------------------------------------------------------
def test_forward():
    '''forward (2 point)'''
    # 2 actions, 3 dimensional state
    m = PolicyNet(2,3) 
    m.W.data[1,1] +=1.
    s = th.Tensor([0.,1.,0.])
    a = m.forward(s)
    assert type(a) == th.Tensor 
    assert np.allclose(a.data.size(),(2))
    assert np.allclose(a.data, [.2689,.7311],atol=1e-2) 

    m.W.data[0,1] +=1.
    a = m.forward(s)
    assert np.allclose(a.data, [.5,.5], atol=1e-2) 

#-------------------------------------------------------------------------
def test_sample_action():
    ''' sample_action (5 point)'''
    a = th.Tensor([.3,.3,.4])
    m,logp = PolicyNet.sample_action(a)
    s = []
    for _ in range(1000):
        m,logp = PolicyNet.sample_action(a)
        assert type(m) == int 
        assert type(logp) == th.Tensor 
        assert np.allclose(logp.data.size(),[1])
        assert m in [0,1,2]
        if m==1:
            assert np.allclose(logp.item(),[-1.2040],atol=1e-2)
        elif m==1:
            assert np.allclose(logp.item(),[-1.2040],atol=1e-2)
        elif m==2:
            assert np.allclose(logp.item(),[-0.9163],atol=1e-2)
            
        s.append(m)
    s0 =  (np.array(s)==0).astype(np.float).sum()
    s1 =  (np.array(s)==1).astype(np.float).sum()
    s2 =  (np.array(s)==2).astype(np.float).sum()
    assert s0 < 350
    assert s0 > 250
    assert s1 < 350
    assert s1 > 250
    assert s2 < 450
    assert s2 > 350

    a = th.Tensor([.5,.5,.0])
    s = []
    for _ in range(1000):
        m,_ = PolicyNet.sample_action(a)
        assert m in [0,1,2]
        s.append(m)
    s0 =  (np.array(s)==0).astype(np.float).sum()
    s1 =  (np.array(s)==1).astype(np.float).sum()
    s2 =  (np.array(s)==2).astype(np.float).sum()
    assert s0 < 550
    assert s0 > 450
    assert s1 < 550
    assert s1 > 450
    assert s2==0


#-------------------------------------------------------------------------
def test_play_episode():
    ''' play_episode (5 point)'''
    env = Game() 
    m = PolicyNet() 
    S,M,logP,R = m.play_episode(env)
    n = len(S)
    assert type(M) == list
    assert len(M)==n
    assert type(logP) == list
    assert len(logP)==n
    assert type(R) == list
    assert len(R)==n
    assert np.allclose(m.W.data,np.zeros((4,16))) # the parameters of the network should not change.
    assert np.allclose(R[:-1],np.zeros(n-1))


#-------------------------------------------------------------------------
def test_discount_rewards():
    ''' discount_rewards (4 point)'''
    r = [1.,0.,1.,0.,0.,-1.]
    dr =PolicyNet.discount_rewards(r,1.)
    assert type(dr) == list
    assert np.allclose(dr, [1,0,0,-1,-1,-1])

    r = [.0,0.,0.,1.]
    dr = PolicyNet.discount_rewards(r,.5)
    assert np.allclose(dr, [.125,.25,.5,1])


#-------------------------------------------------------------------------
def test_compute_L():
    ''' compute_L (5 point)'''
    logP = [th.tensor([-.5],requires_grad=True),
            th.tensor([-.3],requires_grad=True),
            th.tensor([-.2],requires_grad=True)]
    dR = [1.,2.,3.]
    L = PolicyNet.compute_L(logP,dR)
    assert type(L) == th.Tensor 
    
    # check value
    assert np.allclose(L.data,[1.7],atol=1e-2)
    
    # check gradient
    L.backward()
    assert np.allclose(logP[0].grad.data,[-1],atol=1e-2)
    assert np.allclose(logP[1].grad.data,[-2],atol=1e-2)
    assert np.allclose(logP[2].grad.data,[-3],atol=1e-2)


#-------------------------------------------------------------------------
def test_play():
    '''agent_play (5 point)'''
    env = Game() 
    m = PolicyNet() 
    r=m.play(env,4000)
    assert np.allclose(m.W.grad.data,np.zeros((4,16)),atol=1e-2) # test whether the gradients have been cleared
    assert r > 200
    
