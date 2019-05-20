from problem3 import *
import sys
import numpy as np
import torch as th
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader

'''
    Unit test 3:
    This file includes unit tests for problem3.py.
    You could test the correctness of your code by typing `nosetests -v test3.py` in the terminal.
'''
#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 3 (20 points in total)--------------'''
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)

#-------------------------------
def test_tanh():
    '''(4 point) tanh'''
    z = th.Tensor([[0.,1.,-1.,10.,-10]])
    a = tanh(z)
    assert type(a) == th.Tensor 
    a_true = th.tanh(z)
    assert np.allclose(a.data,a_true.data,atol=1e-2)

    z = th.Tensor([[0.,1.,-1.,1000.,-1000]])
    a = tanh(z)
    a_true = th.tanh(z)
    assert np.allclose(a.data,a_true.data,atol=1e-2)
    
    n = np.random.randint(20,100)
    h = np.random.randint(2,100)
    z = th.randn(n,h)
    a = tanh(z)
    a_true = th.tanh(z)
    assert np.allclose(a.data,a_true.data,atol=1e-2)

    z_true = th.tensor([[0.,1.,-1.,10.,-10.]],requires_grad=True)
    a_true = th.tanh(z_true)
    n = a_true.sum()
    n.backward()
    z = th.tensor([[0.,1.,-1.,10.,-10.]],requires_grad=True)
    a = tanh(z)
    m=a.sum()
    m.backward()
    assert np.allclose(z.grad.data,z_true.grad.data,atol=1e-2)

    z_true = th.tensor([[0.,1.,-1.,1000.,-1000.]],requires_grad=True)
    a_true = th.tanh(z_true)
    n = a_true.sum()
    n.backward()
    z = th.tensor([[0.,1.,-1.,1000.,-1000.]],requires_grad=True)
    a = tanh(z)
    m=a.sum()
    m.backward()
    assert np.allclose(z.grad.data,z_true.grad.data,atol=1e-2)



#-------------------------------
def test_init():
    '''(4 point) init'''
    p = np.random.randint(20,100)
    h = np.random.randint(20,100)
    c = np.random.randint(1,100)

    m = RNN(p,h,c)
    assert type(m) == RNN 
    assert isinstance(m, Module)
    assert type(m.U) == th.Tensor 
    assert type(m.V) == th.Tensor 
    assert type(m.b_h) == th.Tensor 
    assert type(m.W) == th.Tensor 
    assert type(m.b) == th.Tensor 
    assert type(m.loss_fn) == CrossEntropyLoss 
    assert m.U.requires_grad== True
    assert m.V.requires_grad == True
    assert m.b_h.requires_grad == True
    assert m.W.requires_grad== True
    assert m.b.requires_grad == True
    assert m.U.dtype == th.float
    assert m.V.dtype == th.float
    assert m.b_h.dtype == th.float
    assert m.W.dtype == th.float
    assert m.b.dtype == th.float
    assert np.allclose(m.U.data.size(), [p,h])
    assert np.allclose(m.V.data.size(), [h,h]) 
    assert np.allclose(m.b_h.data.size(), [h]) 
    assert np.allclose(m.W.data.size(), [h,c])
    assert np.allclose(m.b.data.size(), [c]) 
    assert np.allclose(m.U.data, np.zeros((p,h)))
    assert np.allclose(m.V.data, np.zeros((h,h))) 
    assert np.allclose(m.b_h.data, np.ones(h)) 
    assert np.allclose(m.W.data, np.zeros((h,c)))
    assert np.allclose(m.b.data, np.zeros(c)) 

 
#-------------------------------
def test_forward():
    '''(5 point) forward'''
    # batch_size 3, p = 2
    x = th.ones(3,2)
    m = RNN(p=2,h=2,c=2)
    H = th.ones(3,2)
    z,H_new = m(x,H)
    # check value 
    assert type(z) == th.Tensor 
    assert type(H_new) == th.Tensor 
    assert np.allclose(z.size(),[3,2])
    assert np.allclose(H_new.size(),[3,2])
    assert np.allclose(z.data, np.zeros((3,2)))
    assert np.allclose(H_new.data, .7616*np.ones((3,2)),atol=1e-2)

    m.b.data+=1.
    z,H_new = m(x,H)
    assert np.allclose(z.data, np.ones((3,2)))
    m.W.data+=1.
    z,H_new = m(x,H)
    assert np.allclose(z.data, 2.5232*np.ones((3,2)),atol=1e-2)
    m.V.data+=1.
    z,H_new = m(x,H)
    assert np.allclose(z.data, 2.99*np.ones((3,2)),atol=1e-2)
    m.U.data-=1.
    z,H_new = m(x,H)
    assert np.allclose(z.data, 2.5232*np.ones((3,2)),atol=1e-2)



#-------------------------------
def test_train():
    '''(7 point) train '''
    # n = 4, t=3, p = 2 
    X  = [
          [ # instance 0 
            [0.,0.], # time step 0 
            [0.,0.], # time step 1
            [0.,0.]  # time step 2
          ], 
          [ # instance 1
            [0.,0.], 
            [0.,0.], 
            [0.,1.]
          ],
          [ # instance 2
            [0.,0.], 
            [1.,0.], 
            [0.,0.]
          ],
          [ # instance 3
            [0.,1.], 
            [0.,0.], 
            [0.,0.]
          ] 
         ]
    Y = [
         [0,0,0], # instance 0
         [0,0,1], # instance 1
         [0,1,1],
         [1,1,1]
        ]
    class toy(Dataset):
        def __init__(self):
            self.X  = th.Tensor(X)            
            self.Y = th.LongTensor(Y)
        def __len__(self):
            return 4 
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    d = toy()
    loader = th.utils.data.DataLoader(d, batch_size = 1)
    m = RNN(p=2,h=1,c=2)
    m.train(loader,1,1.)
    assert np.allclose(m.b.data,[1.5,-1.5],atol =1e-2)
    assert np.allclose(m.W.data,[1.1424,-1.1424],atol =1e-2)

    loader = th.utils.data.DataLoader(d, shuffle=True, batch_size = 2)
    m = RNN(p=2,h=1,c=2)
    m.train(loader,1000,.1)
    assert m.V.data.numpy() > 1.

    x = th.Tensor([[0.,0.]])
    h = th.Tensor([[0.]])
    z,ht = m(x,h)
    z = z.data.numpy()
    assert z[0,0]>0.1
    assert z[0,1]<-0.1

    x = th.Tensor([[1.,0.]])
    z,ht = m(x,h)
    z = z.data.numpy()
    assert z[0,0]<-0.1
    assert z[0,1]>0.1

    x = th.Tensor([[0.,0.]])
    z,ht = m(x,ht)
    z = z.data.numpy()
    assert z[0,0]<-0.1
    assert z[0,1]>0.1

    x = th.Tensor([[0.,1.]])
    z,ht = m(x,h)
    z = z.data.numpy()
    assert z[0,0]<-0.1
    assert z[0,1]>0.1

    x = th.Tensor([[0.,0.]])
    z,ht = m(x,ht)
    z = z.data.numpy()
    assert z[0,0]<-0.1
    assert z[0,1]>0.1


