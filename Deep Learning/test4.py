from problem4 import *
import sys
import numpy as np
import torch as th
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader

'''
    Unit test 4:
    This file includes unit tests for problem4.py.
    You could test the correctness of your code by typing `nosetests -v test4.py` in the terminal.
'''
#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 4 (30 points in total)--------------'''
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)

#-------------------------------
def test_init():
    '''(5 point) init'''
    p = np.random.randint(20,100)
    h = np.random.randint(20,100)
    c = np.random.randint(1,100)

    m = LSTM(p,h,c)
    assert isinstance(m, Module)
    assert type(m.W_i) == th.Tensor 
    assert type(m.b_i) == th.Tensor 
    assert type(m.W_o) == th.Tensor 
    assert type(m.b_o) == th.Tensor 
    assert type(m.W_c) == th.Tensor 
    assert type(m.b_c) == th.Tensor 
    assert type(m.W_f) == th.Tensor 
    assert type(m.b_f) == th.Tensor 
    assert type(m.W) == th.Tensor 
    assert type(m.b) == th.Tensor 
    assert type(m.loss_fn) == CrossEntropyLoss 
    assert m.W_i.requires_grad== True
    assert m.b_i.requires_grad == True
    assert m.W_o.requires_grad== True
    assert m.b_o.requires_grad == True
    assert m.W_c.requires_grad == True
    assert m.b_c.requires_grad == True
    assert m.W_o.requires_grad == True
    assert m.b_o.requires_grad == True
    assert m.W.requires_grad== True
    assert m.b.requires_grad == True
    assert m.W_i.dtype == th.float
    assert m.W_o.dtype == th.float
    assert m.W_c.dtype == th.float
    assert m.W_f.dtype == th.float
    assert m.b_i.dtype == th.float
    assert m.b_o.dtype == th.float
    assert m.b_c.dtype == th.float
    assert m.b_f.dtype == th.float
    assert m.W.dtype == th.float
    assert m.b.dtype == th.float
    assert np.allclose(m.W_i.data.size(), [p+h,h])
    assert np.allclose(m.b_i.data.size(), [h])
    assert np.allclose(m.W_o.data.size(), [p+h,h])
    assert np.allclose(m.b_o.data.size(), [h])
    assert np.allclose(m.W_c.data.size(), [p+h,h])
    assert np.allclose(m.b_c.data.size(), [h])
    assert np.allclose(m.W_f.data.size(), [p+h,h])
    assert np.allclose(m.b_f.data.size(), [h])
    assert np.allclose(m.W.data.size(), [h,c])
    assert np.allclose(m.b.data.size(), [c]) 
    assert np.allclose(m.W_i.data, np.zeros((h+p,h)))
    assert np.allclose(m.b_i.data, np.zeros(h))
    assert np.allclose(m.W_o.data, np.zeros((h+p,h))) 
    assert np.allclose(m.b_o.data, np.zeros(h)) 
    assert np.allclose(m.W_c.data, np.zeros((h+p,h))) 
    assert np.allclose(m.b_c.data, np.zeros(h)) 
    assert np.allclose(m.W_f.data, np.zeros((h+p,h))) 
    assert np.allclose(m.b_f.data, np.zeros(h)) 
    assert np.allclose(m.W.data, np.zeros((h,c)))
    assert np.allclose(m.b.data, np.zeros(c)) 


#-------------------------------
def test_gates():
    '''(5 point) gates'''
    # n = 2, p = 3 
    x  = th.Tensor([
          # instance 0
          [ 0.,0.,0. ],
          # instance 1
          [ 0.,0.,0. ] 
         ])
    H  = th.Tensor([
          # instance 0
          [ 0.,0. ],
          # instance 1
          [ 0.,0. ] 
         ])
    m = LSTM(p=3,h=2,c=2)
    f,i,o,C_c = m.gates(x,H)
    m.W_f.data+=1.
    m.W_i.data+=1.
    m.W_o.data+=1.
    m.W_c.data+=1.
    assert type(f) == th.Tensor 
    assert type(i) == th.Tensor 
    assert type(o) == th.Tensor 
    assert type(C_c) == th.Tensor 
    assert np.allclose(f.data.size(), [2,2])
    assert np.allclose(i.data.size(), [2,2])
    assert np.allclose(o.data.size(), [2,2])
    assert np.allclose(C_c.data.size(), [2,2])
    assert np.allclose(f.data, .5*np.ones((2,2)),atol=1e-2) 
    assert np.allclose(i.data, .5*np.ones((2,2)),atol=1e-2) 
    assert np.allclose(o.data, .5*np.ones((2,2)),atol=1e-2) 
    assert np.allclose(C_c.data, np.zeros((2,2)),atol=1e-2) 
   
    H.data+= 1. 
    f,i,o,C_c = m.gates(x,H)
    assert np.allclose(f.data, .8808*np.ones((2,2)),atol=1e-2) 
    assert np.allclose(i.data, .8808*np.ones((2,2)),atol=1e-2) 
    assert np.allclose(o.data, .8808*np.ones((2,2)),atol=1e-2) 
    assert np.allclose(C_c.data, .9640*np.ones((2,2)),atol=1e-2) 

    x.data+= -1. 
    f,i,o,C_c = m.gates(x,H)
    assert np.allclose(f.data, .2689*np.ones((2,2)),atol=1e-2) 
    assert np.allclose(i.data, .2689*np.ones((2,2)),atol=1e-2) 
    assert np.allclose(o.data, .2689*np.ones((2,2)),atol=1e-2) 
    assert np.allclose(C_c.data, -.7616*np.ones((2,2)),atol=1e-2) 

    m.b_f.data+=1.
    f,i,o,C_c = m.gates(x,H)
    assert np.allclose(f.data, .5*np.ones((2,2)),atol=1e-2) 
    assert np.allclose(i.data, .2689*np.ones((2,2)),atol=1e-2) 
    assert np.allclose(o.data, .2689*np.ones((2,2)),atol=1e-2) 
    assert np.allclose(C_c.data, -.7616*np.ones((2,2)),atol=1e-2) 

    m.b_i.data+=1.
    f,i,o,C_c = m.gates(x,H)
    assert np.allclose(f.data, .5*np.ones((2,2)),atol=1e-2) 
    assert np.allclose(i.data, .5*np.ones((2,2)),atol=1e-2) 
    assert np.allclose(o.data, .2689*np.ones((2,2)),atol=1e-2) 
    assert np.allclose(C_c.data, -.7616*np.ones((2,2)),atol=1e-2) 

    m.b_o.data+=1.
    f,i,o,C_c = m.gates(x,H)
    assert np.allclose(f.data, .5*np.ones((2,2)),atol=1e-2) 
    assert np.allclose(i.data, .5*np.ones((2,2)),atol=1e-2) 
    assert np.allclose(o.data, .5*np.ones((2,2)),atol=1e-2) 
    assert np.allclose(C_c.data, -.7616*np.ones((2,2)),atol=1e-2) 

    m.b_c.data+=1.
    f,i,o,C_c = m.gates(x,H)
    assert np.allclose(f.data, .5*np.ones((2,2)),atol=1e-2) 
    assert np.allclose(i.data, .5*np.ones((2,2)),atol=1e-2) 
    assert np.allclose(o.data, .5*np.ones((2,2)),atol=1e-2) 
    assert np.allclose(C_c.data,np.zeros((2,2)),atol=1e-2) 


    m.W_f.data[0,1]+=1.
    m.W_f.data[2,0]+=1.
    x.data[:,0]+= -100. 
    x.data[:,2]+= 100. 
    f,i,o,C_c = m.gates(x,H)
    assert np.allclose(f.data, [[1,0],[1,0]],atol=1e-2) 

    x.data[0]+= -1000. 
    x.data[1]+= 1000. 
    f,i,o,C_c = m.gates(x,H)
    assert np.allclose(f.data, [[0,0],[1,1]],atol=1e-2) 
    assert np.allclose(C_c.data, [[-1,-1],[1,1]],atol=1e-2) 



#-------------------------------
def test_update_cell():
    '''(5 point) update_cell'''
    # n=2, h=3 
    C  = th.Tensor([
          # instance 0
          [ 1.,2.,3. ],
          # instance 1
          [ 4.,5.,6. ] 
         ])

    C_c  = th.Tensor([
          # instance 0
          [ 6.,5.,4. ],
          # instance 1
          [ 3.,2.,1. ] 
         ])
    f = th.Tensor([1.])
    i = th.Tensor([0.])
    C_new = LSTM.update_cell(C,C_c,f,i)
    assert type(C_new) == th.Tensor 
    assert np.allclose(C_new.data,C.data,atol=1e-2)

    f.data+=-1.
    i.data+=1.
    C_new = LSTM.update_cell(C,C_c,f,i)
    assert np.allclose(C_new.data,C_c.data,atol=1e-2)

    f.data+=.5
    i.data+=-.5
    C_new = LSTM.update_cell(C,C_c,f,i)
    assert np.allclose(C_new.data,3.5*np.ones((2,3)),atol=1e-2)


#-------------------------------
def test_output_hidden_state():
    '''(5 point) output_hidden_state'''
    # n=2, h=3 
    C  = th.Tensor([
          # instance 0
          [ -100.,0., 1. ],
          # instance 1
          [  100.,0.,-1. ] 
         ])

    o = th.Tensor([0.])
    H = LSTM.output_hidden_state(C,o)
    assert type(H) == th.Tensor 
    assert np.allclose(H.data,np.zeros((2,3)),atol=1e-2)

    o.data+=1.
    H = LSTM.output_hidden_state(C,o)
    H_true = [[-1., 0., .7616],
              [ 1., 0.,-.7616]]
    assert np.allclose(H.data,H_true,atol=1e-2)

 
#-------------------------------
def test_forward():
    '''(5 point) forward'''
    # batch_size 3, p = 2
    x = th.ones(3,2)
    m = LSTM(p=2,h=2,c=2)
    H = th.ones(3,2)
    C = th.ones(3,2)
    z,H_new, C_new = m(x,H,C)
    # check value 
    assert type(z) == th.Tensor 
    assert type(H_new) == th.Tensor 
    assert type(C_new) == th.Tensor 
    assert np.allclose(z.size(),[3,2])
    assert np.allclose(H_new.size(),[3,2])
    assert np.allclose(C_new.size(),[3,2])
    assert np.allclose(z.data, np.zeros((3,2)))
    assert np.allclose(C_new.data, .5*np.ones((3,2)),atol=1e-2)
    assert np.allclose(H_new.data, .2311*np.ones((3,2)),atol=1e-2)

    m.b.data+=1.
    z,H_new, C_new = m(x,H,C)
    assert np.allclose(z.data, np.ones((3,2)))

    m.W.data+=1.
    z,H_new, C_new = m(x,H,C)
    assert np.allclose(z.data, 1.4621*np.ones((3,2)),atol=1e-2)
    assert np.allclose(C_new.data, .5*np.ones((3,2)),atol=1e-2)
    assert np.allclose(H_new.data, .2311*np.ones((3,2)),atol=1e-2)

    m.W_f.data+=1.
    z,H_new, C_new = m(x,H,C)
    assert np.allclose(z.data, 1.7539*np.ones((3,2)),atol=1e-2)
    assert np.allclose(H_new.data, .377*np.ones((3,2)),atol=1e-2)
    assert np.allclose(C_new.data, .982*np.ones((3,2)),atol=1e-2)

    m.W_o.data-=1.
    z,H_new, C_new = m(x,H,C)
    assert np.allclose(z.data, 1.0271*np.ones((3,2)),atol=1e-2)
    assert np.allclose(H_new.data, .0135*np.ones((3,2)),atol=1e-2)
    assert np.allclose(C_new.data, .982*np.ones((3,2)),atol=1e-2)


#-------------------------------
def test_train():
    '''(5 point) train '''
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
    m = LSTM(p=2,h=1,c=2)
    m.train(loader,1,1.)
    assert np.allclose(m.b.data,[1.5,-1.5],atol =1e-2)
    assert np.allclose(m.W.data,[0.,0.],atol =1e-2)

    loader = th.utils.data.DataLoader(d, shuffle=True, batch_size = 2)
    m = LSTM(p=2,h=1,c=2)
    m.b_f.data+=1.
    m.b_i.data+=1.
    m.b_o.data+=1.
    m.b_c.data+=1.
    m.train(loader,1000,.1)

    x = th.Tensor([[0.,0.]])
    h = th.Tensor([[0.]])
    c = th.Tensor([[0.]])
    z,ht,ct = m(x,h,c)
    z = z.data.numpy()
    assert z[0,0]>0.1
    assert z[0,1]<-0.1

    x = th.Tensor([[1.,0.]])
    z,ht,ct = m(x,h,c)
    z = z.data.numpy()
    assert z[0,0]<-0.1
    assert z[0,1]>0.1

    x = th.Tensor([[0.,0.]])
    z,ht,ct = m(x,ht,ct)
    z = z.data.numpy()
    assert z[0,0]<-0.1
    assert z[0,1]>0.1

    x = th.Tensor([[0.,1.]])
    z,ht,ct = m(x,h,c)
    z = z.data.numpy()
    assert z[0,0]<-0.1
    assert z[0,1]>0.1

    x = th.Tensor([[0.,0.]])
    z,ht,ct = m(x,ht,ct)
    z = z.data.numpy()
    assert z[0,0]<-0.1
    assert z[0,1]>0.1



