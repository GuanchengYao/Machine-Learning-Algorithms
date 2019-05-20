from problem1 import *
import sys
import numpy as np
import torch as th
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_classification

'''
    Unit test 1:
    This file includes unit tests for problem1.py.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''
#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 1 (20 points in total)--------------'''
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)

#-------------------------------
def test_softmax_init():
    '''(3 point) init'''
    p = np.random.randint(1,100)
    c = np.random.randint(1,100)
    m = SoftmaxRegression(p, c)
    assert type(m) == SoftmaxRegression 
    assert isinstance(m, Module)
    assert type(m.W) == Tensor 
    assert type(m.b) == Tensor 
    assert type(m.loss_fn) == CrossEntropyLoss 
    assert m.W.requires_grad== True
    assert m.b.requires_grad == True
    assert m.W.dtype == th.float
    assert m.b.dtype == th.float
    assert np.allclose(m.W.size(), [p,c])
    assert np.allclose(m.b.size(), [c]) 
    assert np.allclose(m.W.data, np.zeros((p,c)))
    assert np.allclose(m.b.data,np.zeros(c)) 

#-------------------------------
def test_softmax_forward():
    '''(3 point) forward'''
    x = th.Tensor(np.mat('1.,1.; 2.,2.;3.,3.')) # 3 (batch_size) by 2 (p) matrix
    m = SoftmaxRegression(2, 3) # p = 2, c = 3
    z = m(x)
    
    assert type(z) == Tensor 
    assert np.allclose(z.size(), (3,3))

    z_true = np.zeros((3,3))
    assert np.allclose(z.data,z_true, atol = 1e-3)

    m.b[0]+=100.
    z = m(x)
    z_true = np.zeros((3,3))
    z_true[:,0] = 100.
    assert np.allclose(z.data, z_true, atol = 1e-3)

    m.b[0]-=100.
    m.W=m.W+1.
    z = m(x)
    z_true = np.mat('2,2,2;4,4,4;6,6,6')
    assert np.allclose(z.data, z_true, atol = 1e-3)


#-------------------------------------------------------------------------
def test_compute_L():
    '''(2 point) compute_L'''
    z = th.Tensor(np.mat('-1000.,0.;0.,1000.'))
    y = th.LongTensor([1,1])
    m = SoftmaxRegression(3, 2)
    L = m.compute_L(z,y)

    assert type(L) == Tensor 
    assert np.allclose(L.data, 0., atol = 1e-3) 

    z = th.Tensor(np.mat('0.,0.;0.,0.'))
    y = th.LongTensor([1,1])
    L = m.compute_L(z,y)
    assert np.allclose(L.data, 0.693147, atol = 1e-3) 

#-------------------------------------------------------------------------
def test_backward():
    '''(2 point) backward'''
    x = th.Tensor(np.mat('1.,2.; 1.,2.;1.,2.')) # 3 (batch_size) by 2 (p) matrix
    y = th.LongTensor([2,2,2])
    m = SoftmaxRegression(2, 3) # p = 2, c = 3
    z = m(x)
    L = m.compute_L(z,y)
    m.backward(L)
    dL_dW, dL_db = m.W.grad, m.b.grad
    dW_true = np.mat('1.,2.;1.,2.;-2.,-4.')/3.
    db_true = [1./3.,1./3.,-2./3.]
    assert np.allclose(dL_dW.data,dW_true.T,atol=1e-3) 
    assert np.allclose(dL_db.data,db_true,atol=1e-3) 


    x = th.Tensor(np.mat('1.,2.; 1.,2.')) # 2 (batch_size) by 2 (p) matrix
    y = th.LongTensor([2,2])
    m = SoftmaxRegression(2, 3) # p = 2, c = 3
    z = m(x)
    L = m.compute_L(z,y)
    m.backward(L)
    dL_dW, dL_db = m.W.grad, m.b.grad
    dW_true = np.mat('1.,2.;1.,2.;-2.,-4.')/3.
    db_true = [1./3.,1./3.,-2./3.]
    assert np.allclose(dL_dW.data,dW_true.T,atol=1e-3) 
    assert np.allclose(dL_db.data,db_true,atol=1e-3) 

#-------------------------------------------------------------------------
def test_train():
    '''(4 point) train'''

    # create a toy dataset for testing mini-batch training
    class toy1(Dataset):
        def __init__(self):
            self.X  = th.Tensor([[1., 1.], [1., 1.]])
            self.Y = th.LongTensor([1, 1])

        def __len__(self):
            return 2 
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    d = toy1()
    loader = DataLoader(d, batch_size=2, shuffle=False, num_workers=1)


    m = SoftmaxRegression(2, 2) # p = 2, c = 2
    # call the function
    m.train(loader, 1,1.)
   
    W = m.W.data
    b = m.b.data
    # after each update, gradients should be zero
    assert np.allclose(m.W.grad.data, np.zeros((2,2))) 
    assert np.allclose(m.b.grad.data, np.zeros(2))
    assert np.allclose(m.W.data, np.mat([[-1,1],[-1,1]])/2.)
    assert np.allclose(m.b.data, np.mat([-1,1])/2.)


    m = SoftmaxRegression(2, 2) # p = 2, c = 2
    # call the function
    m.train(loader, 1,0.1)
    W = m.W.data
    b = m.b.data
    # after each update, gradients should be zero
    assert np.allclose(m.W.data, np.mat([[-1,1],[-1,1]])/20.)
    assert np.allclose(m.b.data, np.mat([-1,1])/20.)


    # create another toy dataset for testing mini-batch training
    class toy2(Dataset):
        def __init__(self):
            self.X  = th.Tensor([[0., 1.],
                                 [1., 0.],
                                 [0., 0.],
                                 [1., 1.]])            
            self.Y = th.LongTensor([0, 1, 0, 1])

        def __len__(self):
            return 4 
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    d = toy2()
    loader = DataLoader(d, batch_size=2, shuffle=False, num_workers=1)


    m = SoftmaxRegression(2, 2) # p = 2, c = 2

    # call the function
    m.train(loader)
   
    W = m.W.data
    b = m.b.data
    assert b[0] > b[1] # x3 is negative 
    assert W[0,1] + W[1,1] + b[1] > W[0,0] + W[1,0] + b[0] # x4 is positive
    assert W[1,1] + b[1] < W[1,0] + b[0] # x1 is negative 
    assert W[0,1] + b[1] > W[0,0] + b[0] # x2 is positive 

  
#-------------------------------------------------------------------------
def test_predict():
    '''(2 points) test'''
    # create a toy dataset for testing mini-batch training
    class toy(Dataset):
        def __init__(self):
            self.X  = th.Tensor([[0., 1.],
                                 [1., 0.],
                                 [0., 0.],
                                 [1., 1.]])            
            self.Y = th.LongTensor([0, 0, 0, 1])

        def __len__(self):
            return 4 
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    d = toy()
    loader = DataLoader(d, batch_size=2, shuffle=False, num_workers=1)
    m = SoftmaxRegression(2, 2) # p = 2, c = 2
    m.b.data[0]+=.1

    # call the function
    acc = m.test(loader)
    assert acc == 0.75

    m.b.data[0]-=.2
    # call the function
    acc = m.test(loader)
    assert acc == 0.25



#-------------------------------------------------------------------------
def test_softmax_regression():
    '''(4 point) softmax regression'''
    
    okay = False
    # test on 5 datasets
    for random_seed in range(1,6):
        # create a multi-class classification dataset
        n_samples = 400
        X,y = make_classification(n_samples= n_samples,
                                  n_features=5, n_redundant=0, n_informative=4,
                                  n_classes= 3,
                                  class_sep = 3.,
                                  random_state=random_seed)
        Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]
        #print('X.shape:',X.shape)
        #print('Y.shape:',y.shape)
        #print('X.type:',type(X[0,0]))
        #print('y.type:',type(y[0]))

        class toy_train(Dataset):
            def __init__(self):
                self.X  = th.Tensor(Xtrain)            
                self.Y = th.LongTensor(Ytrain)
            def __len__(self):
                return int(n_samples/2)
            def __getitem__(self, idx):
                return self.X[idx], self.Y[idx]

        class toy_test(Dataset):
            def __init__(self):
                self.X  = th.Tensor(Xtest)            
                self.Y = th.LongTensor(Ytest)
            def __len__(self):
                return int(n_samples/2)
            def __getitem__(self, idx):
                return self.X[idx], self.Y[idx]

        dtr = toy_train()
        loader_train = DataLoader(dtr, batch_size=10, shuffle=False, num_workers=1)
            
        dte = toy_test()
        loader_test = DataLoader(dte, batch_size=10, shuffle=False, num_workers=1)

        m = SoftmaxRegression(5, 3)

        # call the function
        m.train(loader_train)
        accuracy_train = m.test(loader_train)
        print('Training accuracy:', accuracy_train)
        accuracy_test = m.test(loader_test)
        print('Test accuracy:', accuracy_test)

        if accuracy_train > 0.9 and accuracy_test > 0.9:
            okay = True
            break 
    assert okay 



