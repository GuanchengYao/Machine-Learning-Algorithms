from problem2 import *
import sys
import numpy as np
import torch as th
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader

'''
    Unit test 2:
    This file includes unit tests for problem2.py.
    You could test the correctness of your code by typing `nosetests -v test2.py` in the terminal.
'''
#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 2 (30 points in total)--------------'''
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)

#-------------------------------
def test_conv2d():
    '''(4 point) conv2d'''
    # 1 channel, h = 3, w = 3
    X  = th.Tensor([[[1.,2.,3.],
                     [2.,4.,6.],
                     [3.,6.,9.]]])
    # filter size = 2
    W  = th.Tensor([[[1.,0.],
                     [0.,1.]]])
    b = th.zeros(1)
    z = conv2d(X,W,b)
    assert type(z) == Tensor 
    assert np.allclose(z.data, [[5,8],[8,13]])
    b.data+=1.
    z = conv2d(X,W,b)
    assert np.allclose(z.data, [[6,9],[9,14]])
    
    # 2 channels, h = 3, w = 3
    X  = th.Tensor([[[1.,2.,3.],
                     [2.,4.,6.],
                     [3.,6.,9.]], # channel 1
                    [[1.,2.,3.],
                     [2.,4.,6.],
                     [3.,6.,9.]]  # channel 2
                               ])
    # filter size = 2
    W  = th.Tensor([[[1.,0.],
                     [0.,1.]],
                    [[1.,0.],
                     [0.,1.]]])
    b = th.zeros(1)
    z = conv2d(X,W,b)
    assert np.allclose(z.data, [[10,16],[16,26]])
    b.data+=1.
    z = conv2d(X,W,b)
    assert np.allclose(z.data, [[11,17],[17,27]])

    # 2 channels, h = 3, w = 3
    X  = th.Tensor([[[1.,2.,3.],
                     [2.,4.,6.],
                     [3.,6.,9.]], # channel 1
                    [[9.,8.,7.],
                     [6.,5.,4.],
                     [3.,2.,1.]]  # channel 2
                               ])
    # filter size = 2
    W  = th.Tensor([[[ 2., 1.],
                     [ 0.,-1.]],
                    [[-1., 0.],
                     [ 2.,-2.]]])
    b = th.zeros(1)
    z = conv2d(X,W,b)
    assert np.allclose(z.data, [[-7,-5],[-2,2]])

    # 1 channel, h = 2, w = 2
    X  = th.tensor([[[1.,2.],
                  [3.,4.]]],requires_grad=True)
    # filter size = 2
    W  = th.tensor([[[1.,0.],
                  [0.,1.]]],requires_grad=True)
    b = th.tensor([0.],requires_grad=True)
    z = conv2d(X,W,b)
    assert np.allclose(z.data, [5])
    # compute gradient
    z.backward()
    assert np.allclose(W.grad.data,[[1,2],[3,4]])
    assert np.allclose(b.grad.data,[1])
    assert np.allclose(X.grad.data,[[1,0],[0,1]])
 
    X  = th.tensor([[[1.,2.,3.],
                     [4.,5.,6.]]],requires_grad=True)
    # filter size = 2
    W  = th.tensor([[[1.,0.],
                     [0.,1.]]],requires_grad=True)
    b = th.tensor([0.],requires_grad=True)
    z = conv2d(X,W,b)
    assert np.allclose(z.data, [6,8])
    # compute gradient
    s = z[0,0]+z[0,1]
    s.backward()
    assert np.allclose(b.grad.data,[2])
    assert np.allclose(W.grad.data,[[3,5],[9,11]])
    assert np.allclose(X.grad.data,[[1,1,0],[0,1,1]])


#-------------------------------
def test_Conv2D():
    '''(4 point) Conv2D'''
    # batch_size 1, 1 channel, h = 3, w = 3
    X  = th.Tensor([[[[1.,2.,3.],
                      [2.,4.,6.],
                      [3.,6.,9.]]]])
    # 1 filter of size 2 by 2
    W  = th.Tensor([[[[1.,0.],
                      [0.,1.]]]])
    b = th.Tensor([0.])
    z = Conv2D(X,W,b)
    assert type(z) == Tensor 
    assert np.allclose(z.data, [[[[5,8],[8,13]]]])
    b.data+=1.
    z = Conv2D(X,W,b)
    assert np.allclose(z.data, [[[[6,9],[9,14]]]])
    
    n = np.random.randint(2,10) # batch size 
    l = np.random.randint(2,10) # number of channels 
    n_filters = np.random.randint(2,10) # number of filters 
    h = np.random.randint(10,20) # hight of the image 
    w = np.random.randint(10,20) # width of the image 
    s = np.random.randint(2,min(h,w)) # size of the filter 
    X  = th.randn(n,l,h,w)
    W  = th.randn(n_filters,l,s,s)
    b = th.randn(n_filters)
    z = Conv2D(X,W,b) 
    z_true = th.nn.functional.conv2d(X,W,b) 
    assert np.allclose(z.data,z_true.data,atol = 1e-2)


#-------------------------------
def test_ReLU():
    '''(3 point) ReLU'''
    # batch_size 1, 1 filter, h = 3, w = 3
    z  = th.tensor([[[[ 1., 2.,-3.],
                      [ 2.,-4., 6.],
                      [-3., 6., 9.]]]],requires_grad=True)
    a = ReLU(z)
    # check value 
    assert type(a) == Tensor 
    a_true  = [[[[ 1., 2., 0.],
                 [ 2., 0., 6.],
                 [ 0., 6., 9.]]]]
    assert np.allclose(a.data, a_true)
    # check gradient
    t = a.sum()
    t.backward()
    z_grad_true  = [[[[ 1., 1., 0.],
                      [ 1., 0., 1.],
                      [ 0., 1., 1.]]]]
    assert np.allclose(z.grad.data, z_grad_true)
  
#-------------------------------
def test_avgpooling():
    '''(3 point) avgpooling'''
    # batch_size 1, 1 filter, h = 4, w = 4
    a = th.tensor([[[[ 0., 0., 3., 5.],
                     [ 2., 2., 3., 5.],
                     [-2.,-4., 2.,-2.],
                     [-3.,-3.,-5., 5.]]]],requires_grad=True)
    p = avgpooling(a)
    # check value 
    assert type(p) == Tensor 
    p_true  = [[[[ 1., 4.],
                 [-3., 0.]]]]
    assert np.allclose(p.data, p_true)
    # check gradient
    t = p.sum()
    t.backward()
    assert np.allclose(a.grad.data, np.ones((4,4))/4.)
     
    n = np.random.randint(2,5) # batch size 
    n_filters = np.random.randint(2,5) # number of filters 
    h = np.random.randint(2,5)*2 # hight of the image 
    w = np.random.randint(2,5)*2 # width of the image 
    a  = th.randn(n,n_filters,h,w)
    p_true = th.nn.functional.avg_pool2d(a,2) 
    p = avgpooling(a)
    assert np.allclose(p.data,p_true.data,atol = 1e-2)
   
 
#-------------------------------
def test_maxpooling():
    '''(4 point) maxpooling'''
    # batch_size 1, 1 filter, h = 4, w = 4
    a = th.tensor([[[[ 1., 3.,-3., 0.],
                     [ 2.,-4., 6., 2.],
                     [ 2.,-4., 6.,-2.],
                     [-3., 6., 9., 5.]]]],requires_grad=True)
    p = maxpooling(a)
    # check value 
    assert type(p) == Tensor 
    p_true  = [[[[ 3., 6.],
                 [ 6., 9.]]]]
    assert np.allclose(p.data, p_true)
    # check gradient
    t = p.sum()
    t.backward()
    a_grad_true  = [[[[ 0., 1., 0., 0.],
                      [ 0., 0., 1., 0.],
                      [ 0., 0., 0., 0.],
                      [ 0., 1., 1., 0.]]]]
    assert np.allclose(a.grad.data, a_grad_true)
     
    n = np.random.randint(3,5) # batch size 
    n_filters = np.random.randint(2,5) # number of filters 
    h = np.random.randint(2,5)*2 # hight of the image 
    w = np.random.randint(2,5)*2 # width of the image 
    a  = th.randn(n,n_filters,h,w)
    p_true = th.nn.functional.max_pool2d(a,kernel_size=2) 
    p = maxpooling(a)
    assert np.allclose(p.data,p_true.data,atol = 1e-2)

    # check gradient with multiple max values
    a = th.tensor([[[[ 0., 1.],
                     [ 1., 0.]]]],requires_grad=True)
    p = maxpooling(a)
    t = p.sum()
    t.backward()
    
    a_true = th.tensor([[[[ 0., 1.],
                          [ 1., 0.]]]],requires_grad=True)
    p = th.nn.functional.max_pool2d(a_true,kernel_size=2) 
    t = p.sum()
    t.backward()
    assert np.allclose(a.grad.data,a_true.grad.data,atol=1e-2)

    # check gradient with multiple max values
    a = th.tensor([[[[ 0., 0.],
                     [ 1., 1.]]]],requires_grad=True)
    p = maxpooling(a)
    t = p.sum()
    t.backward()
    
    a_true = th.tensor([[[[ 0., 0.],
                          [ 1., 1.]]]],requires_grad=True)
    p = th.nn.functional.max_pool2d(a_true,kernel_size=2) 
    t = p.sum()
    t.backward()
    assert np.allclose(a.grad.data,a_true.grad.data,atol=1e-2)

    # check gradient with multiple max values
    a = th.tensor([[[[ 1., 1.],
                     [ 1., 1.]]]],requires_grad=True)
    p = maxpooling(a)
    t = p.sum()
    t.backward()
    
    a_true = th.tensor([[[[ 1., 1.],
                          [ 1., 1.]]]],requires_grad=True)
    p = th.nn.functional.max_pool2d(a_true,kernel_size=2) 
    t = p.sum()
    t.backward()
    assert np.allclose(a.grad.data,a_true.grad.data,atol=1e-2)



#-------------------------------
def test_num_flat_features():
    '''(2 point) num_flat_features '''
    assert type(num_flat_features(4,4,3,1)) == int
    assert num_flat_features(4,4,3,1) == 1
    assert num_flat_features(4,4,3,2) == 2
    assert num_flat_features(6,6,3,2) == 8
    assert num_flat_features(6,6,5,2) == 2

#-------------------------------
def test_init():
    '''(2 point) init'''
    l = np.random.randint(1,3)
    h = np.random.randint(20,100)
    w = np.random.randint(20,100)
    s = np.random.randint(1,10)
    n_filters = np.random.randint(1,100)
    c = np.random.randint(1,100)

    m = CNN(l,h, w, s, n_filters, c)
    assert type(m) == CNN 
    assert isinstance(m, Module)
    assert type(m.conv_W) == Tensor 
    assert type(m.conv_b) == Tensor 
    assert type(m.W) == Tensor 
    assert type(m.b) == Tensor 
    assert type(m.loss_fn) == CrossEntropyLoss 
    assert m.conv_W.requires_grad== True
    assert m.conv_b.requires_grad == True
    assert m.W.requires_grad== True
    assert m.b.requires_grad == True
    assert m.conv_W.dtype == th.float
    assert m.conv_b.dtype == th.float
    assert m.W.dtype == th.float
    assert m.b.dtype == th.float
    p=num_flat_features(h,w,s,n_filters)
    assert np.allclose(m.conv_W.data.size(), [n_filters,l,s,s])
    assert np.allclose(m.conv_b.data.size(), [n_filters]) 
    assert np.allclose(m.W.data.size(), [p,c])
    assert np.allclose(m.b.data.size(), [c]) 
    assert np.allclose(m.conv_W.data, np.zeros((n_filters,l,s,s)))
    assert np.allclose(m.conv_b.data, np.ones(n_filters)) 
    assert np.allclose(m.b.data, np.zeros(c)) 
    assert np.allclose(m.W.data, np.zeros((p,c)))
    assert np.allclose(m.b.data, np.zeros(c)) 

 
#-------------------------------
def test_forward():
    '''(4 point) forward'''
    # batch_size 1, 1 channel, h = 3, w = 3
    x = th.Tensor([[[[ 1., 3.,-3.],
                     [ 2.,-4., 6.],
                     [-3., 6., 9.]]]])
    m = CNN(1,3,3,2,1,2)
    z = m(x)
    # check value 
    assert type(z) == Tensor 
    assert np.allclose(z.data, [0,0])

    m.b.data+=1.
    z = m(x)
    assert np.allclose(z.data, [1,1])

    m.W.data+=1.
    z = m(x)
    assert np.allclose(z.data, [2,2])
 
    m.conv_W.data[0,0,0,0]+=1.
    z = m(x)
    assert np.allclose(z.data, [5,5])

    m.conv_W.data[0,0,1,1]+=1.
    z = m(x)
    assert np.allclose(z.data, [11,11])

    # check gradient 
    l = z.sum()
    l.backward()
    assert np.allclose(m.W.grad.data,[10,10])
    assert np.allclose(m.b.grad.data,[1,1])
    assert np.allclose(m.conv_W.grad.data[0,0],[[6,-6],[-8,12]])
    assert np.allclose(m.conv_b.grad.data,[2])

    n = np.random.randint(2,5) # batch size 
    l = np.random.randint(1,3) # nubmer channels
    h = np.random.randint(5,10)*2
    w = h 
    s = np.random.randint(1,2)*2+1
    n_filters = np.random.randint(1,10)
    c = np.random.randint(1,10)

    m = CNN(l,h, w, s, n_filters, c)
    x  = th.randn(n,l,h,w)
    z = m(x)

#-------------------------------
def test_train():
    '''(4 point) train '''
    # n=2, 1 channel, h = 3, w = 3
    X  = [
          [ # Instance 0 
           [ # channel 0
            [0.,1.,1.], 
            [0.,1.,1.],
            [0.,1.,1.]
           ]
          ], 
          [ # Instance 1 
           [ # channel 0
            [0.,0.,0.],
            [1.,1.,1.],
            [1.,1.,1.]
           ]
          ] 
         ]
    Y = [0,1]
    class toy(Dataset):
        def __init__(self):
            self.X  = th.Tensor(X)            
            self.Y = th.LongTensor(Y)
        def __len__(self):
            return 2 
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    d = toy()
    loader = th.utils.data.DataLoader(d, batch_size = 1)
    m = CNN(h=3,w=3,s=2,n_filters=1,c=2)
    m.train(loader,1,1.)
    assert np.allclose(m.b.data,[.5,-.5],atol =1e-2)
    assert np.allclose(m.W.data,[.5,-.5],atol =1e-2)
    assert np.allclose(m.conv_b.data,[1],atol =1e-2)
    assert np.allclose(m.conv_W.data[0,0],np.zeros((2,2)),atol =1e-2)

    m.train(loader,1,1.)
    assert np.allclose(m.b.data,[0.6192,-0.6192],atol =1e-2)
    assert np.allclose(m.W.data,[0.6192,-0.6192],atol =1e-2)
    assert np.allclose(m.conv_b.data,[1.1192],atol =1e-2)
    assert np.allclose(m.conv_W.data[0,0],np.mat('0.,0.1192;0.,0.1192'),atol =1e-2)

    m = CNN(h=3,w=3,s=2,n_filters=1,c=2)
    m.train(loader,2,1.)
    assert np.allclose(m.b.data,[-0.3808,0.3808],atol =1e-2)
    assert np.allclose(m.W.data,[-0.3808,0.3808],atol =1e-2)
    assert np.allclose(m.conv_b.data,[0.1192],atol =1e-2)
    assert np.allclose(m.conv_W.data[0,0],np.mat('0.,0.;-0.8808,-0.8808'),atol =1e-2)
 
    X = th.ones(10,3,28,28) 
    Y = [0,1,0,1,0,1,0,1,0,1]
    class toy2(Dataset):
        def __init__(self):
            self.X  = X 
            self.Y = th.LongTensor(Y)
        def __len__(self):
            return 10 
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    d = toy2()

    # if this test runs out of memory, you could reduce the batch_size (n)
    loader = th.utils.data.DataLoader(d, batch_size = 5)
    m = CNN(l=3,c=2)
    m.train(loader,5,1.)
    assert np.allclose(m.conv_b.data, np.ones(5)*(-16.28), atol=1e-2)
    assert np.allclose(m.conv_W.data, np.ones((5,3,5,5))*(-17.28), atol=1e-2)
    assert np.allclose(m.b.data,[0.006,-0.006],atol=1e-3)
    assert np.allclose(m.W.data,(np.mat('-0.5;0.5')*np.ones((1,720))).T)

 
