from problem3 import *
from problem2 import Node
import sys
import numpy as np
'''
    Unit test 3:
    This file includes unit tests for problem1.py.
    You could test the correctness of your code by typing `nosetests -v test3.py` in the terminal.
'''
#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 3 (15 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)


#-------------------------------------------------------------------------
def test_bootstrap():
    ''' (3 points) bootstrap'''
    X = np.array([[1.,2.,3.,4. ],
                  [2.,4.,6.,8. ],
                  [3.,6.,9.,12.]])
    Y = np.array( [1.,2.,3.,4. ])

    X1, Y1 = Bag.bootstrap(X,Y)
    assert type(X1) == np.ndarray
    assert type(Y1) == np.ndarray
    assert X1.shape == (3,4)
    assert Y1.shape == (4,)
    assert np.allclose(X1[0]*2, X1[1])
    assert np.allclose(X1[0]*3, X1[2])
    assert np.allclose(X1[0], Y1)


    for _ in range(20):
        p = np.random.randint(10,20)
        n1 = np.random.randint(200,500)
        n2 = np.random.randint(200,500)
        X = np.bmat([np.zeros((p,n1)),np.ones((p,n2))])
        Y = np.bmat([np.ones(n1),np.zeros(n2)]).getA1()
        X1, Y1 = Bag.bootstrap(X,Y)
        assert X1.shape == (p,n1+n2)
        assert Y1.shape == (n1+n2,)
        assert np.allclose(Y1.sum()/(n1+n2), float(n1)/(n1+n2),atol = 0.1)
        assert np.allclose(X1.sum()/(n1+n2)/p, float(n2)/(n1+n2),atol = 0.1)



#-------------------------------------------------------------------------
def test_train():
    ''' (3 points) train'''
    b = Bag()

    X = np.array([[1.,1.,1.,1.],
                  [2.,2.,2.,2.],
                  [3.,3.,3.,3.]])
    Y = np.array(['good','good','good','good'])
    T = b.train(X,Y,1) 
    assert len(T) == 1
    t = T[0]
    assert t.isleaf == True
    assert t.p == 'good' 

    for _ in range(20):
        n_tree = np.random.randint(1,10)
        T = b.train(X,Y,n_tree) 
        assert len(T) == n_tree
        for i in range(n_tree):
            t = T[i]
            assert t.isleaf == True
            assert t.p == 'good' 



#-------------------------------------------------------------------------
def test_inference():
    ''' (3 points) inference'''

    t = Node(None,None) 
    t.isleaf = True
    t.p = 'good job' 
    T = [t,t,t]

    x = np.random.random(10)

    y = Bag.inference(T,x)
    assert y == 'good job' 

    #----------------- 
    t.p = 'c1' 
    t2 = Node(None,None) 
    t2.isleaf = False 
    t2.i = 1
    t2.th = 1.5
    c1 = Node(None,None)
    c2 = Node(None,None)
    c1.isleaf= True
    c2.isleaf= True
    
    c1.p = 'c1' 
    c2.p = 'c2' 
    t2.C1 = c1 
    t2.C2 = c2 

    x = np.array([1.,2.,3.,1.])
    y = Bag.inference([t,t2,t2],x)
    assert y == 'c2' 

    y = Bag.inference([t,t,t2],x)
    assert y == 'c1' 



#-------------------------------------------------------------------------
def test_predict():
    ''' (2 points) predict '''
    t = Node(None,None) 
    t.isleaf = True
    t.p = 'c1' 
    t2 = Node(None,None) 
    t2.isleaf = False 
    t2.i = 1
    t2.th = 1.5
    c1 = Node(None,None)
    c2 = Node(None,None)
    c1.isleaf= True
    c2.isleaf= True
    c1.p = 'c1' 
    c2.p = 'c2' 
    t2.C1 = c1 
    t2.C2 = c2 


    X = np.array([[1.,1.,1.,1.],
                  [1.,2.,3.,1.]])
    Y = Bag.predict([t,t,t2],X)

    assert type(Y) == np.ndarray
    assert Y.shape == (4,) 
    assert Y[0] == 'c1'
    assert Y[1] == 'c1'
    assert Y[2] == 'c1'
    assert Y[3] == 'c1'

    Y = Bag.predict([t,t2,t2],X)
    assert Y[0] == 'c1'
    assert Y[1] == 'c2'
    assert Y[2] == 'c2'
    assert Y[3] == 'c1'


#-------------------------------------------------------------------------
def test_load_dataset():
    ''' (1 points) load dataset'''
    X, Y = Bag.load_dataset()
    assert type(X) == np.ndarray
    assert type(Y) == np.ndarray
    assert X.shape ==(2,800) 
    assert Y.shape ==(800,) 
    assert Y[0] == 0 
    assert Y[1] == 1 
    assert Y[-1] == 1 
    assert Y[-2] == 0 
    assert Y[-3] == 1
    assert np.allclose(X[0,0],0.008609,atol=1e-3) 
    assert np.allclose(X[0,-1],1.959456, atol = 1e-3)
    assert np.allclose(X[1,0], 0.953711, atol = 1e-3) 
    assert np.allclose(X[1,-1],0.1713, atol= 1e-3)

#-------------------------------------------------------------------------
def test_dataset3():
    ''' (3 points) test dataset3'''
    b = Bag()
    X, Y = Bag.load_dataset()
    n = float(len(Y))

    # train over half of the dataset
    T = b.train(X[:,::2],Y[::2],1) 
    # test on the other half
    Y_predict = Bag.predict(T,X[:,1::2]) 
    accuracy = sum(Y[1::2]==Y_predict)/n*2. 
    print('test accuracy of one decision tree:', accuracy)
    assert accuracy >= .95


    # train over half of the dataset
    T = b.train(X[:,::2],Y[::2],11) 
    # test on the other half
    Y_predict = Bag.predict(T,X[:,1::2]) 
    accuracy = sum(Y[1::2]==Y_predict)/n*2. 
    print('test accuracy of a bagging ensemble of 11 trees:', accuracy)
    assert accuracy >= .95


