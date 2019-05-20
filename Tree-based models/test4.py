from problem4 import *
from problem2 import DT
import sys
import numpy as np
'''
    Unit test 4:
    This file includes unit tests for problem1.py.
    You could test the correctness of your code by typing `nosetests -v test4.py` in the terminal.
'''
#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 4 (10 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)


#-------------------------------------------------------------------------
def test_best_attribute():
    ''' (5 points) best attribute'''
    r = RF()
    X = np.array([[1.,2.],
                  [3.,4.]])
    Y = np.array(['good','bad'])
    c = 0
    for _ in range(50):
        i,th = r.best_attribute(X,Y)
        
        assert i == 0 or i == 1
        if i==0:
            assert th == 1.5
            c+=1
        else:
            assert th == 3.5
    assert c>=15
    assert c<=35


    X = np.array([[0.,0.,1.,1.],
                  [1.,2.,1.,2.],
                  [3.,4.,3.,4.],
                  [2.,3.,2.,3.]])
    Y = np.array(['good','bad','good','bad'])
    i,th = r.best_attribute(X,Y)
    assert i == 1 or i == 2 or i == 3
    if i ==1:
        assert th == 1.5
    elif i == 2:
        assert th == 3.5
    elif i == 3:
        assert th == 2.5
        


    X = np.array([[0.,1.,1.,1.],
                  [1.,1.,2.,2.],
                  [2.,2.,2.,3.],
                  [1.,1.,1.,1.]])
    Y = np.array(['good','bad','perfect','okay'])
    c = np.zeros(3)
    for _ in range(200):
        i,th = r.best_attribute(X,Y)
        assert i == 1 or i==2 or i == 0 
        c[i]+=1

    assert c[1] - c[0] - c[2] <= 40 
    assert c[1] - c[0] - c[2] >= -40 


#-------------------------------------------------------------------------
def test_dataset3():
    ''' (2 points) test dataset3'''
    r = RF()
    X, Y = Bag.load_dataset()
    n = float(len(Y))

    # train over half of the dataset
    T = r.train(X[:,::2],Y[::2],11) 
    # test on the other half
    Y_predict = RF.predict(T,X[:,1::2]) 
    accuracy = sum(Y[1::2]==Y_predict)/n*2. 
    print('test accuracy of a random forest of 11 trees:', accuracy)
    assert accuracy >= .95


#-------------------------------------------------------------------------
def test_load_dataset():
    ''' (1 points) load dataset4'''
    X, Y = RF.load_dataset()
    assert type(X) == np.ndarray
    assert type(Y) == np.ndarray
    assert X.shape ==(16,400) 
    assert Y.shape ==(400,) 
    assert Y[0] == 0 
    assert Y[1] == 0 
    assert Y[2] == 1
    assert Y[-1] == 0
    assert Y[-2] == 0 
    assert np.allclose(X[0,0],0.328805,atol=1e-3) 
    assert np.allclose(X[-1,-1],-1.246555, atol= 1e-3)



#-------------------------------------------------------------------------
def test_dataset4():
    ''' (2 points) test dataset4'''
    n= 400
    X, Y = RF.load_dataset()

    assert X.shape == (16,400)
    assert Y.shape == (400,)
    d = DT()
    # train over half of the dataset
    t = d.train(X[:,::2],Y[::2]) 
    # test on the other half
    Y_predict = DT.predict(t,X[:,1::2]) 
    accuracy0 = sum(Y[1::2]==Y_predict)/float(n)*2. 
    print('test accuracy of a decision tree:', accuracy0)

    b = Bag()
    # train over half of the dataset
    T = b.train(X[:,::2],Y[::2],21) 
    # test on the other half
    Y_predict = Bag.predict(T,X[:,1::2]) 
    accuracy1 = sum(Y[1::2]==Y_predict)/float(n)*2. 
    print('test accuracy of a bagging of 21 trees:', accuracy1)

    r = RF()
    # train over half of the dataset
    T = r.train(X[:,::2],Y[::2],21) 
    # test on the other half
    Y_predict = RF.predict(T,X[:,1::2]) 
    accuracy2 = sum(Y[1::2]==Y_predict)/float(n)*2. 
    print('test accuracy of a random forest of 21 trees:', accuracy2)
    assert accuracy1 >= accuracy0
    assert accuracy2 >= accuracy0
    assert accuracy2 >= accuracy1-.05

