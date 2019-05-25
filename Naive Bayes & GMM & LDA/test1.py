from problem1 import *
import numpy as np
import sys

'''
    Unit test 1:
    This file includes unit tests for problem1.py.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 1 (20 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)


#-------------------------------------------------------------------------
def test_estimate_prob_smooth():
    ''' (2 points) estimate_prob_smooth'''
    X = np.array([0,0,0])
    P = prob_smooth(X,2,0)
    assert np.allclose(P, [1.,0.], atol = 1e-3) 
    P = prob_smooth(X,2,1)
    assert np.allclose(P, [.8,.2], atol = 1e-3) 
    X = np.array([0,0,0,1])
    P = prob_smooth(X,3,2)
    assert np.allclose(P, [.5,.3,.2], atol = 1e-3) 

#-------------------------------------------------------------------------
def test_compute_class_prior():
    ''' (3 points) compute_class_prior'''
    y = np.array([0,0])
    py = class_prior(y)
    assert np.allclose(py, [1., 0.], atol = 1e-3) 

    y = np.array([1,1])
    py = class_prior(y)
    assert np.allclose(py, [0.,1.], atol = 1e-3) 

    y = np.array([0,1,0,1])
    py = class_prior(y)
    assert np.allclose(py, [.5, .5], atol = 1e-3) 

#-------------------------------------------------------------------------
def test_conditional_prob():
    ''' (2 points) compute_conditional_prob'''
    X = np.array([0,0,1,1])
    Y = np.array([0,0,1,1])
    PX_Y = conditional_prob(X,Y,k=0)
    assert type(PX_Y) == np.ndarray
    assert np.allclose(PX_Y, [[1,0],[0,1]])
    PX_Y = conditional_prob(X,Y,k=1)
    assert np.allclose(PX_Y, [[.75,.25],[.25,.75]])

    X = np.array([0,1,0,1])
    Y = np.array([0,0,1,1])
    PX_Y = conditional_prob(X,Y,k=0)
    assert type(PX_Y) == np.ndarray
    assert np.allclose(PX_Y,.5*np.ones((2,2)))


    X = np.array([0,1,1,1])
    Y = np.array([0,0,1,1])
    PX_Y = conditional_prob(X,Y,k=0)
    assert np.allclose(PX_Y, [[.5,.5],[.0,1.]])


#-------------------------------------------------------------------------
def test_train():
    ''' (3 points) train'''
    # two attributes (p=2), 4 instances (n=4)
    X = np.array([[0,0,1,1],
                  [0,1,0,1]])
    Y = np.array([0,0,1,1])
    PX_Y,PY = train(X,Y,k=0)
    assert type(PX_Y) == np.ndarray
    assert type(PY) == np.ndarray
    assert np.allclose(PY, [.5,.5])
    PX_Y_true = [
                 [# attribute 0 (X0)
                    [1.,0.], # Prob of X0 = 0 or 1 given Y = 0 
                    [0.,1.]  # Prob of X0 = 0 or 1 given Y = 1 
                 ], 
                 [# attribute 1 (X1)
                    [.5,.5], # Prob of X1 = 0 or 1 given Y = 0
                    [.5,.5]  # Prob of X1 = 0 or 1 given Y = 1
                 ]  
                ]
    assert np.allclose(PX_Y, PX_Y_true)


    # smoothing
    PX_Y,PY = train(X,Y,k=1)
    assert np.allclose(PY, [.5,.5])
    PX_Y_true = [
                 [# attribute 0 (X0)
                    [.75,.25], # Prob of X0 = 0 or 1 given Y = 0 
                    [.25,.75]  # Prob of X0 = 0 or 1 given Y = 1 
                 ], 
                 [# attribute 1 (X1)
                    [.5,.5], # Prob of X1 = 0 or 1 given Y = 0
                    [.5,.5]  # Prob of X1 = 0 or 1 given Y = 1
                 ]  
                ]
    assert np.allclose(PX_Y, PX_Y_true)



#-------------------------------------------------------------------------
def test_inference():
    ''' (2 points) inference'''
    # two attributes (p=2)
    X = np.array([0,0])
    PY = np.array([.5,.5])
    PX_Y = [
            [# attribute 0 (X0)
               [.5,.5], # Prob of X0 = 0 or 1 given Y = 0 
               [.5,.5]  # Prob of X0 = 0 or 1 given Y = 1 
            ], 
            [# attribute 1 (X1)
               [.5,.5], # Prob of X1 = 0 or 1 given Y = 0
               [.5,.5]  # Prob of X1 = 0 or 1 given Y = 1
            ]  
           ]
    PX_Y = np.array(PX_Y)
    Y, P = inference(X,PY,PX_Y)
    assert Y == 0 # when there is a tie, predict 0
    assert np.allclose(P, [.5,.5], atol = 1e-2)

    # two attributes (p=2)
    X = np.array([0,1])
    PY = np.array([.4,.6])
    PX_Y = [
            [# attribute 0 (X0)
               [.2,.8], # Prob of X0 = 0 or 1 given Y = 0 
               [.3,.7]  # Prob of X0 = 0 or 1 given Y = 1 
            ], 
            [# attribute 1 (X1)
               [.1,.9], # Prob of X1 = 0 or 1 given Y = 0
               [.5,.5]  # Prob of X1 = 0 or 1 given Y = 1
            ]  
           ]
    PX_Y = np.array(PX_Y)
    Y, P = inference(X,PY,PX_Y)
    assert Y == 1
    assert np.allclose(P, [.444,.555], atol = 1e-2)

    X = np.array([1,1])
    Y, P = inference(X,PY,PX_Y)
    assert Y == 0
    assert np.allclose(P, [.578,.422], atol = 1e-2)

    X = np.array([1,0])
    Y, P = inference(X,PY,PX_Y)
    assert Y == 1
    assert np.allclose(P, [.132,.868], atol = 1e-2)

    X = np.array([0,0])
    Y, P = inference(X,PY,PX_Y)
    assert Y == 1
    assert np.allclose(P, [.082,.918], atol = 1e-2)


#-------------------------------------------------------------------------
def test_predict():
    ''' (3 points) predict'''
    # two attributes (p=2), 4 instances
    X = np.array([ [0,0,1,1],
                   [0,1,0,1]])
    PY = np.array([.4,.6])
    PX_Y = [
            [# attribute 0 (X0)
               [.2,.8], # Prob of X0 = 0 or 1 given Y = 0 
               [.3,.7]  # Prob of X0 = 0 or 1 given Y = 1 
            ], 
            [# attribute 1 (X1)
               [.1,.9], # Prob of X1 = 0 or 1 given Y = 0
               [.5,.5]  # Prob of X1 = 0 or 1 given Y = 1
            ]  
           ]
    PX_Y = np.array(PX_Y)
    Y = predict(X,PY,PX_Y)
    assert np.allclose(Y, [1,1,1,0], atol = 1e-2)

#-------------------------------------------------------------------------
def test_dataset1():
    ''' (5 points) test dataset1'''
    # load spam email dataset
    X = np.loadtxt('data1.csv', dtype=int, delimiter=',',skiprows=1, unpack=True)
    X = X[1:]
    Y = np.loadtxt('data1.csv', dtype=int, delimiter=',',skiprows=1,usecols=0, unpack=True)
    X_train = X[:,0::2]
    Y_train = Y[0::2]
    X_test = X[:,1::2]
    Y_test = Y[1::2]

    # without smoothing
    PX_Y, PY = train(X_train, Y_train,k =0)
    Y_pred =  predict(X_train,PY, PX_Y)
    accuracy = sum(Y_train==Y_pred)/2300. 
    print('training accuracy:', accuracy)
    assert accuracy > .87
    Y_pred =  predict(X_test,PY, PX_Y)
    accuracy = sum(Y_test==Y_pred)/2300. 
    print('test accuracy:', accuracy)
    assert accuracy > .85

    # with too much smoothing
    PX_Y, PY = train(X_train, Y_train,k =10000)
    Y_pred =  predict(X_train,PY, PX_Y)
    accuracy = sum(Y_train==Y_pred)/2300. 
    print('training accuracy:', accuracy)
    assert accuracy < .7

    Y_pred =  predict(X_test,PY, PX_Y)
    accuracy = sum(Y_test==Y_pred)/2300. 
    print('test accuracy:', accuracy)
    assert accuracy < .7
     
