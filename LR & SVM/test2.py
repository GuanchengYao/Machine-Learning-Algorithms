from problem2 import *
import numpy as np
import sys
from sklearn.datasets import make_classification
'''
    Unit test 2:
    This file includes unit tests for problem2.py.
    You could test the correctness of your code by typing `nosetests -v test2.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 2 (30 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)


#-------------------------------------------------------------------------
def test_predict():
    ''' (5 points) predict'''

    X = np.mat([[0.,1.],
                [1.,0.],
                [0.,0.],
                [1.,1.]])
    w = np.mat('1.;1.')
    b = -.5 
    y = predict(X,w,b)
    assert type(y) == np.matrixlib.defmatrix.matrix 
    assert y.shape == (4,1)
    assert np.allclose(y, np.mat('1;1;-1;1'), atol = 1e-3) 

    b = -1.5 
    y = predict(X,w,b)
    assert np.allclose(y, np.mat('-1;-1;-1;1'), atol = 1e-3) 

    w = np.mat('2.;1.')
    b = -1.5 
    y = predict(X,w,b)
    assert np.allclose(y, np.mat('-1;1;-1;1'), atol = 1e-3) 


#-------------------------------------------------------------------------
def test_subgradient():
    ''' (5 points) subgradient'''

    x = np.mat('1.;1.')
    y = -1.
    w = np.mat('1.;1.')
    b = -1.
    dL_dw, dL_db = subgradient(x,y,w,b,l=1.)
    assert type(dL_dw) == np.matrixlib.defmatrix.matrix 
    assert dL_dw.shape == (2,1)
    assert np.allclose(dL_dw, np.mat('2;2'), atol = 1e-3) 
    assert type(dL_db) == float 
    assert dL_db == 1.

    x = np.mat('1.;2.')
    dL_dw, dL_db = subgradient(x,y,w,b,l=1.)
    assert np.allclose(dL_dw, np.mat('2;3'), atol = 1e-3) 
    assert dL_db == 1.

    x = np.mat('2.;1.')
    dL_dw, dL_db = subgradient(x,y,w,b,l=1.)
    assert np.allclose(dL_dw, np.mat('3;2'), atol = 1e-3) 
    assert dL_db == 1.

    x = np.mat('1.;1.')
    dL_dw, dL_db = subgradient(x,y,w,b,l=.5)
    assert np.allclose(dL_dw, np.mat('1.5;1.5'), atol = 1e-3) 
    assert dL_db == 1.

    x = np.mat('2.;2.')
    y = 1.
    dL_dw, dL_db = subgradient(x,y,w,b,l=1.)
    assert np.allclose(dL_dw, np.mat('1.;1.'), atol = 1e-3) 
    assert dL_db == 0.

    dL_dw, dL_db = subgradient(x,y,w,b,l=.5)
    assert np.allclose(dL_dw, np.mat('.5;.5'), atol = 1e-3) 
    assert dL_db == 0.

    w = np.mat('2.;1.')
    dL_dw, dL_db = subgradient(x,y,w,b,l=.5)
    assert np.allclose(dL_dw, np.mat('1.;.5'), atol = 1e-3) 
    assert dL_db == 0.

    x = np.mat('1.;1.')
    w = np.mat('1.;1.')
    dL_dw, dL_db = subgradient(x,y,w,b,l=.5)
    assert np.allclose(dL_dw, np.mat('.5;.5'), atol = 1e-3) 
    assert dL_db == 0.


   

#-------------------------------------------------------------------------
def test_update_w():
    ''' (5 points) update_w'''
    w = np.mat('1.;1.')
    dL_dw = np.mat('2.;3.')
    w_new = update_w(w,dL_dw,1.)
    assert type(w_new) == np.matrixlib.defmatrix.matrix 
    assert np.allclose(w_new, np.mat('-1;-2'))

    w_new = update_w(w,dL_dw,.5)
    assert np.allclose(w_new, np.mat('0;-.5'))

    w = np.mat('4.;6.')
    w_new = update_w(w,dL_dw,1.)
    assert np.allclose(w_new, np.mat('2;3'))


#-------------------------------------------------------------------------
def test_update_b():
    ''' (5 points) update_b'''
    b = 1. 
    dL_db = 2. 
    b_new = update_b(b,dL_db,1.)
    assert np.allclose(b_new, -1)

    b_new = update_b(b,dL_db,.5)
    assert np.allclose(b_new, 0)



#-------------------------------------------------------------------------
def test_train():
    '''(5 point) train'''
    # an example feature matrix (2 instances, 2 features)
    X  = np.mat( [[0., 0.],
                  [1., 1.]])
    Y = np.mat([-1., 1.]).T
    w, b = train(X, Y, 0.01,n_epoch = 1000)
    assert np.allclose(w[0]+w[1]+ b, 1.,atol = 0.1)  # x2 a positive support vector 
    assert np.allclose(b, -1.,atol =0.1)  # x1 a negative support vector 

    #------------------
    # another example
    X  = np.mat( [[0., 1.],
                  [1., 0.],
                  [2., 0.],
                  [0., 2.]])
    Y = np.mat([-1., -1., 1., 1.]).T
    w, b = train(X, Y, 0.01, C= 10000., n_epoch = 1000)
    assert np.allclose(w[0]+b, -1, atol = 0.1)
    assert np.allclose(w[1]+b, -1, atol = 0.1)
    assert np.allclose(w[0]+w[1]+b, 1, atol = 0.1)
 

    w, b = train(X, Y, 0.01, C= 0.01, n_epoch = 1000)
    assert np.allclose(w, [0,0], atol = 0.1)

    #------------------
    X  = np.mat( [[0., 0.],
                  [1., 1.],
                  [0., 10]])
    Y = np.mat([-1., 1., 1.]).T
    w, b = train(X, Y, 0.01, C= 100000., n_epoch = 1000)
    assert np.allclose(b, -1, atol = 0.1)
    assert np.allclose(w, np.mat('1;1'), atol = 0.1)

    #------------------
    X  = np.mat( [[0., 0.],
                  [2., 2.],
                  [0., 190]])
    Y = np.mat([-1., 1., 1.]).T
    w, b = train(X, Y, 0.01, C= 100000., n_epoch = 1000)
    assert np.allclose(b, -1, atol = 0.1)
    assert np.allclose(w, np.mat('.5;.5'), atol = 0.1)


    #------------------
    X  = np.mat( [[0., 0.],
                  [5., 5.],
                  [0., 190]])
    Y = np.mat([-1., 1., 1.]).T
    w, b = train(X, Y, 0.01, C= 100000., n_epoch = 1000)
    assert np.allclose(b, -1, atol = 0.1)
    assert np.allclose(w, np.mat('.2;.2'), atol = 0.1)


    #------------------
    X  = np.mat( [[0., 0.],
                  [1., 1.],
                  [0., 1.5]])
    Y = np.mat([-1., 1., 1.]).T
    w, b = train(X, Y, 0.01, C= 100000., n_epoch = 1000)
    assert np.allclose(b, -1, atol = 0.1)
    assert np.allclose(w, np.mat('.68;1.34'), atol = 0.1)

    #------------------
    X  = np.mat( [[ 10., 0.],
                  [ 0., 10.],
                  [-10., 0.],
                  [ 0.,-10.]])
    Y = np.mat([ 1., 1.,-1.,-1.]).T
    w, b = train(X, Y, 0.001, C= 1e10, n_epoch = 1000)
    assert np.allclose(b, 0, atol = 0.1)
    assert np.allclose(w, np.mat('.1;.1'), atol = 0.1)

    #------------------
    X  = np.mat( [[ 15., 0.],
                  [ 0., 10.],
                  [-15., 0.],
                  [ 0.,-10.]])
    Y = np.mat([ 1., 1.,-1.,-1.]).T
    w, b = train(X, Y, 0.001, C= 1e10, n_epoch = 1000)
    assert np.allclose(b, 0, atol = 0.1)
    assert np.allclose(w, np.mat('.1;.1'), atol = 0.1)




#-------------------------------------------------------------------------
def test_svm():
    '''(5 point) test svm'''
    # create a binary classification dataset
    n_samples = 200
    X,y = make_classification(n_samples= n_samples,
                              n_features=2, n_redundant=0, n_informative=2,
                              n_classes= 2,
                              class_sep = 2.,
                              random_state=1)
    X = np.asmatrix(X)
    y = np.asmatrix(y).T
    y[y==0]=-1
        
    Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]
    w,b = train(Xtrain, Ytrain, .001, C=1000., n_epoch=500)
    Y = predict(Xtrain, w, b)
    accuracy = (Y == Ytrain).sum()/(n_samples/2.)
    print('Training accuracy:', accuracy)
    assert accuracy > 0.9
    Y = predict(Xtest, w, b)
    accuracy = (Y == Ytest).sum()/(n_samples/2.)
    print('Test accuracy:', accuracy)
    assert accuracy > 0.9

