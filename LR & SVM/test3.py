from problem3 import *
import numpy as np
import sys
from sklearn.datasets import make_classification
'''
    Unit test 3:
    This file includes unit tests for problem3.py.
    You could test the correctness of your code by typing `nosetests -v test3.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 3 (55 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)


#-------------------------------------------------------------------------
def test_linear_kernel():
    ''' (2 points) linear kernel'''

    X = np.mat([[0.,1.],
                [1.,0.],
                [1.,1.]])
    K = linear_kernel(X,X)
    assert type(K) == np.matrixlib.defmatrix.matrix 
    assert K.shape == (3,3)
    assert np.allclose(K, np.mat('1,0,1;0,1,1;1,1,2'), atol = 1e-3) 


    X = np.mat([[1.,2.],
                [3.,4.]])
    K = linear_kernel(X,X)
    assert K.shape == (2,2)
    assert np.allclose(K, np.mat('5,11;11,25'), atol = 1e-3) 


    X1 = np.mat([[0.,1.],
                 [0.,1.],
                 [0.,1.]])
    X2 = np.mat([[1.,0.],
                 [1.,1.]])
    K = linear_kernel(X1,X2)
    assert K.shape == (3,2)
    assert np.allclose(K, np.mat('0,1;0,1;0,1'), atol = 1e-3) 

#-------------------------------------------------------------------------
def test_polynomial_kernel():
    ''' (2 points) polynoimial kernel'''

    X = np.mat([[0.,1.],
                [1.,0.],
                [1.,1.]])
    K = polynomial_kernel(X,X,d=1)
    assert type(K) == np.matrixlib.defmatrix.matrix 
    assert K.shape == (3,3)
    assert np.allclose(K, np.mat('1,0,1;0,1,1;1,1,2')+1, atol = 1e-3) 
    assert np.allclose(K, np.mat('2,1,2;1,2,2;2,2,3'), atol = 1e-3) 


    X = np.mat([[1.,2.],
                [3.,4.]])
    K = polynomial_kernel(X,X,d=1)
    assert K.shape == (2,2)
    assert np.allclose(K, np.mat('5,11;11,25')+1, atol = 1e-3) 

    X = np.mat([[0.,1.],
                [1.,0.],
                [1.,1.]])
    K = polynomial_kernel(X,X,d=2)
    assert K.shape == (3,3)
    assert np.allclose(K, np.mat('4,1,4;1,4,4;4,4,9'), atol = 1e-3) 
    K = polynomial_kernel(X,X,d=3)
    assert np.allclose(K, np.mat('8,1,8;1,8,8;8,8,27'), atol = 1e-3) 


    X1 = np.mat([[0.,1.],
                 [0.,1.],
                 [0.,1.]])
    X2 = np.mat([[1.,0.],
                 [1.,1.]])
    K = polynomial_kernel(X1,X2,d=1)
    assert K.shape == (3,2)
    assert np.allclose(K, np.mat('0,1;0,1;0,1')+1, atol = 1e-3) 


#-------------------------------------------------------------------------
def test_gaussian_kernel():
    ''' (2 points) gaussian kernel'''

    X = np.mat([[1.,1.],
                [1.,1.]])
    K = gaussian_kernel(X,X,1)
    assert K.shape == (2,2)
    assert np.allclose(K, np.mat('1,1;1,1'), atol = 1e-3) 

    X = np.mat([[0.,1.],
                [1.,0.]])
    K = gaussian_kernel(X,X,1.)
    assert type(K) == np.matrixlib.defmatrix.matrix 
    assert K.shape == (2,2)
    assert np.allclose(K, np.mat('1,.367879;.367879,1'), atol = 1e-3) 

    X = np.mat([[0.,100.],
                [100.,0.]])
    K = gaussian_kernel(X,X,1.)
    assert K.shape == (2,2)
    assert np.allclose(K, np.mat('1,0;0,1'), atol = 1e-3) 

    X = np.mat([[0.,1.],
                [1.,0.]])
    K = gaussian_kernel(X,X,0.1)
    assert type(K) == np.matrixlib.defmatrix.matrix 
    assert K.shape == (2,2)
    assert np.allclose(K, np.mat('1,0;0,1'), atol = 1e-3) 


    X = np.mat([[1.,1.],
                [1.,1.],
                [1.,1.]])
    K = gaussian_kernel(X,X,1.)
    assert K.shape == (3,3)
    assert np.allclose(K, np.asmatrix(np.ones((3,3))), atol = 1e-3) 

    X1 = np.mat([[0.,1.],
                 [0.,1.],
                 [0.,1.]])
    X2 = np.mat([[0.,1.],
                 [1.,1.]])
    K = gaussian_kernel(X1,X2,0.1)
    assert K.shape == (3,2)
    assert np.allclose(K, np.mat('1,0;1,0;1,0'), atol = 1e-3) 



#-------------------------------------------------------------------------
def test_predict():
    ''' (5 points) predict'''

    K = np.mat([[1.,1.],
                [1.,1.]])
    a = np.mat('1.;1.')
    y = np.mat('1.;1.')
    b = 0. 
    y_test = predict(K,a,y,b)
    assert type(y_test) == np.matrixlib.defmatrix.matrix 
    assert y_test.shape == (2,1)
    assert np.allclose(y_test, np.mat('1;1'), atol = 1e-3) 

    K = np.mat([[1.,0.],
                [0.,1.]])
    a = np.mat('1.;1.')
    y = np.mat('1.;-1.')
    b = 0. 
    y_test = predict(K,a,y,b)
    assert type(y_test) == np.matrixlib.defmatrix.matrix 
    assert y_test.shape == (2,1)
    assert np.allclose(y_test, np.mat('1;-1'), atol = 1e-3) 

    K = np.mat([[1.,0.],
                [1.,1.]])
    a = np.mat('1.;1.')
    y = np.mat('1.;-1.')
    b = .1 
    y_test = predict(K,a,y,b)
    assert type(y_test) == np.matrixlib.defmatrix.matrix 
    assert y_test.shape == (2,1)
    assert np.allclose(y_test, np.mat('1;1'), atol = 1e-3) 

    K = np.mat([[1.,0.],
                [1.,1.]])
    a = np.mat('1.;2.')
    y = np.mat('1.;-1.')
    b = .1 
    y_test = predict(K,a,y,b)
    assert type(y_test) == np.matrixlib.defmatrix.matrix 
    assert y_test.shape == (2,1)
    assert np.allclose(y_test, np.mat('1;-1'), atol = 1e-3) 


    K = np.mat([[1.,1.],
                [2.,3.],
                [1.,1.]])
    a = np.mat('1.;1.')
    y = np.mat('1.;1.')
    b = 0. 
    y_test = predict(K,a,y,b)
    assert type(y_test) == np.matrixlib.defmatrix.matrix 
    assert y_test.shape == (3,1)
    assert np.allclose(y_test, np.mat('1;1;1'), atol = 1e-3) 


#-------------------------------------------------------------------------
def test_compute_HL():
    ''' (5 points) compute_HL'''
    ai = 0. 
    yi = 1.
    aj = 0. 
    yj = 1.
    H,L = compute_HL(ai,yi,aj,yj,C=1.) 
    assert H == 0.
    assert L == 0. 

    H,L = compute_HL(0.,1.,0.,-1.,C=1.) 
    assert H == 1.
    assert L == 0. 

    H,L = compute_HL(0.,-1.,0.,1.,C=1.) 
    assert H == 1.
    assert L == 0. 

    H,L = compute_HL(0.,-1.,0.,1.,C=10.) 
    assert H == 10.
    assert L == 0. 

    H,L = compute_HL(0.,-1.,2.,1.,C=10.) 
    assert H == 8.
    assert L == 0. 

    H,L = compute_HL(3.,-1.,2.,1.,C=8.) 
    assert H == 8.
    assert L == 1. 

    H,L = compute_HL(3.,-1.,2.,-1.,C=8.) 
    assert H == 5.
    assert L == 0. 

#-------------------------------------------------------------------------
def test_compute_E():
    ''' (5 points) compute_E'''

    Ki = np.mat('1.,1.')
    a = np.mat('1.;1.')
    y = np.mat('1.;1.')
    b = 0. 
    E = compute_E(Ki,a,y,b,i=0)
    assert type(E) == float
    assert E == 1.

    Ki = np.mat('1.,.5')
    E = compute_E(Ki,a,y,b,i=0)
    assert E == .5

    a = np.mat('1.;2.')
    E = compute_E(Ki,a,y,b,i=0)
    assert E == 1.

    b = 1.
    E = compute_E(Ki,a,y,b,i=0)
    assert E == 2.

    y = np.mat('1.;-1.')
    E = compute_E(Ki,a,y,b,i=0)
    assert E == 0. 
    


#-------------------------------------------------------------------------
def test_compute_eta():
    ''' (5 points) compute_eta'''
    e = compute_eta(1.,1.,1.5)
    assert type(e) == float
    assert e == 1.

    e = compute_eta(1.,1.,2.)
    assert e == 2.

    e = compute_eta(.5,1.,2.)
    assert e == 2.5

    e = compute_eta(.5,.7,2.)
    assert e == 2.8

#-------------------------------------------------------------------------
def test_update_ai():
    ''' (5 points) update_ai'''
    an = update_ai(1.,1.,1.,4.,1.,10,0.)
    assert type(an) == float
    assert an == 4.

    an = update_ai(1.,2.,1.,4.,1.,10,0.)
    assert an == 3.

    an = update_ai(0.,2.,1.,4.,1.,10,0.)
    assert an == 2.

    an = update_ai(0.,2.,2.,4.,1.,10,0.)
    assert an == 3.

    an = update_ai(0.,2.,1.,4.,1.,10.,3.)
    assert an == 3.

    an = update_ai(0.,2.,1.,4.,-1.,10.,3.)
    assert an == 6.

    an = update_ai(0.,2.,1.,4.,-1.,5.,3.)
    assert an == 5.

    an = update_ai(0.,2.,0.,4.,-1.,5.,3.)
    assert an == 4. # if eta =0, ai is not changed. (when i= j)


#-------------------------------------------------------------------------
def test_update_aj():
    ''' (5 points) update_aj'''
    an = update_aj(1.,1.,2.,1.,1.)
    assert type(an) == float
    assert an == 0.

    an = update_aj(1.,1.,2.,1.,-1.)
    assert an == 2.

    an = update_aj(1.,1.,2.,-1.,1.)
    assert an == 2.

    an = update_aj(1.,0.,2.,-1.,1.)
    assert an == 3.

    an = update_aj(1.,0.,3.,-1.,1.)
    assert an == 4.

    an = update_aj(2.,0.,3.,-1.,1.)
    assert an == 5.


#-------------------------------------------------------------------------
def test_update_b():
    ''' (5 point) update_b'''
    b = update_b(1.,2.,3.,1.,1.,1.,1.,1.,1.,1.,1.,1.,3.)
    assert type(b) == float
    assert b == -3.

    b = update_b(1.,3.,2.,1.,1.,1.,1.,1.,1.,1.,1.,1.,3.)
    assert b == -3.

    b = update_b(1.,3.,2.,1.,1.,1.,1.,1.,1.,1.,1.,1.,4.)
    assert b == -3.

    b = update_b(2.,3.,2.,1.,1.,1.,1.,1.,1.,1.,1.,1.,4.)
    assert b == -2.

    b = update_b(2.,4.,2.,1.,1.,1.,1.,1.,1.,1.,1.,1.,5.)
    assert b == -3.

    b = update_b(0.,3.,0.,1.,0.,-1.,1.,0.,1.,1.,1.,1.,5.)
    assert b == 2.

    b = update_b(0.,3.,0.,2.,0.,-1.,1.,0.,1.,1.,1.,1.,5.)
    assert b == 1.

    b = update_b(0.,3.,0.,2.,0.,1.,1.,0.,1.,1.,1.,1.,5.)
    assert b == -1.

    b = update_b(0.,3.,0.,2.,0.,1.,1.,0.,1.,2.,1.,1.,5.)
    assert b == -2.

    b = update_b(0.,3.,0.,2.,0.,1.,1.,1.,1.,2.,1.,1.,5.)
    assert b == -3.

    b = update_b(1.,3.,0.,2.,0.,1.,1.,1.,1.,2.,1.,1.,5.)
    assert b == -2.

    b = update_b(1.,3.,1.,2.,0.,1.,1.,1.,1.,2.,1.,1.,5.)
    assert b == -3.

    b = update_b(1.,3.,2.,2.,1.,1.,1.,1.,1.,2.,1.,1.,5.)
    assert b == -3.

    b = update_b(1.,3.,2.,2.,1.,1.,-1.,1.,1.,2.,1.,1.,5.)
    assert b == -1.

    b = update_b(1.,3.,2.,2.,1.,1.,-1.,1.,1.,2.,1.,2.,5.)
    assert b == 0.

    b = update_b(0.,0.,2.,0.,1.,1.,1.,1.,0.,1.,1.,0.,5.)
    assert b == -1.

    b = update_b(0.,0.,3.,0.,1.,1.,1.,1.,0.,1.,1.,0.,5.)
    assert b == -2.

    b = update_b(0.,0.,3.,0.,1.,1.,1.,1.,1.,1.,1.,0.,5.)
    assert b == -3.

    b = update_b(0.,0.,3.,0.,1.,1.,-1.,1.,1.,1.,1.,0.,5.)
    assert b == 1.

    b = update_b(0.,5.,3.,0.,1.,1.,-1.,1.,1.,1.,1.,0.,5.)
    assert b == 1.

    b = update_b(0.,5.,3.,4.,1.,1.,-1.,1.,1.,1.,1.,1.,5.)
    assert b == 0.

    b = update_b(0.,5.,3.,4.,1.,1.,-1.,1.,1.,1.,1.,2.,5.)
    assert b == -1.

    b = update_b(0.,5.,3.,3.,1.,1.,-1.,1.,1.,1.,1.,2.,5.)
    assert b == -3.

    b = update_b(0.,5.,3.,3.,1.,-1.,-1.,1.,1.,1.,1.,2.,5.)
    assert b == 5.

    b = update_b(1.,5.,3.,3.,1.,-1.,-1.,1.,1.,1.,1.,2.,5.)
    assert b == 6.

    b = update_b(0.,5.,5.,4.,4.,1.,1.,1.,1.,0.,0.,0.,5.)
    assert b == -1.

    b = update_b(0.,5.,5.,4.,4.,1.,1.,2.,2.,0.,0.,0.,5.)
    assert b == -2.

    b = update_b(0.,5.,5.,4.,4.,1.,1.,1.,2.,0.,0.,0.,5.)
    assert b == -1.5

    b = update_b(0.,5.,5.,4.,4.,1.,1.,0.,0.,0.,0.,1.,5.)
    assert b == -1.

    b = update_b(0.,5.,5.,3.,4.,1.,1.,0.,0.,0.,0.,1.,5.)
    assert b == -1.5


#-------------------------------------------------------------------------
def test_train():
    ''' (5 points) train'''
    # linear kernel x1 = 0, x2 = 1
    K = np.mat([[0.,0.],
                [0.,1.]])
    y = np.mat('-1.;1.')
    C = 1000.
    a,b = train(K,y,C,10)
    assert type(a) == np.matrixlib.defmatrix.matrix 
    assert a.shape == (2,1)
    assert np.allclose(a, np.mat('2;2'),atol = 1e-3)
    assert np.allclose(b , -1, atol=1e-3) 

    a,b = train(K,y,C,2)
    assert np.allclose(a, np.mat('2;2'),atol = 1e-3)
    assert np.allclose(b , -1, atol=1e-3) 

    # linear kernel x1 = 0, x2 = 2
    K = np.mat([[0.,0.],
                [0.,4.]])
    a,b = train(K,y,C)
    assert np.allclose(a, np.mat('.5;.5'),atol = 1e-3)
    assert np.allclose(b , -1, atol=1e-3) 

    # linear kernel x1 = -1, x2 = 1
    K = np.mat([[1.,-1],
                [-1.,1.]])
    a,b = train(K,y,C)
    assert np.allclose(a, np.mat('.5;.5'),atol = 1e-3)
    assert np.allclose(b , 0, atol=1e-3) 


    # linear kernel x1 = -1, x2 = 1, x3 = 2
    K = np.mat([[ 1.,-1.,-2.],
                [-1., 1., 2.],
                [-2., 2., 4.]])
    y = np.mat('-1.;1.;1.')
    a,b = train(K,y,C)
    assert np.allclose(a, np.mat('.5;.5;0.'),atol = 1e-3)
    assert np.allclose(b , 0, atol=1e-3) 

    # linear kernel x1 = -1, x2 = 1, x3 = 3
    K = np.mat([[ 1.,-1.,-3.],
                [-1., 1., 3.],
                [-3., 3., 9.]])
    y = np.mat('-1.;1.;1.')
    a,b = train(K,y,C)
    assert np.allclose(a, np.mat('.5;.5;0.'),atol = 1e-3)
    assert np.allclose(b , 0, atol=1e-3) 

    # linear kernel x1 = -1, x2 = 1, x3 = 1.1 
    K = np.mat([[ 1.,-1.,-1.1],
                [-1., 1., 1.1],
                [-1.1, 1.1, 1.21]])
    y = np.mat('-1.;1.;1.')
    a,b = train(K,y,C)
    assert np.allclose(a, np.mat('.5;.5;0.'),atol = 1e-3)
    assert np.allclose(b , 0, atol=1e-3) 

    # linear kernel x1 = -2, x2 = -1, x3 = 1, x4 = 2
    K = np.mat([[ 4., 2.,-2.,-4.],
                [ 2., 1.,-1.,-2.],
                [-2.,-1., 1., 2.],
                [-4.,-2., 2., 4.]])
    y = np.mat('-1.;-1;1.;1.')
    a,b = train(K,y,C)
    assert np.allclose(a, np.mat('0.;.5;.5;0.'),atol = 1e-3)
    assert np.allclose(b , 0, atol=1e-3) 


    # linear kernel x1 = -2, x2 = -1, x3 = 1, x4 = 2
    K = np.mat([[ 4., 2.,-2.,-4.],
                [ 2., 1.,-1.,-2.],
                [-2.,-1., 1., 2.],
                [-4.,-2., 2., 4.]])
    y = np.mat('-1.;-1;1.;1.')
    a,b = train(K,y,0.2)
    assert np.allclose(a, np.mat('0.025;.2;.2;0.025'),atol = 1e-3)
    assert np.allclose(b , 0, atol=1e-3) 

    # linear kernel x1 = (-1,0), x2 = (0,-1), x3 = (1,0), x4 = (0,1)
    K = np.mat([[ 1., 0.,-1., 0.],
                [ 0., 1., 0.,-1.],
                [-1., 0., 1., 0.],
                [ 0.,-1., 0., 1.]])
    y = np.mat('-1.;-1;1.;1.')
    a,b = train(K,y,1)
    assert np.allclose(a, np.mat('0.5;.5;.5;0.5'),atol = 1e-2)
    assert np.allclose(b , 0, atol=1e-2) 

    # linear kernel x1 = (-1.1,0), x2 = (0,-1), x3 = (1,0), x4 = (0,1.1)
    K = np.mat([[1.21, 0.,-1.1, 0.],
                [  0., 1.,  0.,-1.1],
                [-1.1, 0.,  1., 0.],
                [  0.,-1.1,  0., 1.21]])
    y = np.mat('-1.;-1;1.;1.')
    a,b = train(K,y,10)
    assert np.allclose(a, np.mat('0;1;1;0'),atol =.01)
    assert np.allclose(b , 0, atol=1e-2) 


#-------------------------------------------------------------------------
def test_svm_linear():
    '''(3 point) test svm (linear kernel)'''
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

    K1 = linear_kernel(Xtrain,Xtrain)
    K2 = linear_kernel(Xtest,Xtrain)
    a,b = train(K1, Ytrain, C=1., n_epoch=1)
    n_SV = (a>0).sum() # number of support vectors
    assert n_SV < 25 
    Y = predict(K1, a, Ytrain,b)
    accuracy = (Y == Ytrain).sum()/(n_samples/2.)
    print('Training accuracy:', accuracy)
    assert accuracy > 0.85
    Y = predict(K2, a, Ytrain,b)
    accuracy = (Y == Ytest).sum()/(n_samples/2.)
    print('Test accuracy:', accuracy)
    assert accuracy > 0.85

#-------------------------------------------------------------------------
def test_svm_poly():
    '''(3 point) test svm (polynomial kernel)'''
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

    K1 = polynomial_kernel(Xtrain,Xtrain,2)
    K2 = polynomial_kernel(Xtest,Xtrain,2)
    a,b = train(K1, Ytrain, C=1., n_epoch=1)
    n_SV = (a>0).sum() # number of support vectors
    assert n_SV < 50 
    Y = predict(K1, a, Ytrain,b)
    accuracy = (Y == Ytrain).sum()/(n_samples/2.)
    print('Training accuracy:', accuracy)
    assert accuracy > 0.9
    Y = predict(K2, a, Ytrain,b)
    accuracy = (Y == Ytest).sum()/(n_samples/2.)
    print('Test accuracy:', accuracy)
    assert accuracy > 0.9


#-------------------------------------------------------------------------
def test_svm_RBF():
    '''(3 point) test svm (gaussian kernel)'''
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

    K1 = gaussian_kernel(Xtrain,Xtrain)
    K2 = gaussian_kernel(Xtest,Xtrain)
    a,b = train(K1, Ytrain, C=1., n_epoch=1)
    n_SV = (a>0).sum() # number of support vectors
    assert n_SV < 50 
    Y = predict(K1, a, Ytrain,b)
    accuracy = (Y == Ytrain).sum()/(n_samples/2.)
    print('Training accuracy:', accuracy)
    assert accuracy > 0.9
    Y = predict(K2, a, Ytrain,b)
    accuracy = (Y == Ytest).sum()/(n_samples/2.)
    print('Test accuracy:', accuracy)
    assert accuracy > 0.9

