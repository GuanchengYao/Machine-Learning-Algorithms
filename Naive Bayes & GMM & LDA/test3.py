from problem3 import *
import numpy as np
import sys

'''
    Unit test 3:
    This file includes unit tests for problem3.py.
    You could test the correctness of your code by typing `nosetests -v test3.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 3 (30 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)


#-------------------------------------------------------------------------
def test_variational_inference():
    '''(6 points) variational_inference'''
    # a document with 5 words ( 0, 1, 2 denote the IDs of the words in the vocabulary)
    w = np.array([0, 0, 1, 1, 2])
    p=3
    k=2
    beta = np.ones((k,p))/p
    gamma, phi = variational_inference(w,beta,alpha=1., n_iter=0)
    assert np.allclose(gamma,[3.5,3.5],atol=1e-1)
    assert np.allclose(phi,np.ones((5,2))*.5,atol=1e-1)
       
    p=4
    k=3
    beta = np.ones((k,p))/p
    gamma, phi = variational_inference(w,beta,alpha=2., n_iter=0) 
    assert np.allclose(gamma,3.667*np.ones(3),atol=1e-2)
    assert np.allclose(phi,np.ones((5,3))*.33,atol=1e-2)

    p=3
    k=2
    beta = np.ones((k,p))/p
    gamma, phi = variational_inference(w,beta,alpha=1., n_iter=1)
    assert np.allclose(gamma,3.5*np.ones(2),atol=1e-2)
    assert np.allclose(phi,np.ones((5,2))*.5,atol=1e-2)


    beta=np.array([[.2,.8], # topic 0
                   [.3,.7], # topic 1
                  ])

    w = np.array([0, 0, 1, 1])
    gamma, phi = variational_inference(w,beta,alpha=1., n_iter=1)
    assert np.allclose(gamma,[2.86666667, 3.13333333],atol=1e-2)
    phi_true = [[ 0.4       , 0.6       ],
                [ 0.4       , 0.6       ],
                [ 0.53333333, 0.46666667],
                [ 0.53333333, 0.46666667]]
    assert np.allclose(phi,phi_true,atol=1e-2)

    gamma, phi = variational_inference(w,beta,alpha=1., n_iter=30)
    assert np.allclose(gamma,[ 2.44057924, 3.55942076],atol=1e-2)
    phi_true = [[ 0.29850263, 0.70149737],
                [ 0.29850263, 0.70149737],
                [ 0.42178699, 0.57821301],
                [ 0.42178699, 0.57821301]]
    assert np.allclose(phi,phi_true,atol=1e-2)


    w = np.array([0, 1, 1, 1])
    gamma, phi = variational_inference(w,beta,alpha=2., n_iter=10)
    assert np.allclose(gamma,[4,4],atol=1e-2)
    phi_true = [[ 0.4       , 0.6       ],
                [ 0.53333333, 0.46666667],
                [ 0.53333333, 0.46666667],
                [ 0.53333333, 0.46666667]]
    assert np.allclose(phi,phi_true,atol=1e-2)


#-------------------------------------------------------------------------
def test_E_step():
    '''(4 points) E_step'''

    # Document (m = 1, n = 4)
    W = np.array([
                    [0, 0, 1, 1] # 0-th document: word 0, word 1, ...
                 ])
    beta=np.array([[.2,.8], # topic 0 (word 0, word 1)
                   [.3,.7], # topic 1 (word 0, word 1)
                  ])

    gamma, phi = E_step(W,beta,alpha=1., n_iter=30)
    assert np.allclose(gamma.shape, [1,2])
    assert np.allclose(phi.shape, [1,4,2])
    assert np.allclose(gamma,[[2.44057924, 3.55942076]],atol=1e-2)
    phi_true =[[[ 0.29850263, 0.70149737],
                [ 0.29850263, 0.70149737],
                [ 0.42178699, 0.57821301],
                [ 0.42178699, 0.57821301]]]
    assert np.allclose(phi,phi_true,atol=1e-2)


    # Document (m = 2, n = 4)
    W = np.array([
        [0, 0, 1, 1], # 0-th document: word 0, word 1, ...
        [0, 1, 1, 1], # 1-st document: word 0, word 1, ...
    ])
    gamma, phi = E_step(W,beta,alpha=1., n_iter=30)
    assert np.allclose(gamma.shape, [2,2])
    assert np.allclose(phi.shape, [2,4,2])
    assert np.allclose(gamma,[[2.44057924, 3.55942076],[3,3]],atol=1e-2)
    phi_true =[# document 0
               [[ 0.29850263, 0.70149737],
                [ 0.29850263, 0.70149737],
                [ 0.42178699, 0.57821301],
                [ 0.42178699, 0.57821301]],
               # document 1
               [[ 0.4       , 0.6       ],
                [ 0.53333333, 0.46666667],
                [ 0.53333333, 0.46666667],
                [ 0.53333333, 0.46666667]]
              ]
    assert np.allclose(phi,phi_true,atol=1e-2)


#-------------------------------------------------------------------------
def test_udpate_beta():
    '''(4 points) update_beta'''
    # 2 Documents, each with 4 words (m = 2, n = 4)
    W = np.array([
        [0, 0, 1, 1], # 0-th document: word 0, word 1, ...
        [0, 0, 1, 1], # 1-st document: word 0, word 1, ...
    ]) 
    # 2 topics 
    phi = np.array(
    [
      [ # document 0
        [.1,.9],#  word 0 
        [.2,.8],#  word 1 
        [.3,.7], 
        [.4,.6] 
      ],
      [ # document 1
        [.1,.9],#  word 0 
        [.2,.8],#  word 1 
        [.3,.7], 
        [.4,.6] 
      ]
    ]) 
    beta = update_beta(W,phi,2)
    assert np.allclose(beta,[[.3,.7],[.567,.433]],atol=1e-2)

    W = np.array([
        [0, 0, 1, 1], # 0-th document: word 0, word 1, ...
        [0, 0, 2, 2], # 1-st document: word 0, word 1, ...
    ]) 

    beta = update_beta(W,phi,3)
    assert np.allclose(beta,[[.3,.35,.35],[.567,.217,.217]],atol=1e-2)


#-------------------------------------------------------------------------
def test_newtons_1d():
    '''(2 points) newtons_1d'''
    def f(x):
        return 6*x**5-5*x**4-4*x**3+3*x**2
    def df(x):
        return 30*x**4-20*x**3-12*x**2+6*x

    x = newtons_1d(f,df,0.)
    assert np.allclose(x,0.,atol=1e-2)
    
    x = newtons_1d(f,df,1.)
    assert np.allclose(x,1.,atol=1e-2)

    x = newtons_1d(f,df,.5,1e-5)
    assert np.allclose(x,0.628668078167,atol=1e-3)


    x = newtons_1d(f,df,100.,1e-5)
    print('x:',x)
    assert np.allclose(x,1.,atol=1e-3)


#-------------------------------------------------------------------------
def test_min_newtons_1d():
    '''(1 points) min_newtons_1d'''
    for _ in range(20):
        s = np.random.random()*100
        n = np.random.random()*100
        def f(x):
            return (x-s)**2 + n
        def df(x):
            return 2.*(x-s) 
        def d2f(x):
            return 2.
        x,v = min_newtons_1d(f,df,d2f,0.)
        assert np.allclose(x,s,atol=1e-1)
        assert np.allclose(v,n,atol=1e-1)
    
    for _ in range(20):
        s = np.random.random()*100
        n = np.random.random()*100
        def f(x):
            return (x-s)**4 + n
        def df(x):
            return 4.*(x-s)**3
        def d2f(x):
            return 12.*(x-s)**2
        x,v = min_newtons_1d(f,df,d2f,0.,tol=1e-5)
        
        assert np.allclose(v,n,atol=1e-1)
        assert np.allclose(x,s,atol=1e-1)



#-------------------------------------------------------------------------
def test_newtons():
    '''(2 points) newtons'''
    # when p = 1 (one dimensional case)
    def f(x):
        return np.array([6*x**5-5*x**4-4*x**3+3*x**2])
    def df(x):
        return np.array([[30*x**4-20*x**3-12*x**2+6*x]])
    x0 = np.array([0.])
    x = newtons(f,df,x0)
    assert np.allclose(x,x0,atol=1e-2)
    
    x0 = np.array([1.])
    x = newtons_1d(f,df,x0)
    assert np.allclose(x,x0,atol=1e-2)

    x0 = np.array([.5])
    x = newtons_1d(f,df,x0,1e-5)
    assert np.allclose(x,[0.628668078167],atol=1e-3)

    # when p = 2 (two dimensional case)
    def f(x):
        return np.array([(x[0]-2.)**2, (x[1]-3.)**2])
    def df(x):
        return np.array([[2*(x[0]-2),0.],[0.,2*(x[1]-3.)]])

    x0 = np.array([0.,0.])
    x = newtons(f,df,x0,1e-4)
    assert np.allclose(x,[2,3],atol=1e-2)

    def f(x):
        return np.array([-x[0]**3+x[1], x[0]**2+x[1]**2-1.])
    def df(x):
        return np.array([[-3.*x[0]**2,1.],[2.*x[0],2.*x[1]]])

    x0 = np.array([1.,2.])
    x = newtons(f,df,x0,1e-4)
    assert np.allclose(x,[0.826031357,0.563624162],atol=1e-2)

    # when p = 3 (three dimensional case)
    def f(x):
        return np.array([x[0]**2+x[1]**2+x[2]**2-3, 
                         x[0]**2+x[1]**2-x[2]-1,
                         x[0]**2+x[1]+x[2]-3
                        ])
    def df(x):
        return np.array([[2*x[0],   2*x[1], 2*x[2]],
                         [2*x[0],   2*x[1],     -1],
                         [2*x[0],       1.,     1.]
                        ])

    x0 = np.array([2.,2.,2.])
    x = newtons(f,df,x0,1e-4)
    assert np.allclose(x,[1,1,1],atol=1e-2)




#-------------------------------------------------------------------------
def test_newtons_linear():
    '''(1 points) newtons_linear'''
    # when p = 2 (two dimensional case)
    def f(x):
        return np.array([-x[0]**3+2*x[0]+2*x[1], x[1]**2+2*x[0]+2*x[1]])
    def df(x):
        return np.array([[-3.*x[0]**2+2., 2.        ],
                         [2.            , 2.*x[1]+2.]])
    def h(x):
        return np.array([-3.*x[0]**2, 2.*x[1]])
    def z(x):
        return 2.

    x0 = np.array([1.,1.])
    x1 = newtons(f,df,x0,1e-4)
    x = newtons_linear(f,h,z,x0,1e-4)
    assert np.allclose(x,[0.00554696, -0.00554729],atol=1e-2) #comma?
    assert np.allclose(x,x1,atol=1e-2)

    # test scalability (linear-time)
    def f(x):
        return np.ones(10000)*(x**2)
    def h(x):
        return np.ones(10000)*2.*x
    def z(x):
        return 0.01 
    
    x0= np.ones(10000)*1000
    x = newtons_linear(f,h,z,x0,1e-2)
    assert np.allclose(x,np.zeros(10000),atol=1e-1)



#-------------------------------------------------------------------------
def test_compute_df():
    '''(2 points) compute_df'''
    # 2 topics
    a = 1.
    gamma =  np.array([# document 0
                       [ 0.3, 0.7],
                       # document 1
                       [ 0.4, 0.6]])
    df = compute_df(a,gamma)
    assert np.allclose(df,-2.90947744+0.39378856,atol=1e-2)

    a = 2.
    gamma =  np.array([# document 0
                       [ 0.3, 0.7],
                       # document 1
                       [ 0.4, 0.6]])
    df = compute_df(a,gamma)
    assert np.allclose(df,-3.18235554,atol=1e-2)


#-------------------------------------------------------------------------
def test_compute_d2f():
    '''(2 points) compute_d2f'''
    # 2 topics
    a = 1.
    d2f = compute_d2f(a,2,2)
    assert np.allclose(d2f,-1.42026373,atol=1e-2)

    a = 2.
    d2f = compute_d2f(a,2,2)
    assert np.allclose(d2f,-0.3091526,atol=1e-2)


#-------------------------------------------------------------------------
def test_udpate_alpha():
    '''(3 points) update_alpha'''
    a = .1 
    gamma =  np.array([# document 0
                       [ 0.5, 0.5],
                       # document 1
                       [ 0.5, 0.5]])
    a_new = update_alpha(a,gamma)
    assert np.allclose(a_new,.5,atol=1e-2)
    gamma =  np.array([# document 0
                       [ 0.2, 0.8],
                       # document 1
                       [ 0.8, 0.2]])
    a_new = update_alpha(a,gamma)
    assert np.allclose(a_new,0.216,atol=1e-2)

    gamma =  np.array([# document 0
                       [ 0.1, 0.9],
                       # document 1
                       [ 0.9, 0.1]])
    a_new = update_alpha(a,gamma)
    assert np.allclose(a_new,0.10,atol=1e-2)


    a = 10. 
    gamma =  np.array([# document 0
                       [ 0.5, 0.5],
                       # document 1
                       [ 0.5, 0.5]])
    a_new = update_alpha(a,gamma)
    assert np.allclose(a_new,.5,atol=1e-2)


#-------------------------------------------------------------------------
def test_LDA():
    '''(3 points) LDA'''
    # 2 Documents, each with 4 words (m = 2, n = 4)
    W = np.array([
        [0, 0, 1, 1], # 0-th document: word 0, word 1, ...
        [0, 1, 1, 1], # 1-st document: word 0, word 1, ...
    ]) 
    alpha,beta, gamma, phi = LDA(W, k=2, p=2,n_iter_em=1)
    
    
    beta_true =  [[ 0.31, 0.68],
                  [ 0.43, 0.56]]
    gamma_true = [[ 0.1, 4.1],
                  [ 4.1, 0.1]]
    phi_true = [[[ 0., 1.],
                 [ 0., 1.],
                 [ 0., 1.],
                 [ 0., 1.]],
                [[ 1., 0.],
                 [ 1., 0.],
                 [ 1., 0.],
                 [ 1., 0.]]]
    assert np.allclose(alpha,0.13,atol=1e-1)
    assert np.allclose(beta,beta_true,atol=1e-1)
    assert np.allclose(gamma,gamma_true,atol=1e-1)
    assert np.allclose(phi,phi_true,atol=1e-1)

    # 2 Documents, each with 4 words (m = 2, n = 4,p=3)
    W = np.array([
        [0, 0, 1, 2], # 0-th document: word 0, word 1, ...
        [0, 1, 2, 1], # 1-st document: word 0, word 1, ...
    ]) 
    alph,beta, gamma, phi = LDA(W, k=2, p=3,n_iter_em=3)


    beta_true =  [[ .25, .5 , .25],
                  [ .5 , .25, .25]]
    gamma_true = [[ .2  , 4.19],
                  [ 4.19, .2  ]]
    phi_true = [[[ 0., 1.],
                 [ 0., 1.],
                 [ 0., 1.],
                 [ 0., 1.]],
                [[ 1., 0.],
                 [ 1., 0.],
                 [ 1., 0.],
                 [ 1., 0.]]]
    assert np.allclose(alpha,0.13,atol=1e-1)
    assert np.allclose(beta,beta_true,atol=1e-1)
    assert np.allclose(gamma,gamma_true,atol=3e-1)
    assert np.allclose(phi,phi_true,atol=1e-1)

    
