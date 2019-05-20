from problem2 import *
import sys
import numpy as np
'''
    Unit test 2:
    This file includes unit tests for problem1.py.
    You could test the correctness of your code by typing `nosetests -v test2.py` in the terminal.
'''
#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 2 (25 points in total)--------------'''
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)


#-------------------------------------------------------------------------
def test_cutting_points():
    ''' (3 points) cutting_points'''

    y = np.array([1.,1.,0.,0.])
    x = np.array([1.,2.,3.,4.])
    cp = DT.cutting_points(x,y)
    assert type(cp) == np.ndarray
    assert cp.shape == (1,)
    assert cp[0]==2.5

    y = np.array([1.,0.,1.,0.])
    x = np.array([2.,4.,1.,3.])
    cp = DT.cutting_points(x,y)
    assert type(cp) == np.ndarray
    assert cp.shape == (1,)
    assert cp[0]==2.5


    y = np.array([0.,0.,1.,0.])
    x = np.array([1.,2.,3.,4.])
    cp = DT.cutting_points(x,y)
    assert type(cp) == np.ndarray
    assert cp.shape == (2,)
    assert cp[0]==2.5
    assert cp[1]==3.5

    y = np.array([0.,0.,1.,0., 1., 0.,0.])
    x = np.array([1.,2.,3.,4., 5., 6.,7.])
    cp = DT.cutting_points(x,y)
    assert type(cp) == np.ndarray
    assert cp.shape == (4,)
    assert cp[0]==2.5
    assert cp[1]==3.5
    assert cp[2]==4.5
    assert cp[3]==5.5


    y = np.array([0.,0.,1.,0., 0.])
    x = np.array([1.,2.,3.,3., 4.])
    cp = DT.cutting_points(x,y)
    assert type(cp) == np.ndarray
    assert cp.shape == (2,)
    assert cp[0]==2.5
    assert cp[1]==3.5

    y = np.array([0.,0.,1.,0., 1.])
    x = np.array([1.,2.,3.,3., 4.])
    cp = DT.cutting_points(x,y)
    assert type(cp) == np.ndarray
    assert cp.shape == (2,)
    assert cp[0]==2.5
    assert cp[1]==3.5

    y = np.array([0.,0.,1.,1., 1.])
    x = np.array([1.,2.,3.,3., 4.])
    cp = DT.cutting_points(x,y)
    assert type(cp) == np.ndarray
    assert cp.shape == (1,)
    assert cp[0]==2.5


#-------------------------------------------------------------------------
def test_best_threshold():
    ''' (3 points) best_threshold'''
    y = np.array([1.,1.,0.,0.])
    x = np.array([1.,2.,3.,4.])
    th,g = DT.best_threshold(x,y)
    assert th == 2.5 
    assert g == 1. 


    y = np.array([0.,0.,1.,0., 1., 1.,1.])
    x = np.array([1.,2.,3.,4., 5., 6.,7.])
    th,g = DT.best_threshold(x,y)
    assert th == 4.5 
    assert np.allclose(g,0.52164063634,atol = 1e-3)

    y = np.array([0.,0.,1.])
    x = np.array([1.,1.,1.]) # if all values are the same
    th, g = DT.best_threshold(x,y)
    assert th ==  - float('inf') # manulally set threshold as -inf 
    assert g == -1. # manually set gain as -1 to avoid selecting this attribute.


#-------------------------------------------------------------------------
def test_best_attribute():
    ''' (3 points) best_attribute'''
    d = DT()
    X = np.array([[0.,0.],
                  [2.,2.],
                  [1.,2.],
                  [3.,3.]])
    Y = np.array(['good','bad'])
    i,th = d.best_attribute(X,Y)
    assert i == 2 
    assert th == 1.5



    X = np.array([[0.,0.,1.,1.],
                  [2.,2.,1.,1.],
                  [1.,2.,1.,1.],
                  [0.,3.,1.,2.]])
    Y = np.array(['good','bad','good','bad'])
    i,th = d.best_attribute(X,Y)
    assert i == 3 
    assert th == 1.5


    X = np.array([[0.,0.,0.,0.],
                  [2.,2.,1.,1.],
                  [0.,0.,1.,1.]])
    Y = np.array(['good','bad','good','bad'])
    i,th = d.best_attribute(X,Y)
    assert i == 1 or i==2
    assert th == 1.5 or th == 0.5





#-------------------------------------------------------------------------
def test_split():
    ''' (2 points) split'''

    X = np.array([[1.,2.,3.,4.],
                  [2.,4.,6.,8.],
                  [1.,1.,2.,2.]])
    Y = np.array(['good','bad','okay','perfect'])
    C1,C2 = DT.split(X,Y,1,5.)

    assert isinstance(C1, Node)
    assert isinstance(C2, Node)

    assert C1.X.shape == (3,2)
    assert C1.Y.shape == (2,)
    assert C1.i == None 
    assert C1.C1 == None 
    assert C1.C2 == None 
    assert C1.isleaf == False 
    assert C1.p == None 
    assert C2.X.shape == (3,2)
    assert C2.Y.shape == (2,)
    assert C2.i == None 
    assert C2.C1 == None 
    assert C2.C2 == None 
    assert C2.isleaf == False 
    assert C2.p == None 
    assert np.allclose(C1.X, [[1.,2.],[2.,4.],[1.,1.]])
    assert np.allclose(C2.X, [[3.,4.],[6.,8.],[2.,2.]])
    assert C1.Y[0] == 'good'
    assert C1.Y[1] == 'bad'
    assert C2.Y[0] == 'okay'
    assert C2.Y[1] == 'perfect'


    C1,C2 = DT.split(X,Y,0,2.5)
    assert C1.X.shape == (3,2)
    assert C1.Y.shape == (2,)
    assert C1.i == None 
    assert C1.C1 == None 
    assert C1.C2 == None 
    assert C1.isleaf == False 
    assert C1.p == None 
    assert C2.X.shape == (3,2)
    assert C2.Y.shape == (2,)
    assert C2.i == None 
    assert C2.C1 == None 
    assert C2.C2 == None 
    assert C2.isleaf == False 
    assert C2.p == None 
    assert np.allclose(C1.X, [[1.,2.],[2.,4.],[1.,1.]])
    assert np.allclose(C2.X, [[3.,4.],[6.,8.],[2.,2.]])
    assert C1.Y[0] == 'good'
    assert C1.Y[1] == 'bad'
    assert C2.Y[0] == 'okay'
    assert C2.Y[1] == 'perfect'


    C1,C2 = DT.split(X,Y,0,3.5)
    assert C1.X.shape == (3,3)
    assert C1.Y.shape == (3,)
    assert C2.X.shape == (3,1)
    assert C2.Y.shape == (1,)

#-------------------------------------------------------------------------
def test_build_tree():
    ''' (3 points) build_tree'''
    d = DT()
    X = np.array([[1.],
                  [2.],
                  [3.]])
    Y = np.array(['bad'])
    t = Node(X=X, Y=Y) # root node
    
    # build tree
    d.build_tree(t)

    assert t.isleaf == True
    assert t.p == 'bad' 
    assert t.C1 == None
    assert t.C2 == None
    assert t.i == None


    #------------
    X = np.random.random((5,4))
    Y = np.array(['good','good','good','good'])
    t = Node(X=X, Y=Y) # root node
    
    # build tree
    d.build_tree(t)

    assert t.isleaf == True
    assert t.p == 'good' 
    assert t.C1 == None
    assert t.C2 == None
    assert t.i == None

    #------------
    X = np.ones((5,4))
    Y = np.array(['good','bad','bad','bad'])
    t = Node(X=X, Y=Y) # root node
    
    # build tree
    d.build_tree(t)

    assert t.isleaf == True
    assert t.p == 'bad' 
    assert t.C1 == None
    assert t.C2 == None
    assert t.i == None


    X = np.array([[1.,2.,3.,4.],
                  [2.,4.,6.,8.],
                  [1.,2.,1.,1.]])
    Y = np.array(['good','bad','good','good'])
    t = Node(X=X, Y=Y) # root node
    
    # build tree
    d.build_tree(t)

    assert t.i == 2 
    assert t.isleaf == False 
    assert t.p == 'good' 
    assert isinstance(t.C1, Node)
    assert isinstance(t.C2, Node)

    c1 = t.C1
    c2 = t.C2

    assert c1.isleaf == True
    assert c1.p == 'good' 
    assert c2.isleaf == True
    assert c2.p == 'bad' 


    X = np.array([[1.,1.,1.,1.],
                  [1.,2.,3.,1.]])
    Y = np.array(['okay','bad','good','okay'])
    t = Node(X=X, Y=Y) # root node
    
    # build tree
    d.build_tree(t)

    assert t.i == 1 
    assert t.th == 1.5
    assert t.isleaf == False 
    assert t.p == 'okay' 
    assert isinstance(t.C1, Node)
    assert isinstance(t.C2, Node)

    c1 = t.C1
    c2 = t.C2

    assert c1.isleaf == True
    assert c1.p == 'okay' 
    assert c2.isleaf == False 
    assert c2.i == 1 
    assert c2.th == 2.5 
    assert c2.p == 'bad' or c2.p == 'good' 
    assert isinstance(c2.C1, Node)
    assert isinstance(c2.C2, Node)
    
    c2c1 = c2.C1
    c2c2 = c2.C2

    assert c2c1.isleaf == True
    assert c2c1.p == 'bad' 
    assert c2c2.isleaf == True
    assert c2c2.p == 'good' 


    #--------
    # test overwritting (we will need to overwrite DT in problem 4)
    class Test(DT):
        def best_attribute(self,X,Y):
            if X.shape[1]==4:
                return 0, 1.5
            else:
                return super(Test,self).best_attribute(X,Y)

    d = Test()
    X = np.array([[1.,1.,1.,2.],
                  [1.,1.,3.,3.]])
    Y = np.array(['good','bad','good','bad'])
    t = Node(X=X, Y=Y) # root node
    d.build_tree(t)

    assert t.i == 0 
    assert t.th == 1.5
    assert t.p == 'bad' or t.p == 'good' 
    assert t.isleaf == False 
    assert isinstance(t.C1, Node)
    assert isinstance(t.C2, Node)

    c1 = t.C1
    c2 = t.C2
    assert c1.isleaf == False 
    assert c1.p == 'good' 
    assert c1.i == 1
    assert c1.th == 2. 
    assert isinstance(c1.C1, Node)
    assert isinstance(c1.C2, Node)
    assert c2.isleaf == True 
    assert c2.p == 'bad' 

#-------------------------------------------------------------------------
def test_inference():
    ''' (2 points) inference'''

    t = Node(None,None) 
    t.isleaf = True
    t.p = 'good job' 

    x = np.random.random(10)

    y = DT.inference(t,x)
    assert y == 'good job' 

    #----------------- 
    t = Node(None,None) 
    t.isleaf = False 
    t.i = 1
    t.th = 1.5
    c1 = Node(None,None)
    c2 = Node(None,None)
    c1.isleaf= True
    c2.isleaf= True
    
    c1.p = 'c1' 
    c2.p = 'c2' 
    t.C1 = c1 
    t.C2 = c2 

    x = np.array([1.,2.,3.,1.])
    y = DT.inference(t,x)
    assert y == 'c2' 

    t.th = 2.5
    y = DT.inference(t,x)
    assert y == 'c1' 





#-------------------------------------------------------------------------
def test_predict():
    ''' (2 points) predict'''
    t = Node(None,None) 
    t.isleaf = False 
    t.i = 1
    t.th = 1.5
    c1 = Node(None,None)
    c2 = Node(None,None)
    c1.isleaf= True
    c2.isleaf= True
    
    c1.p = 'c1' 
    c2.p = 'c2' 
    t.C1 = c1 
    t.C2 = c2 

    X = np.array([[1.,1.,1.,1.],
                  [1.,2.,3.,1.]])
    Y = DT.predict(t,X)

    assert type(Y) == np.ndarray
    assert Y.shape == (4,) 
    assert Y[0] == 'c1'
    assert Y[1] == 'c2'
    assert Y[2] == 'c2'
    assert Y[3] == 'c1'

    t.th = 2.5
    Y = DT.predict(t,X)

    assert type(Y) == np.ndarray
    assert Y.shape == (4,) 
    assert Y[0] == 'c1'
    assert Y[1] == 'c1'
    assert Y[2] == 'c2'
    assert Y[3] == 'c1'



#-------------------------------------------------------------------------
def test_train():
    ''' (2 points) train'''
    d = DT()
    X = np.array([[1.,2.,3.,4.],
                  [2.,4.,6.,8.],
                  [1.,2.,1.,1.]])
    Y = np.array(['good','bad','good','good'])
    t = d.train(X,Y) 

    assert t.i == 2 
    assert t.isleaf == False 
    assert t.p == 'good' 
    assert isinstance(t.C1, Node)
    assert isinstance(t.C2, Node)

    c1 = t.C1
    c2 = t.C2

    assert c1.isleaf == True
    assert c1.p == 'good' 
    assert c2.isleaf == True
    assert c2.p == 'bad' 


    X = np.array([[1.,1.,1.,1.],
                  [1.,2.,3.,1.]])
    Y = np.array(['okay','bad','good','okay'])
    t = d.train(X,Y) 

    assert t.i == 1 
    assert t.th == 1.5
    assert t.isleaf == False 
    assert t.p == 'okay' 
    assert isinstance(t.C1, Node)
    assert isinstance(t.C2, Node)

    c1 = t.C1
    c2 = t.C2

    assert c1.isleaf == True
    assert c1.p == 'okay' 
    assert c2.isleaf == False 
    assert c2.i == 1 
    assert c2.th == 2.5 
    assert c2.p == 'bad' or c2.p == 'good' 
    assert isinstance(c2.C1, Node)
    assert isinstance(c2.C2, Node)
    
    c2c1 = c2.C1
    c2c2 = c2.C2

    assert c2c1.isleaf == True
    assert c2c1.p == 'bad' 
    assert c2c2.isleaf == True
    assert c2c2.p == 'good' 

#-------------------------------------------------------------------------
def test_load_dataset():
    ''' (2 points) load_dataset'''
    X, Y = DT.load_dataset()
    assert type(X) == np.ndarray
    assert type(Y) == np.ndarray
    assert X.shape ==(7,42) 
    assert Y.shape ==(42,) 
    assert Y[0] == 'Bad'
    assert Y[1] == 'Bad'
    assert Y[-3] == 'Good'
    assert Y[-1] == 'Bad'
    assert Y[-11] == 'OK'
    assert X[0,0] ==8.
    assert X[0,-1] ==4.
    assert X[1,0] == 350.
    assert X[1,-1] == 105.
    assert X[-1,0] == 0.
    assert X[-1,-5] == 2.


#-------------------------------------------------------------------------
def test_dataset2():
    ''' (3 points) test dataset2'''
    d = DT()
    X, Y = DT.load_dataset()
    t = d.train(X,Y) 
    Y_predict = DT.predict(t,X) 
    accuracy = sum(Y==Y_predict)/42. # training accuracy of 42 training samples
    print('training accuracy:', accuracy)
    assert accuracy >= .95 # training accuracy 

    # train over half of the dataset
    t = d.train(X[:,::2],Y[::2]) 
    # test on the other half
    Y_predict = DT.predict(t,X[:,1::2]) 
    accuracy = sum(Y[1::2]==Y_predict)/21. 
    print('test accuracy:', accuracy)
    assert accuracy >= .75




