from problem5 import *
from problem3 import Bag
import sys
import numpy as np
'''
    Unit test 5:
    This file includes unit tests for problem5.py.
    You could test the correctness of your code by typing `nosetests -v test5.py` in the terminal.
'''
#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 5 (25 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)



#-------------------------------------------------------------------------
def test_entropy():
    ''' (1 points) entropy'''
    y = np.array([0.,0.])
    d = np.array([.5,.5])
    e = DS.entropy(y,d)
    assert np.allclose(e, 0., atol = 1e-3) 

    y = np.array([2.,2.])
    d = np.array([.5,.5])
    e = DS.entropy(y,d)
    assert np.allclose(e, 0., atol = 1e-3) 

    y = np.array([0.,1.])
    d = np.array([.5,.5])
    e = DS.entropy(y,d)
    assert np.allclose(e, 1.0, atol = 1e-3) 

    y = np.array([0.,1.])
    d = np.array([0.,1.])
    e = DS.entropy(y,d)
    assert np.allclose(e, 0., atol = 1e-3)  

    y = np.array([0.,1.,0.,1.])
    d = np.array([.25,.25,.25,.25])
    e = DS.entropy(y,d)
    assert np.allclose(e, 1.0, atol = 1e-3) 

    y = np.array([0.,1.,1.])
    d = np.array([.5,.25,.25])
    e = DS.entropy(y,d)
    assert np.allclose(e, 1.0, atol = 1e-3) 

    y = np.array([4.,8.,4.,8.])
    d = np.array([.25,.25,.25,.25])
    e = DS.entropy(y,d)
    assert np.allclose(e, 1.0, atol = 1e-3) 

    y = np.array([0.,1.,1.,1.,1.,1.])
    d = np.ones_like(y)/len(y)
    e = DS.entropy(y,d)
    assert np.allclose(e, .65, atol = 1e-3) 

    y = np.array(['apple','apple'])
    d = np.array([.5,.5])
    e = DS.entropy(y,d)
    assert np.allclose(e, 0., atol = 1e-3) 

    y = np.array(['orange','apple'])
    d = np.array([.5,.5])
    e = DS.entropy(y,d)
    assert np.allclose(e, 1., atol = 1e-3) 

    y = np.array(['orange','apple'])
    d = np.array([1./6.,5./6.])
    e = DS.entropy(y,d)
    assert np.allclose(e, .65, atol = 1e-3) 

    y = np.array(['orange','apple','orange','apple'])
    d = np.array([.25,.25,.25,.25])
    e = DS.entropy(y,d)
    assert np.allclose(e, 1., atol = 1e-3) 

    y = np.array(['orange','apple','banana','pineapple'])
    d = np.array([.25,.25,.25,.25])
    e = DS.entropy(y,d)
    assert np.allclose(e, 2., atol = 1e-3) 

#-------------------------------------------------------------------------
def test_conditional_entropy():
    ''' (1 points) conditional entropy'''

    y = np.array([0.,0.])
    d = np.array([.5,.5])
    x = np.array([1.,1.])
    ce = DS.conditional_entropy(y,x,d)
    assert np.allclose(ce, 0., atol = 1e-3) 

    y = np.array([0.,1.])
    x = np.array([1.,2.])
    d = np.array([.5,.5])
    ce = DS.conditional_entropy(y,x,d)
    assert np.allclose(ce, 0., atol = 1e-3) 

    y = np.array([0.,1.])
    x = np.array([1.,2.])
    d = np.array([1.,0.])
    ce = DS.conditional_entropy(y,x,d)
    assert np.allclose(ce, 0., atol = 1e-3) 

    y = np.array([0.,1.,0.,1.])
    x = np.array([1.,4.,1.,4.])
    d = np.array([.25,.25,.25,.25])
    ce = DS.conditional_entropy(y,x,d)
    assert np.allclose(ce, 0., atol = 1e-3) 

    y = np.array([0.,1.,0.,1.])
    x = np.array([1.,1.,4.,4.])
    d = np.array([.25,.25,.25,.25])
    ce = DS.conditional_entropy(y,x,d)
    assert np.allclose(ce, 1., atol = 1e-3) 

    y = np.array([0.,0.,1.])
    x = np.array([1.,4.,4.])
    d = np.ones_like(y)/len(y)
    ce = DS.conditional_entropy(y,x,d)
    assert np.allclose(ce, 0.666666666667, atol = 1e-3) 

    y = np.array([0.,0.,0.,1.])
    x = np.array([1.,1.,4.,4.])
    d = np.array([.25,.25,.25,.25])
    ce = DS.conditional_entropy(y,x,d)
    assert np.allclose(ce, .5, atol = 1e-3) 

    y = np.array([0.,0.,1.])
    x = np.array([1.,4.,4.])
    d = np.array([.5,.25,.25])
    ce = DS.conditional_entropy(y,x,d)
    assert np.allclose(ce, .5, atol = 1e-3) 

    y = np.array(['apple','apple'])
    x = np.array(['good','good'])
    d = np.array([.5,.5])
    ce = DS.conditional_entropy(y,x,d)
    assert np.allclose(ce, 0., atol = 1e-3) 

    y = np.array(['apple','orange'])
    x = np.array(['good','good'])
    d = np.array([.5,.5])
    ce = DS.conditional_entropy(y,x,d)
    assert np.allclose(ce, 1., atol = 1e-3) 

    y = np.array(['apple','orange'])
    x = np.array(['good','bad'])
    d = np.array([.5,.5])
    ce = DS.conditional_entropy(y,x,d)
    assert np.allclose(ce, 0., atol = 1e-3) 

    y = np.array(['apple','orange','pineapple','banana'])
    x = np.array(['a','a','a','a'])
    d = np.array([.25,.25,.25,.25])
    ce = DS.conditional_entropy(y,x,d)
    assert np.allclose(ce, 2., atol = 1e-3) 

    y = np.array(['apple','orange','pineapple','banana'])
    x = np.array(['a','a','b','b'])
    d = np.array([.25,.25,.25,.25])
    ce = DS.conditional_entropy(y,x,d)
    assert np.allclose(ce, 1., atol = 1e-3) 


    y = np.array(['apple','orange','apple','orange'])
    x = np.array(['a','a','b','b'])
    d = np.array([.5,0.,.5,0.])
    ce = DS.conditional_entropy(y,x,d)
    assert np.allclose(ce, 0., atol = 1e-3) 


#-------------------------------------------------------------------------
def test_information_gain():
    ''' (1 point) information gain'''

    y = np.array([0.,1.])
    x = np.array([1.,2.])
    d = np.array([.5,.5])
    g = DS.information_gain(y,x,d)
    assert np.allclose(g, 1., atol = 1e-3) 

    y = np.array([0.,0.])
    x = np.array([1.,1.])
    d = np.array([.5,.5])
    g = DS.information_gain(y,x,d)
    assert np.allclose(g, 0., atol = 1e-3) 
  
    y = np.array([0.,0.,0.,1.])
    x = np.array([1.,1.,4.,4.])
    d = np.array([.25,.25,.25,.25])
    g = DS.information_gain(y,x,d)
    assert np.allclose(g, 0.311278124459, atol = 1e-3)   


    y = np.array([0.,0.,1.])
    x = np.array([1.,4.,4.])
    d = np.ones_like(y)/len(y)
    g = DS.information_gain(y,x,d)
    assert np.allclose(g, 0.251629167388, atol = 1e-3) 

    y = np.array([0.,0.,1.])
    x = np.array([1.,4.,4.])
    d = np.array([.5,.25,.25])
    g = DS.information_gain(y,x,d)
    assert np.allclose(g, 0.311278124459, atol = 1e-3) 

    y = np.array(['apple','orange'])
    x = np.array(['good','bad'])
    d = np.array([.5,.5])
    g = DS.information_gain(y,x,d)
    assert np.allclose(g, 1., atol = 1e-3)

    y = np.array(['apple','apple'])
    x = np.array(['good','bad'])
    d = np.array([.5,.5])
    g = DS.information_gain(y,x,d)
    assert np.allclose(g, 0., atol = 1e-3)

#-------------------------------------------------------------------------
def test_best_threshold():
    ''' (1 point) best threshold'''
    y = np.array([1.,1.,0.,0.])
    x = np.array([1.,2.,3.,4.])
    d = np.ones_like(y)/len(y)
    th,g = DS.best_threshold(x,y,d)
    assert th == 2.5 
    assert g == 1. 

    y = np.array([0.,1.,0.,0.])
    x = np.array([1.,2.,3.,4.])
    d = np.array([3,1,1,1])/6.
    th,g = DS.best_threshold(x,y,d)
    assert th == 1.5 
    assert np.allclose(g,0.1908,atol = 1e-3)

    y = np.array([0.,0.,1.,0., 1., 1.,1.])
    x = np.array([1.,2.,3.,4., 5., 6.,7.])
    d = np.ones_like(y)/len(y)
    th,g = DS.best_threshold(x,y,d)
    assert th == 4.5 
    assert np.allclose(g,0.52164063634,atol = 1e-3)

    y = np.array([0.,0.,1.])
    x = np.array([1.,1.,1.]) # if all values are the same
    d = np.ones_like(y)/len(y)
    th, g = DS.best_threshold(x,y,d)
    assert th ==  - float('inf') # manulally set threshold as -inf 
    assert g == -1. # manually set gain as -1 to avoid selecting this attribute.


#-------------------------------------------------------------------------
def test_best_attribute():
    ''' (1 points) best attribute'''
    s = DS()
    X = np.array([[0.,0.],
                  [2.,2.],
                  [1.,2.],
                  [3.,3.]])
    Y = np.array(['good','bad'])
    d = np.ones(2)/2
    i,th = s.best_attribute(X,Y,d)
    assert i == 2 
    assert th == 1.5



    X = np.array([[0.,0.,1.,1.],
                  [2.,2.,1.,1.],
                  [1.,2.,1.,1.],
                  [0.,3.,1.,2.]])
    Y = np.array(['good','bad','good','bad'])
    d = np.ones(4)/4
    i,th = s.best_attribute(X,Y,d)
    assert i == 3 
    assert th == 1.5


    X = np.array([[0.,0.,0.,0.],
                  [2.,2.,1.,1.],
                  [0.,1.,1.,1.]])
    Y = np.array(['good','bad','okay','okay'])
    d = np.array([.5,.5,.0,.0]) 
    i,th = s.best_attribute(X,Y,d)
    assert i == 2 
    assert th == 0.5 

#-------------------------------------------------------------------------
def test_most_common():
    ''' (1 points) most common'''
    Y = np.array(['good','bad','good','perfect'])
    D = np.ones(4)/4.
    assert DS.most_common(Y,D) == 'good'

    Y = np.array(['a','b','b','b','c','c'])
    D = np.ones(6)/6.
    assert DS.most_common(Y,D) == 'b'

    Y = np.array(['good','bad','good','perfect'])
    D = np.array([.1,.2,.2,.5])
    assert DS.most_common(Y,D) == 'perfect'
#-------------------------------------------------------------------------
def test_build_tree():
    ''' (2 points) build tree'''
    d = DS()
    X = np.array([[1.],
                  [2.],
                  [3.]])
    Y = np.array(['bad'])
    D = np.array([1.])
    
    # build tree
    t=d.build_tree(X,Y,D)

    assert t.isleaf == True
    assert t.p == 'bad' 
    assert t.C1 == None
    assert t.C2 == None
    assert t.i == None

    #------------
    X = np.random.random((5,4))
    Y = np.array(['good','good','good','good'])
    D = np.ones(4)/4.
    
    # build tree
    t=d.build_tree(X,Y,D)

    assert t.isleaf == True
    assert t.p == 'good' 
    assert t.C1 == None
    assert t.C2 == None
    assert t.i == None


    #------------
    X = np.ones((5,4))
    Y = np.array(['good','bad','bad','bad'])
    D = np.ones(4)/4.
    
    # build tree
    t=d.build_tree(X,Y,D)

    assert t.isleaf == True
    assert t.p == 'bad' 
    assert t.C1 == None
    assert t.C2 == None
    assert t.i == None


    #------------
    X = np.array([[1.,2.,3.,4.],
                  [2.,4.,6.,8.],
                  [1.,2.,1.,1.]])
    Y = np.array(['good','bad','good','good'])
    D = np.ones(4)/4.
    
    # build tree
    t=d.build_tree(X,Y,D)

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


    #------------
    X = np.array([[1.,1.,1.,1.],
                  [1.,2.,3.,1.]])
    Y = np.array(['okay','bad','good','okay'])
    D = np.ones(4)/4.
    
    # build tree
    t=d.build_tree(X,Y,D)

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
    assert c2.isleaf == True 
    assert c2.i == None 
    assert c2.th == None 
    assert c2.p == 'bad' or c2.p == 'good' 
    assert c2.C1== None 
    assert c2.C2== None 
    
    #------------
    X = np.array([[1.,1.,1.,1.],
                  [1.,2.,3.,1.]])
    Y = np.array(['okay','bad','good','okay'])
    D = np.array([.25,.1,.4,.25])
    
    # build tree
    t=d.build_tree(X,Y,D)

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
    assert c2.isleaf == True 
    assert c2.i == None 
    assert c2.th == None 
    assert c2.p == 'good' 
    assert c2.C1== None 
    assert c2.C2== None 
    
    
    #------------
    X = np.array([[1.,1.,1.,1.],
                  [1.,2.,3.,1.]])
    Y = np.array(['okay','bad','good','okay'])
    D = np.array([.0,.1,.9,.0])
    
    # build tree
    t=d.build_tree(X,Y,D)

    assert t.i == 1 
    assert t.th == 2.5
    assert t.isleaf == False 
    assert t.p == 'good' 
    assert isinstance(t.C1, Node)
    assert isinstance(t.C2, Node)

    c1 = t.C1
    c2 = t.C2

    assert c1.isleaf == True
    assert c1.p == 'bad' 
    assert c2.isleaf == True 
    assert c2.i == None 
    assert c2.th == None 
    assert c2.p == 'good' 
    assert c2.C1== None 
    assert c2.C2== None 


#-------------------------------------------------------------------------
def test_ds_predict():
    ''' (2 points) DS predict'''
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
    Y_ = np.array(['c1','c1','c2','c2'])
    Y = DS.predict(t,X)

    assert type(Y) == np.ndarray
    assert Y.shape == (4,) 
    assert Y[0] == 'c1'
    assert Y[1] == 'c2'
    assert Y[2] == 'c2'
    assert Y[3] == 'c1'

#-------------------------------------------------------------------------
def test_weighted_error_rate():
    ''' (1 point) weighted error rate'''
    Y = np.array(['c1','c2','c2','c1'])
    Y_ = np.array(['c1','c1','c2','c2'])
    D = np.ones(4)/4.
    e=AB.weighted_error_rate(Y,Y_,D)
    assert e == 0.5
    D = np.array([.5,.0,.5,.0])
    e=AB.weighted_error_rate(Y,Y_,D)
    assert e == 0.0
    D = np.array([0.,.5,.0,.5])
    e=AB.weighted_error_rate(Y,Y_,D)
    assert e == 1.0

#-------------------------------------------------------------------------
def test_compute_alpha():
    ''' (2 points) compute_alpha'''
    assert np.allclose(AB.compute_alpha(0.5),0.,1e-3) 
    assert np.allclose(AB.compute_alpha(0.4),0.202732554054,1e-3) 
    assert np.allclose(AB.compute_alpha(0.3),0.423648930194,1e-3) 
    assert np.allclose(AB.compute_alpha(0.2),0.69314718056,1e-3) 
    assert np.allclose(AB.compute_alpha(0.1),1.09861228867,1e-3) 
    assert np.allclose(AB.compute_alpha(0.01),2.29755992507,1e-3) 
    assert np.allclose(AB.compute_alpha(1e-3),3.4533773893,1e-3) 
    assert np.allclose(AB.compute_alpha(1e-4),4.60512018349,1e-3) 
    assert np.allclose(AB.compute_alpha(1e-5),5.75645773246, 1e-3) 
    assert np.allclose(AB.compute_alpha(1e-6),6.90775477898, 1e-3) 
    assert np.allclose(AB.compute_alpha(1e-7),8.05904777548, 1e-3) 
    assert AB.compute_alpha(0.)>100.
    assert math.exp(AB.compute_alpha(0.))<float('inf') # will cause problem when computing exp(a) in update_D function, if a is larger than 700

    assert np.allclose(AB.compute_alpha(0.6),-0.202732554054,1e-3) 
    assert np.allclose(AB.compute_alpha(0.7),-0.423648930194,1e-3) 
    assert np.allclose(AB.compute_alpha(0.8),-0.69314718056,1e-3) 
    assert np.allclose(AB.compute_alpha(0.9),-1.09861228867,1e-3) 
    assert np.allclose(AB.compute_alpha(0.99),-2.29755992507,1e-3) 
    assert np.allclose(AB.compute_alpha(1.-1e-3),-3.4533773893,1e-3) 
    assert np.allclose(AB.compute_alpha(1.-1e-4),-4.60512018349,1e-3) 
    assert np.allclose(AB.compute_alpha(1.-1e-5),-5.75645773246, 1e-3) 
    assert np.allclose(AB.compute_alpha(1.-1e-6),-6.90775477898, 1e-3) 
    assert np.allclose(AB.compute_alpha(1.-1e-7),-8.05904777548, 1e-3) 
    assert AB.compute_alpha(1.)<-100.
    assert math.exp(-AB.compute_alpha(1.))<float('inf') # will cause problem when computing exp(-a) in update_D function, if a is smaller than -700
 

#-------------------------------------------------------------------------
def test_update_D():
    ''' (2 points) update_D'''
    D = np.ones(3)/3.
    Y = np.array([1.,1.,1.])
    Y_ = np.array([1.,1.,1.])
    a = 1.
    D_new = AB.update_D(D,a,Y,Y_) 
    assert np.allclose(D_new.sum(),1.,atol=1e-3)
    assert np.allclose(D_new,D,atol=1e-3)

    D = np.array([.5,.2,.3])
    Y = np.array([1.,1.,1.])
    Y_ = np.array([1.,1.,1.])
    a = 1.
    D_new = AB.update_D(D,a,Y,Y_) 
    assert np.allclose(D_new.sum(),1.,atol=1e-3)
    assert np.allclose(D_new,D,atol=1e-3)

    D = np.array([.5,.2,.3])
    Y = np.array([1.,1.,1.])
    Y_ = np.array([-1.,-1.,-1.])
    a = 1.
    D_new = AB.update_D(D,a,Y,Y_) 
    assert np.allclose(D_new.sum(),1.,atol=1e-3)
    assert np.allclose(D_new,D,atol=1e-3)

    D = np.ones(2)/2.
    Y = np.array([1.,1.])
    Y_ = np.array([1.,-1.])
    a = 1.
    D_new = AB.update_D(D,a,Y,Y_) 
    assert np.allclose(D_new.sum(),1.,atol=1e-3)
    assert np.allclose(D_new,[0.11920292,0.88079708],atol=1e-3)

    D = np.ones(2)/2.
    Y = np.array([1.,1.])
    Y_ = np.array([1.,-1.])
    a = 0.
    D_new = AB.update_D(D,a,Y,Y_) 
    assert np.allclose(D_new.sum(),1.,atol=1e-3)
 
    D = np.ones(2)/2.
    Y = np.array(['good','good'])
    Y_ = np.array(['good','bad'])
    a = 0.
    D_new = AB.update_D(D,a,Y,Y_) 
    assert np.allclose(D_new.sum(),1.,atol=1e-3)
    assert np.allclose(D_new,D,atol=1e-3)

#-------------------------------------------------------------------------
def test_step():
    ''' (2 points) step'''
    X = np.array([[1.,2.,3.,4.],
                  [2.,4.,6.,8.],
                  [1.,2.,1.,1.]])
    Y = np.array(['good','bad','good','good'])
    D = np.ones(4)/4.
    t,a,D_new = AB.step(X,Y,D) 
    assert np.allclose(D_new, D, atol=1e-3)

    X = np.ones((3,4)) 
    Y = np.array(['good','bad','good','bad'])
    D = np.array([0.5,0.,0.5,0.])
    t,a,D_new = AB.step(X,Y,D) 
    assert np.allclose(D_new, D, atol=1e-3)

    X = np.ones((3,4)) 
    Y = np.array(['good','bad','good','bad'])
    D = np.array([0.5,0.5,0.,0.])
    t,a,D_new = AB.step(X,Y,D) 
    assert np.allclose(D_new, D, atol=1e-3)

    X = np.array([[1.,1.,1.,1.],
                  [2.,2.,4.,4.],
                  [1.,1.,1.,1.]])
    Y = np.array(['good','bad','good','good'])
    D = np.ones(4)/4.
    t,a,D_new = AB.step(X,Y,D) 
    assert np.allclose(D_new[2:],[0.16666667,0.16666667],atol=1e-3)
    assert np.allclose(D_new[:2],[0.5,0.16666667],atol=1e-3) or np.allclose(D_new[:2],[0.16666667,.5], atol = 1e-3) 



#-------------------------------------------------------------------------
def test_inference():
    ''' (2 points) inference'''

    t = Node(None,None) 
    t.isleaf = True
    t.p = 'good job' 
    T = [t,t,t]
    A = np.ones(3)/3.

    x = np.random.random(10)

    y = AB.inference(x,T,A)
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
    y = AB.inference(x,[t,t2,t2],A)
    assert y == 'c2' 

    y = AB.inference(x,[t,t,t2],A)
    assert y == 'c1' 

    A = np.array([.6,.2,.2])
    y = AB.inference(x,[t,t2,t2],A)
    assert y == 'c1' 

    A = np.array([.2,.2,.6])
    y = AB.inference(x,[t,t,t2],A)
    assert y == 'c2' 


#-------------------------------------------------------------------------
def test_predict():
    ''' (2 points) AB predict'''
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
    A = np.ones(3)/3.
    Y = AB.predict(X,[t,t,t2],A)

    assert type(Y) == np.ndarray
    assert Y.shape == (4,) 
    assert Y[0] == 'c1'
    assert Y[1] == 'c1'
    assert Y[2] == 'c1'
    assert Y[3] == 'c1'

    Y = AB.predict(X,[t,t2,t2],A)
    assert Y[0] == 'c1'
    assert Y[1] == 'c2'
    assert Y[2] == 'c2'
    assert Y[3] == 'c1'

    A = np.array([.6,.2,.2])
    Y = AB.predict(X,[t,t2,t2],A)
    assert Y[0] == 'c1'
    assert Y[1] == 'c1'
    assert Y[2] == 'c1'
    assert Y[3] == 'c1'


    A = np.array([.2,.2,.6])
    Y = AB.predict(X,[t,t,t2],A)
    assert Y[0] == 'c1'
    assert Y[1] == 'c2'
    assert Y[2] == 'c2'
    assert Y[3] == 'c1'


#-------------------------------------------------------------------------
def test_train():
    ''' (2 points) AB train'''
    b = AB()

    X = np.array([[1.,1.,1.,1.],
                  [2.,2.,2.,2.],
                  [3.,3.,3.,3.]])
    Y = np.array(['good','good','good','good'])
    T,A = b.train(X,Y,1) 
    assert type(A)==np.ndarray
    assert len(T) == 1
    t = T[0]
    assert t.isleaf == True
    assert t.p == 'good' 

    for _ in range(20):
        n_tree = np.random.randint(1,10)
        T,A = b.train(X,Y,n_tree) 
        assert len(T) == n_tree
        for i in range(n_tree):
            t = T[i]
            assert t.isleaf == True
            assert t.p == 'good' 


#-------------------------------------------------------------------------
def test_dataset3():
    ''' (2 points) test dataset3'''
    b = AB()
    X, Y = Bag.load_dataset()
    n = float(len(Y))

    # train over half of the dataset
    T,A = b.train(X[:,::2],Y[::2],5) 
    # test on the other half
    Y_predict = AB.predict(X[:,1::2],T,A) 
    accuracy = sum(Y[1::2]==Y_predict)/n*2. 
    print('test accuracy of an AdaBoost ensemble of 5 trees:', accuracy)
    assert accuracy >= .85

    # train over half of the dataset
    T,A = b.train(X[:,::2],Y[::2],20) 
    # test on the other half
    Y_predict = AB.predict(X[:,1::2],T,A) 
    accuracy = sum(Y[1::2]==Y_predict)/n*2. 
    print('test accuracy of an AdaBoost ensemble of 20 trees:', accuracy)
    assert accuracy >= .98

