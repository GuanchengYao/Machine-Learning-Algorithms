import math
import numpy as np
from problem2 import DT,Node
#-------------------------------------------------------------------------
'''
    Problem 5: Boosting (on continous attributes). 
               We will implement AdaBoost algorithm in this problem.
    You could test the correctness of your code by typing `nosetests -v test5.py` in the terminal.
'''

#-----------------------------------------------
class DS(DT):
    '''
        Decision Stump (with contineous attributes) for Boosting.
        Decision Stump is also called 1-level decision tree.
        Different from other decision trees, a decision stump can have at most one level of child nodes.
        In order to be used by boosting, here we assume that the data instances are weighted.
    '''
    #--------------------------
    @staticmethod
    def entropy(Y, D):
        '''
            Compute the entropy of the weighted instances.
            Input:
                Y: a list of labels of the instances, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                e: the entropy of the weighted samples, a float scalar
            Hint: you could use np.unique(). 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        dic = {}
        for item in zip(Y,D):
            dic[item[0]] = dic[item[0]]+item[1] if item[0] in dic.keys() else item[1]
        e = sum(-y_i * math.log2(y_i) for y_i in dic.values() if y_i != 0)

        #########################################
        return e 
            
    #--------------------------
    @staticmethod
    def conditional_entropy(Y,X,D):
        '''
            Compute the conditional entropy of y given x on weighted instances
            Input:
                Y: a list of values, a numpy array of int/float/string values.
                X: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                ce: the weighted conditional entropy of y given x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        dicX, dicY = {}, {}
        for item in zip(X,D):
            dicX[item[0]] = dicX[item[0]]+item[1] if item[0] in dicX.keys() else item[1]
        for item in zip(Y,X,D):
            dicY[(item[0],item[1])] = dicY[(item[0],item[1])]+item[2] if (item[0],item[1]) in dicY.keys() else item[2]
        ce = 0
        for x_j in dicX.keys():
            sum_p = 0
            for key in dicY.keys():
                if key[1] == x_j and dicY[key] != 0: # Doesn't need to ensure dicX[x_j] != 0 because dicY[key] != 0 -> dicX[x_j] != 0
                    sum_p += dicY[key]/dicX[x_j] * math.log2(dicY[key]/dicX[x_j])
            ce += -dicX[x_j] * sum_p

        #########################################
        return ce 

    #--------------------------
    @staticmethod
    def information_gain(Y,X,D):
        '''
            Compute the information gain of y after spliting over attribute x
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                g: the weighted information gain of y after spliting over x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        g = DS.entropy(Y,D) - DS.conditional_entropy(Y,X,D)

        #########################################
        return g

    #--------------------------
    @staticmethod
    def best_threshold(X,Y,D):
        '''
            Find the best threshold among all possible cutting points in the continous attribute of X. The data instances are weighted. 
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
            Output:
                th: the best threhold, a float scalar. 
                g: the weighted information gain by using the best threhold, a float scalar. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        th, g = float('-inf'), -1
        cp = DS.cutting_points(X,Y)
        if cp[0] == float('-inf'):
            return th, g
        sortedX, sortedY, sortedD = [[x[i] for x in sorted(zip(X,Y,D))] for i in range(3)]
        for i in range(len(cp)):
            X_classified = [0] * len(sortedX)
            for j in range(len(sortedX)):
                if sortedX[j] > cp[i]:
                    X_classified[j] = 1
            cur_g = DS.information_gain(sortedY,X_classified,sortedD)
            if cur_g > g:
                th = cp[i]
                g = cur_g

        #########################################
        return th,g 
     
    #--------------------------
    def best_attribute(self,X,Y,D):
        '''
            Find the best attribute to split the node. The attributes have continous values (int/float). The data instances are weighted.
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Output:
                i: the index of the attribute to split, an integer scalar
                th: the threshold of the attribute to split, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        th, i, g = float('-inf'), 0, -1 
        for j in range(X.shape[0]):
            cur_th, cur_g = DS.best_threshold(X[j,:],Y,D)
            if cur_g > g:
                g = cur_g
                th = cur_th
                i = j

        #########################################
        return i, th
             
    #--------------------------
    @staticmethod
    def most_common(Y,D):
        '''
            Get the most-common label from the list Y. The instances are weighted.
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
                D: the weights of instances, a numpy float vector of length n
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        dic = {}
        for i in range(len(Y)):
            dic[Y[i]] = dic[Y[i]] + D[i] if Y[i] in dic.keys() else D[i]
        y = sorted(dic.items(), key = lambda x:x[1], reverse = True)[0][0]

        #########################################
        return y
 

    #--------------------------
    def build_tree(self, X,Y,D):
        '''
            build decision stump by overwritting the build_tree function in DT class.
            Instead of building tree nodes recursively in DT, here we only build at most one level of children nodes.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Return:
                t: the root node of the decision stump.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t = Node(X,Y)
        t.p = DS.most_common(t.Y, D)
        # if Condition 1 or 2 holds, stop splitting 
        if DS.stop1(t.Y) or DS.stop2(t.X):
            t.isleaf = True
            return t
        # find the best attribute to split
        t.i, t.th = self.best_attribute(t.X,t.Y,D)
        t.C1, t.C2 = DS.split(t.X,t.Y,t.i,t.th)
        # configure each child node
        D_child = sorted(zip(X[t.i,:],D))
        k = 0
        for j in range(len(D_child)):
            if D_child[j][0] > t.th:
                k = j
                break
        t.C1.p = DS.most_common(t.C1.Y,[x[1] for x in D_child[:k]])
        t.C2.p = DS.most_common(t.C2.Y,[x[1] for x in D_child[k:]])
        t.C1.isleaf = True
        t.C2.isleaf = True

        #########################################
        return t
    
 

#-----------------------------------------------
class AB(DS):
    '''
        AdaBoost algorithm (with contineous attributes).
    '''

    #--------------------------
    @staticmethod
    def weighted_error_rate(Y,Y_,D):
        '''
            Compute the weighted error rate of a decision on a dataset. 
            Input:
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                Y_: the predicted class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Output:
                e: the weighted error rate of the decision stump
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        e  = sum([D[i] for i in range(len(Y)) if Y[i] != Y_[i]])

        #########################################
        return e

    #--------------------------
    @staticmethod
    def compute_alpha(e):
        '''
            Compute the weight a decision stump based upon weighted error rate.
            Input:
                e: the weighted error rate of a decision stump
            Output:
                a: (alpha) the weight of the decision stump, a float scalar.
        '''
        #########################################
        ## INSERT YOUR CODE HERE    
        if e == 0:
            a = 200
        elif e == 1:
            a = -200
        else:
            a = 0.5 * math.log((1 - e) / e)

        #########################################
        return a

    #--------------------------
    @staticmethod
    def update_D(D,a,Y,Y_):
        '''
            update the weight the data instances 
            Input:
                D: the current weights of instances, a numpy float vector of length n
                a: (alpha) the weight of the decision stump, a float scalar.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                Y_: the predicted class labels by the decision stump, a numpy array of length n. Each element can be int/float/string.
            Output:
                D: the new weights of instances, a numpy float vector of length n
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        # (Pass-by-object-reference), Can't change values in D directly.
        D_ = [0] * len(Y)
        for i in range(len(Y)):
            D_[i] = D[i] * math.e ** -a if Y[i] == Y_[i] else D[i] * math.e ** a
        D = D_ / sum(D_)

        #########################################
        return D

    #--------------------------
    @staticmethod
    def step(X,Y,D):
        '''
            Compute one step of Boosting.  
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the current weights of instances, a numpy float vector of length n
            Output:
                t:  the root node of a decision stump trained in this step
                a: (alpha) the weight of the decision stump, a float scalar.
                D: the new weights of instances, a numpy float vector of length n
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t = AB().build_tree(X,Y,D)
        Y_ = DT.predict(t,X)
        e = AB.weighted_error_rate(Y,Y_,D)
        a = AB.compute_alpha(e)
        D = AB.update_D(D,a,Y,Y_)

        #########################################
        return t,a,D

    
    #--------------------------
    @staticmethod
    def inference(x,T,A):
        '''
            Given an adaboost ensemble of decision trees and one data instance, infer the label of the instance. 
            Input:
                x: the attribute vector of a data instance, a numpy vectr of shape p.
                   Each attribute value can be int/float
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
            Output:
                y: the class label, a scalar of int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        Y = []
        Y = [DT.inference(t,x) for t in T]
        y = DS.most_common(Y, A)

        #########################################
        return y
 

    #--------------------------
    @staticmethod
    def predict(X,T,A):
        '''
            Given an AdaBoost and a dataset, predict the labels on the dataset. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
            Output:
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        Y = []
        Y.extend(AB.inference(X[:,i],T,A) for i in range(X.shape[1]))
        Y = np.asarray(Y)

        #########################################
        return Y 
 

    #--------------------------
    @staticmethod
    def train(X,Y,n_tree=10):
        '''
            train adaboost.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                n_tree: the number of trees in the ensemble, an integer scalar
            Output:
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        # initialize weight as 1/n
        n = X.shape[1]
        D = np.ones(n)/n
        # iteratively build decision stumps
        T, A = [], np.array([])
        for _ in range(n_tree):
            t, a, D = AB.step(X,Y,D)
            T.append(t)
            A = np.append(A, a)

        #########################################
        return T, A
   



 
