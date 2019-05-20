import math
import numpy as np
from collections import Counter
#-------------------------------------------------------------------------
'''
    Problem 1: Decision Tree (with Descrete Attributes)
    In this problem, you will implement the decision tree method for classification problems.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''
 
#--------------------------
def Terms_and_Conditions():
    ''' 
        By submiting this homework or changing this function, you agree with the following terms:
       (1) Not sharing your code/solution with any student before and after the homework due. For example, sending your code segment to another student, putting your solution online or lending your laptop (if your laptop contains your solution or your dropbox automatically sychronize your solution between your home computer and your laptop) to another student to work on this homework will violate this term.
       (2) Not using anyone's code in this homework, build your own solution. For example, using some code segments from another student or online resources due to any reason (like too busy recently) will violate this term. Changing other people's code as your solution (such as changing the variable names) will also violate this term.
       (3) When discussing with any other student about this homework, only discuss high-level ideas or using psudo-code. Don't discuss about the solution at the code level. For example, discussing with another student about the solution of a function (which needs 5 lines of code to solve), and then working on the solution "independently", however the code of the two solutions are exactly the same, or only with minor differences  (like changing variable names) will violate this term.
      All violations of (1),(2) or (3) will be handled in accordance with the WPI Academic Honesty Policy.  For more details, please visit: https://www.wpi.edu/about/policies/academic-integrity/dishonesty
      Historical Data: in one year, we ended up finding 25% of the students in the class violating this term in their homework submissions and we handled ALL of these violations according to the WPI Academic Honesty Policy. 
    '''
    #########################################
    ## CHANGE CODE HERE
    Read_and_Agree = True
    #########################################
    return Read_and_Agree
        
#-----------------------------------------------
class Node:
    '''
        Decision Tree Node (with discrete attributes)
        Inputs: 
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float/string.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar 
            C: the dictionary of attribute values and children nodes. 
               Each (key, value) pair represents an attribute value and its corresponding child node.
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''
    def __init__(self,X,Y, i=None,C=None, isleaf= False,p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.C= C
        self.isleaf = isleaf
        self.p = p

#-----------------------------------------------
class Tree(object):
    '''
        Decision Tree (with discrete attributes). 
        We are using ID3(Iterative Dichotomiser 3) algorithm. So this decision tree is also called ID3.
    '''
    #--------------------------
    @staticmethod
    def entropy(Y):
        '''
            Compute the entropy of a list of values.
            Input:
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                e: the entropy of the list of values, a float scalar
            Hint: you could use collections.Counter.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        e = sum([-y_i/len(Y) * math.log2(y_i/len(Y)) for y_i in Counter(Y).values()])

        #########################################
        return e 
    
    
            
    #--------------------------
    @staticmethod
    def conditional_entropy(Y,X):
        '''
            Compute the conditional entropy of y given x.
            Input:
                Y: a list of values, a numpy array of int/float/string values.
                X: a list of values, a numpy array of int/float/string values.
            Output:
                ce: the conditional entropy of y given x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        ce = sum([-Counter(X)[x_j]/len(X) * sum([Counter(zip(X,Y))[key]/Counter(X)[x_j] * math.log2(Counter(zip(X,Y))[key]/Counter(X)[x_j]) for key in Counter(zip(X,Y)).keys() if key[0] == x_j]) for x_j in set(X)])

        #########################################
        return ce 
    
    
    
    #--------------------------
    @staticmethod
    def information_gain(Y,X):
        '''
            Compute the information gain of y after spliting over attribute x
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                g: the information gain of y after spliting over x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        g = Tree.entropy(Y) -  Tree.conditional_entropy(Y,X)

        #########################################
        return g


    #--------------------------
    @staticmethod
    def best_attribute(X,Y):
        '''
            Find the best attribute to split the node. 
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        i = sorted([(Tree.information_gain(Y,X[j,:]), j) for j in range(X.shape[0])], key = lambda x:x[0], reverse = True)[0][1]

        #########################################
        return i

        
    #--------------------------
    @staticmethod
    def split(X,Y,i):
        '''
            Split the node based upon the i-th attribute.
            (1) split the matrix X based upon the values in i-th attribute
            (2) split the labels Y based upon the values in i-th attribute
            (3) build children nodes by assigning a submatrix of X and Y to each node
            (4) build the dictionary to combine each  value in the i-th attribute with a child node.
    
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C: the dictionary of attribute values and children nodes. 
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        C = {}
        dicX, dicY = {}, {}
        X = np.mat(X)
        for j in range(X.shape[1]):
            if X[i,j] in dicX:
                dicX[X[i,j]] = np.concatenate((dicX[X[i,j]],X[:,j]), axis=1)
                dicY[X[i,j]] = np.append(dicY[X[i,j]],Y[j])
            else:
                dicX[X[i,j]] = X[:,j]
                dicY[X[i,j]] = np.array([Y[j]])
        for k,v in dicX.items():
            C[k] = Node(v.getA(),dicY[k])

        #########################################
        return C

    #--------------------------
    @staticmethod
    def stop1(Y):
        '''
            Test condition 1 (stop splitting): whether or not all the instances have the same label. 
    
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                s: whether or not Conidtion 1 holds, a boolean scalar. 
                True if all labels are the same. Otherwise, false.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        s = len(set(Y)) == 1

        #########################################
        return s
    
    #--------------------------
    @staticmethod
    def stop2(X):
        '''
            Test condition 2 (stop splitting): whether or not all the instances have the same attributes. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
            Output:
                s: whether or not Conidtion 2 holds, a boolean scalar. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        s = np.all(X[:,0] == [X[:,i] for i in range(X.shape[1])]) == True

        #########################################
        return s
    
            
    #--------------------------
    @staticmethod
    def most_common(Y):
        '''
            Get the most-common label from the list Y. 
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        y = Counter(Y).most_common()[0][0]

        #########################################
        return y
    
    
    
    #--------------------------
    @staticmethod
    def build_tree(t):
        '''
            Recursively build tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape n by p.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
                t.C: the dictionary of attribute values and children nodes. 
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t.p = Tree.most_common(t.Y)
        # if Condition 1 or 2 holds, stop recursion 
        if Tree.stop1(t.Y) or Tree.stop2(t.X):
            t.isleaf = True
            return
        # find the best attribute to split
        t.i = Tree.best_attribute(t.X,t.Y)
        t.C = Tree.split(t.X,t.Y,t.i)
        # recursively build subtree on each child node
        for child in t.C.values():
            Tree.build_tree(child)

        #########################################
    
    
    #--------------------------
    @staticmethod
    def train(X, Y):
        '''
            Given a training set, train a decision tree. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                t: the root of the tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t = Node(X, Y)
        Tree.build_tree(t)

        #########################################
        return t
    
    
    
    #--------------------------
    @staticmethod
    def inference(t,x):
        '''
            Given a decision tree and one data instance, infer the label of the instance recursively. 
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float/string.
            Output:
                y: the class label, a scalar, which can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        # predict label, if the current node is a leaf node
        if t.isleaf or x[t.i] not in t.C.keys():
            return t.p
        y = Tree.inference(t.C[x[t.i]],x)

        #########################################
        return y
    
    #--------------------------
    @staticmethod
    def predict(t,X):
        '''
            Given a decision tree and a dataset, predict the labels on the dataset. 
            Input:
                t: the root of the tree.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        '''Y = np.array([])
        for i in range(X.shape[1]):
            Y = np.append(Y, Tree.inference(t,X[:,i]))'''
        Y = []
        Y.extend([Tree.inference(t,X[:,i]) for i in range(X.shape[1])])
        Y = np.asarray(Y)
        #########################################
        return Y



    #--------------------------
    @staticmethod
    def load_dataset(filename='data1.csv'):
        '''
            Load dataset 1 from the CSV file: 'data1.csv'. 
            The first row of the file is the header (including the names of the attributes)
            In the remaining rows, each row represents one data instance.
            The first column of the file is the label to be predicted.
            In remaining columns, each column represents an attribute.
            Input:
                filename: the filename of the dataset, a string.
            Output:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element is a string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element is a string.
            Note: Here you can assume the data type is always str.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        Z = np.loadtxt(filename, dtype=np.str, delimiter=',')
        X = Z[1:,1:].T
        Y = Z[1:,0]
        #########################################
        return X,Y



