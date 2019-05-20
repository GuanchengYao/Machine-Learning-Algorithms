import math
import numpy as np
from collections import Counter
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 1: Naive Bayes Classifier (with discrete attributes)
    In this problem, you will implement the naive Bayes classification method. 
    In the data1.csv file, we have a collection of email spam detection data. 
    The class label indicate whether or not an email is spam (1: spam; 0: not spam).
    Each email has many features where each feature represents whether or not a certain word has appeared in the email.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
    Note: please don't use any existing package for classification problems, implement your own version.
'''

#-----------------------------------------------
def prob_smooth(X,c=2,k=1):
    '''
        Estimate the probability distribution of a random variable with Laplace smoothing.
        Input:
            X: the observed values of training samples, an integer numpy vector of length n. 
                Here n is the number of training instances. Each X[i] = 0,1, ..., or c-1. 
            c: the number of possible values of the variable, an integer scalar.
            k: the parameter of Laplace smoothing, an integer, denoting the number of imagined instances observed for each possible value. 
        Output:
            P: the estimated probability distribution, a numpy vector of length c.
                Each P[i] is the estimated probability of the i-th value.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    n = len(X)
    P = np.array([k / (n + k * c)] * c)
    count = Counter(X)
    for key,v in count.items():
    	P[key] = (v + k) / (n + k * c)

    #########################################
    return P
    


#--------------------------
def class_prior(Y):
    '''
        Estimate the prior probability of Class labels: P(Class=y).
        Here we assume this is binary classification problem.
        Input:
            Y : the labels of training instances, an integer numpy vector of length n. 
                Here n is the number of training instances. Each Y[i] = 0 or 1. 
        Output:
            PY: the prior probability of each class, a numpy vector of length c.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    PY = np.array([Counter(Y).get(i,0)/len(Y) for i in range(2)])
    #########################################
    return PY


#--------------------------
def conditional_prob(X,Y,k=1):
    '''
        Estimate the conditional probability of P(X=x|Class=y) for each value of attribute X given each class.
        Input:
            X : the values of one attribute for training instances, an integer numpy vector of length n. 
                n is the number of training instances. 
                Here we assume X is a binary variable, with 0 or 1 values. 
            Y : the labels of training instances, an integer numpy vector of length n. 
                Each Y[i] = 0,1
            k: the parameter of Laplace smoothing, an integer, denoting the number of imagined instances observed for each possible value of X given each value of Y.
        Output:
            PX_Y: the probability of P(X|Class), a numpy array (matrix) of shape 2 by 2.
                  PX_Y[i,j] represents the probability of X=j given the class label is i.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    X1 = [X[i] for i in range(len(Y)) if Y[i] == 0]
    X2 = [X[i] for i in range(len(Y)) if Y[i] == 1]
    PX_Y = np.concatenate((np.mat(prob_smooth(X1,2,k)),np.mat(prob_smooth(X2,2,k))), axis=0).getA()
    #########################################
    return PX_Y


#--------------------------
def train(X,Y,k=1):
    '''
        Training the model parameters on a training dataset.
        Input:
            X : the values of attributes for training instances, an integer numpy matrix of shape p by n. 
                p is the number of attributes. 
                n is the number of training instances. 
                Here we assume X is binary-valued, with 0 or 1 elements. 
            Y : the labels of training instances, an integer numpy vector of length n. 
                Each Y[i] = 0,1
            k: the parameter of Laplace smoothing, an integer, denoting the number of imagined instances observed for each possible value. 
        Output:
            PX_Y: the estimated probability of P(X|Class), a numpy array of shape p by 2 by 2.
                  PX_Y[i,j,k] represents the probability of the i-th attribute to have value k given the class label is j.
            PY: the estimated prior probability distribution of the class labels, a numpy vector of length 2.
                Each PY[i] is the estimated probability of the i-th class.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    p = X.shape[0]
    PX_Y = np.zeros((p,2,2))
    for i in range(p):
    	PX_Y[i] = conditional_prob(X[i],Y,k)
    PY = class_prior(Y)

    #########################################
    return PX_Y, PY


#--------------------------
def inference(X,PY, PX_Y):
    '''
        Given a trained model, predict the label of one test instance in the test dataset.
        Input:
            X : the values of attributes for one test instance, an integer numpy vector of length p. 
                p is the number of attributes. 
                Here we assume X is binary-valued, with 0 or 1 elements. 
            PX_Y: the estimated probability of P(X|Class), a numpy array of shape p by 2 by 2.
                  PX_Y[i,j,k] represents the probability of the i-th attribute to have value k given the class label is j.
            PY: the estimated prior probability distribution of the class labels, a numpy vector of length 2.
                Each PY[i] is the estimated probability of the i-th class.
        Output:
            Y: the predicted class label, an integer scalar of value 0 or 1.
            P: the probability P(class | X), a float array of length 2.
                P[i] is the probability of the instance X in the i-th class.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    P_CX = np.ones(2)
    for i in range(len(PX_Y)):
    	P_CX *= np.array(PX_Y[i])[:,X[i]]
    P_CX = PY * P_CX
    P = np.array([i/P_CX.sum() for i in P_CX])
    Y = np.argmax(P)

    #########################################
    return Y, P


#--------------------------
def predict(X,PY, PX_Y):
    '''
        Given a trained model, predict the labels of test instances in the test dataset.
        Input:
            X : the values of attributes for test instances, an integer numpy matrix of shape p by n. 
                p is the number of attributes. 
                n is the number of test instances. 
                Here we assume X is binary-valued, with 0 or 1 elements. 
            PX_Y: the estimated probability of P(X|Class), a numpy array of shape p by 2 by 2.
                  PX_Y[i,j,k] represents the probability of the i-th attribute to have value k given the class label is j.
            PY: the estimated prior probability distribution of the class labels, a numpy vector of length 2.
                Each PY[i] is the estimated probability of the i-th class.
        Output:
            Y: the predicted class labels, a numpy vector of length n.
               Each Y[i] is the predicted label of the i-th instance.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    n = X.shape[1]
    Y = np.zeros(n)
    for i in range(n):
    	Y[i] = inference(X[:,i],PY,PX_Y)[0]

    #########################################
    return Y

