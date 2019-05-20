import math
import numpy as np
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 1: Linear Regression (Maximum Likelihood)
    In this problem, you will implement the linear regression method based upon maximum likelihood (least square).
    w'x + b = y
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
    Note: please don't use any existing package for linear regression problem, implement your own version.
'''

#--------------------------
def Terms_and_Conditions():
    ''' 
        By submiting this homework or changing this function, you agree with the following terms:
       (1) Not sharing your code/solution with any student before and after the homework due. For example, sending your code segment to another student, putting your solution online or lending your laptop (if your laptop contains your solution or your dropbox automatically copied your solution from your desktop computer and your laptop) to another student to work on this homework will violate this term.
       (2) Not using anyone's code in this homework and building your own solution. For example, using some code segments from another student or online resources due to any reason (like too busy recently) will violate this term. Changing other's code as your solution (such as changing the variable names) will also violate this term.
       (3) When discussing with any other student about this homework, only discuss high-level ideas or use pseudo-code. Don't discuss about the solution at the code level. For example, two students discuss about the solution of a function (which needs 5 lines of code to solve) and they then work on the solution "independently", however the code of the two solutions are exactly the same, or only with minor differences (variable names are different). In this case, the two students violate this term.
      All violations of (1),(2) or (3) will be handled in accordance with the WPI Academic Honesty Policy.  For more details, please visit: https://www.wpi.edu/about/policies/academic-integrity/dishonesty
      Note: we will use the Stanford Moss system to check your code for code similarity. https://theory.stanford.edu/~aiken/moss/
      Historical Data: in one year, we ended up finding 25% of the students in that class violating this term in their homework submissions and we handled ALL of these violations according to the WPI Academic Honesty Policy. 
    '''
    #*******************************************
    # CHANGE HERE
    Read_and_Agree = True  #if you have read and agree with the term above, change "False" to "True".
    #*******************************************
    return Read_and_Agree
 

#--------------------------
def compute_Phi(x,p):
    '''
        Let's start with a simple dataset. This dataset (polynomial curve fitting) has been used in many machine learning textbooks.
        In this dataset, we create a feature matrix by constructing multiple features from one single feature.
        For example, suppose we have a data example, with an input feature of value 2., we can create a 4-dimensional feature vector (when p=4) 
        using the different polynoials of the input like this: ( 1., 2., 4., 8.). 
        Here the first dimension is the 0-th polynoial of 2., 2^0 = 1.
        The second dimension is the 1st polynoial of 2., 2^1 = 2
        The third dimension is the 2nd polynoial of 2., 2^2 = 4
        The third dimension is the 3rd polynoial of 2., 2^3 = 8
        Now in this function, x is a vector, containing multiple data samples. For example, x = [2,3,5,4] (4 data examples)
        Then, if we want to create a 3-dimensional feature vector (when p=3) for each of these examples, then we have a feature matrix Phi (4 by 3 matrix).
        [[1, 2,  4],
         [1, 3,  9],
         [1, 4, 16],
         [1, 5, 25]]
        In this function , we need to compute the feature matrix (or design matrix) Phi from x for polynoial curve fitting problem. 
        We will construct p polynoials of x as the p features of the data samples. 
        The features of each sample, is x^0, x^1, x^2 ... x^(p-1)
        Input:
            x : a vector of samples in one dimensional space, a numpy vector of shape n by 1.
                Here n is the number of samples.
            p : the number of polynomials/features
        Output:
            Phi: the design/feature matrix of x, a numpy matrix of shape (n by p).
                 The i-th column of Phi represent the i-th polynomials of x. 
                 For example, Phi[i,j] should be x[i] to the power of j.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    Phi = np.zeros((x.shape[0],p))
    for i in range(x.shape[0]):
    	for j in range(p):
    		Phi[i,j] = x[i,0] ** j
    Phi = np.mat(Phi)

    #########################################
    return Phi 


#--------------------------
def least_square(Phi, y):
    '''
        Fit a linear model on training samples. Compute the paramter w using Maximum likelihood (equal to least square).
        Input:
            Phi: the design/feature matrix of the training samples, a numpy matrix of shape n by p
                Here n is the number of training samples, p is the number of features
            y : the sample labels, a numpy vector of shape n by 1.
        Output:
            w: the weights of the linear regression model, a numpy float vector of shape p by 1. 
        Hint: you could use np.linalg.inv() to compute the inverse of a matrix
              The problem can be solved using 1 line of code.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    w = np.linalg.inv(Phi.T * Phi) * Phi.T * y

    #########################################
    return w 



#--------------------------
def ridge_regression(Phi, y, alpha=0.001):
    '''
        Fit a linear model on training samples. Compute the paramter w using Maximum posterior (equal to least square with L2 regularization).
        min_w sum_i (y_i - Phi_i * w)^2/2 + alpha * w^T * w 
        Input:
            Phi: the design/feature matrix of the training samples, a numpy matrix of shape n by p
                Here n is the number of training samples, p is the number of features
            y : the sample labels, a numpy vector of shape n by 1.
            alpha: the weight of the L2 regularization term, a float scalar.
        Output:
            w: the weights of the linear regression model, a numpy float vector of shape p by 1. 
        Hint: you could use np.linalg.inv() to compute the inverse of a matrix. 
              The problem can be solved using 2 lines of code.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    I = np.eye(Phi.shape[1],Phi.shape[1])
    w = np.linalg.inv(Phi.T * Phi + alpha * I) * Phi.T * y 

    #########################################
    return w 

