#-------------------------------------------------------------------------
# Note: please don't use any additional package except the following packages
import numpy as np
from scipy.special import psi,polygamma
from scipy.linalg import inv
#-------------------------------------------------------------------------
'''
    Problem 3: LDA (Latent Dirichlet Allocation) using Variational EM method
    In this problem, you will implement the Latent Dirichlet Allocation to model text data.
    You could test the correctness of your code by typing `nosetests -v test3.py` in the terminal.

    Notations:
            ---------- dimensions ------------------------
            m: the number of data documents, an integer scalar.
            n: the number of words in each document, an integer scalar.
            p: the number of all possible words (the size of the vocabulary), an integer scalar.
            k: the number of topics, an integer scalar
            ---------- model parameters ----------------------
            alpha: the parameter of the Dirichlet distribution for generating topic-mixture for each document, a float scalar (alpha>0).
                    Here we are assuming the parameters in all dimensions to have the same value.
            beta: the parameters of word distribution on k topics, a numpy float matrix of shape k by p. 
                Each element beta[i] represents the vector of word probabilitis in the i-th topic. 
            -----------------------------------------------
'''

#--------------------------
def variational_inference(w,beta, alpha=1., n_iter=100):
    '''
        Given a document, find the optimal values of the variational parameters: gamma and phi, using mean-field variational inference.
        Input:
            w:  the vector word ids of a document, an integer numpy vector of length n. 
                n: the number of words in the document, an integer scalar.
            beta: the current estimation of parameters for word distribution on k topics, a numpy float matrix of shape k by p. 
                p: the number of all possible words (the size of the vocabulary), an integer scalar.
                k: the number of topics, an integer scalar
                Each element beta[i] represents the vector of word probabilitis in the i-th topic. 
            alpha: the parameter of the Dirichlet distribution for generating topic-mixture for each document, a float scalar (alpha>0).
                    Here we are assuming the parameters in all dimensions to have the same value.
            n_iter: the number of iterations for iteratively updating gamma and phi. 
        Output:
            gamma:  the optimal value for gamma, a numpy float vector of length k. 
            phi:  the optimal values for phi, a numpy float matrix of shape n by k.
                Here k is the number of topics.
        Hint: you could use the psi() in scipy package to compute digamma function
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    N = len(w)
    k,p = beta.shape
    # initialization
    phi = np.ones((N,k)) / k
    gamma = np.ones(k) * (alpha + N / k)
    for _ in range(n_iter):
        for n in range(N):
            for i in range(k):
                phi[n,i] = beta[i,w[n]] * np.exp(psi(gamma[i]))
            phi[n] /= phi[n].sum()
        gamma_new = alpha + np.sum(phi, axis= 0)
        if np.allclose(gamma_new, gamma):
            break
        else:
            gamma = gamma_new

    #########################################
    return gamma, phi

#--------------------------
def E_step(W,beta, alpha=1., n_iter=100):
    '''
        Infer the optimal values for variational parameters on all documents: phi and gamma.
        Input:
            W:  the document matrix, an integer numpy matrix of shape m by n. 
                Here m is the number of text documents, an integer scalar.
                n: the number of words in each document, an integer scalar.
            beta: the current estimation of parameters for word distribution on k topics, a numpy float matrix of shape k by p. 
                p: the number of all possible words (the size of the vocabulary), an integer scalar.
                k: the number of topics, an integer scalar
                Each element beta[i] represents the vector of word probabilitis in the i-th topic. 
            alpha: the parameter of the Dirichlet distribution for generating topic-mixture for each document, a float scalar (alpha>0).
                    Here we are assuming the parameters in all dimensions to have the same value.
            n_iter: the number of iterations for iteratively updating gamma and phi. 
        Output:
            gamma:  the optimal gamma values for all documents, a numpy float matrix of shape m by k. 
            phi:  the optimal phi values for all documents, a numpy float tensor of shape m by n by k. 
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    m,n = W.shape
    k,p = beta.shape
    gamma = np.zeros((m,k))
    phi = np.zeros((m,n,k))
    for i in range(m):
        gamma[i], phi[i] = variational_inference(W[i],beta,alpha,n_iter)

    #########################################
    return gamma, phi 



#--------------------------
def update_beta(W, phi,p):
    '''
        update beta based upon the new values of the variational parameters. 
        Input:
            W:  the document matrix, a float numpy matrix of shape m by n. 
                Here m is the number of text documents, an integer scalar.
            phi:  the optimal phi values for all documents, a numpy float tensor of shape m by n by k. 
            p: the number of all possible words in the vocabulary.
        Output:
            beta: the updated estimation of parameters for word distribution on k topics, a numpy float matrix of shape k by p. 
                Each element beta[i] represents the vector of word probabilitis in the i-th topic. 
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    m,n,k = phi.shape
    beta = np.zeros((k,p))
    for d in range(m):
        for i in range(k):
            for j in range(p):
                W_temp = W.copy()
                for n_id in range(n):
                    if W[d,n_id] == j:
                        W_temp[d,n_id] = 1
                    else:
                        W_temp[d,n_id] = 0
                beta[i,j] += (phi[d,:,i] * W_temp[d]).sum()
    beta /= np.sum(beta, axis=1, keepdims=True)

    #########################################
    return beta 


#------------------------------------------------------------------------------
#  Newton's Method (for computing Alpha)
#------------------------------------------------------------------------------

'''
    Let's first practise with a simpler case: Newton's method in one dimensional space.
'''

#--------------------------
def newtons_1d(f,df,x,tol=1e-2):
    '''
        compute the root of function f using Newton's method.
        Input:
            f: the function f(x), a python function that takes a scalar x as input.
            df: the first derivative of function f(x), a python function that takes a scalar x as input.
            x: the initial solution, a float scalar.
            tol: the tolerance of error, a float scalar. 
                If the function values |f(x)| < tol  then stop the iteration.
        Output:
            x: a root of f(x), a float scalar 
        Hint: you could solve this problem using 2 lines of code.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    while np.abs(f(x)) >= tol:
        x = x - f(x) / df(x)

    #########################################
    return x

#--------------------------
def min_newtons_1d(f,df,d2f,x,tol=1e-2):
    '''
        compute the minimium value of function f using Newton's method (1 dimensional input)
        Input:
            f: the function f(x), a python function that takes a scalar x as input.
            df: the first derivative of function f(x), a python function that takes a scalar x as input.
            d2f: the second derivative of function f(x), a python function that takes a scalar x as input.
            x: the initial solution, a float scalar.
            tol: the tolerance of error, a float scalar. 
                If the function values |df(x)| < tol  then stop the iteration.
        Output:
            x: a local optimal solution, where f(x) has a minimium value, a float scalar 
            v: a local minimium value of f(x), a float scalar 
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    while np.abs(df(x)) >= tol:
        x = x - df(x) / d2f(x)
    v = f(x)
    #########################################
    return x, v 


#--------------------------
def newtons(f,df,x,tol=1e-2):
    '''
        compute the root of function f using Multi-variate Newton's method (p dimensional space).
        Input:
            f: the function f(x), a python function that takes a p-dimensional vector x as input and output a p-dimensional vector.
            df: the first derivative of function f(x), a python function that takes a p-dimensional vector x as input and output a pXp matrix (Jacobian matrix)
            x: the initial solution, a float p-dimensional vector.
            tol: the tolerance of error, a float scalar. 
                If the function values |f(x)| < tol  then stop the iteration. You could use np.allclose() function and set 'atol=tol'.
        Output:
            x: a root of f(x), a float p-dimensional vector 
        Hint: you could use np.linalg.inv() to compute the inverse of a matrix
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    while not np.allclose(f(x),0,atol=tol):
        x = x - np.dot(np.linalg.inv(df(x)), f(x))
        #x = x - np.squeeze(np.dot(np.linalg.inv(df(x)), f(x)[:,None]).T)

    #########################################
    return x

#--------------------------
def newtons_linear(f,h,z,x,tol=1e-2):
    '''
        compute the root of function f using Linear-time Multivariate Newton's method (p dimensional space).
        Input:
            f: the function f(x), a python function that takes a p-dimensional vector x as input and output a p-dimensional vector.
            df: the first derivative of function f(x), a python function that takes a p-dimensional vector x as input and output a pXp matrix (Jacobian matrix)
                Here Jacobian matrix is represented as diag(h) + 1z1^T
                h: the diagonal elements of Jacobian matrix, a python function outputing a float vector of length p.
                z: the constant background of Jacobian matrix, a python function outputing  a float scalar.
            x: the initial solution, a float p-dimensional vector.
            tol: the tolerance of error, a float scalar. 
                If the function values |f(x)| < tol  then stop the iteration. You could use np.allclose() function and set 'atol=tol'.
        Output:
            x: a root of f(x), a float p-dimensional vector 
        Hint: you could use np.linalg.inv() to compute the inverse of a matrix
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    while not np.allclose(f(x),0,atol=tol):     
        denomi = (1/z(x) + (1/h(x)).sum())
        c = (f(x) / h(x)).sum() / denomi
        x = x - (f(x) - c) / h(x)

    #########################################
    return x


#------------------------------------------------------------------------------
'''
    Let's use Newton's method to compute alpha 
    For simplicity, we assume alpha is a symmetric prior: alpha vector has the same value on all topics.
    So this problem can be solved by one-dimension Newton's method.
    (For assymetric priors, you will need to use linear-time Newton's method to compute alpha.)
'''

#--------------------------
def compute_df(alpha,gamma):
    '''
        compute the first derivative of ELBO function w.r.t. alpha 
        Here we assume alpha is a symmetric prior, and the output  is a sum of the first derivative values on all topics.
        Input:
            alpha: the current estimate of parameter alpha in the Dirichlet distribution for generating topic-mixture for each document, a float scalar (alpha>0).
                    Here we assume alpha is a symmetric prior, where alpha in all topics have the same value.
            gamma:  the gamma values for all documents, a numpy float matrix of shape m by k. 
                    Here k is the number of topics,  m is the number of documents.
        Output:
            g: the first derivative of ELBO function w.r.t. alpha, a float scalar 
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    M, k = gamma.shape
    gamma0 = np.sum(gamma, axis = 1)
    df_m = np.zeros(k)
    for i in range(k):
        df_m[i] = M * (psi(k*alpha)-psi(alpha)) + sum([(psi(gamma[d,i]) - psi(gamma0[d])) for d in range(M)])
    df = df_m.sum()

    #########################################
    return df 

#--------------------------
def compute_d2f(alpha,m,k):
    '''
        compute the second derivative of ELBO function w.r.t. alpha.
        Here we assume alpha is a symmetric prior, and the output  is a sum of all the values in Hessian matrix.
        Input:
            alpha: the current estimate of parameter alpha in the Dirichlet distribution for generating topic-mixture for each document, a float scalar (alpha>0).
                    Here we assume alpha is a symmetric prior, where alpha in all topics have the same value.
            m: the number of documents
            k: the number of topics
        Output:
            d2f: the second derivative of ELBO function w.r.t. alpha, a float scalar 
        Hint: you could use the polygamma() in scipy package to compute first derivative of the digamma function
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    d2f = m * k * (k * polygamma(1,k*alpha) - polygamma(1,alpha))
    # H_M = np.zeros((k,k))
    # for i in range(k):
    #     for j in range(k):
    #         kronecker = 1 if j == i else 0
    #         H_M[i,j] = kronecker * m * polygamma(1,alpha) - polygamma(1,k*alpha)
    # d2f = H_M.sum()

    #########################################
    return d2f 



#--------------------------
def update_alpha(alpha,gamma,tol=1e-2):
    '''
        update alpha using linear-time Newton method.
        Input:
            alpha: the current estimate of parameter alpha in the Dirichlet distribution for generating topic-mixture for each document, a float scalar (alpha>0).
                    Here we assume alpha is a symmetric prior, where alpha in all topics have the same value.
            gamma:  the gamma values for all documents, a numpy float matrix of shape m by k. 
            tol: the tolerance of error, a float scalar. 
                If the function values |df(x)| < tol  then stop the iteration. 
        Output:
            alpha: the updated estimation of parameters alpha, a float scalar
        Hint: alpha is a variable with value >0, but log alpha is a variable that can take any real-value. We need to apply Newton's method on log alpha, instead of alpha.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    m,k = gamma.shape
    # if alpha > 1:
    #     alpha = 1 / alpha
    alpha = np.log(alpha)
    while not np.allclose(compute_df(np.exp(alpha),gamma),0,atol=tol):
        df = compute_df(np.exp(alpha),gamma)
        d2f = compute_d2f(np.exp(alpha),m,k)
        alpha = alpha - df/(d2f*np.exp(alpha)+df) 
        #alpha = alpha - df / d2f     
    alpha = np.exp(alpha)

    # while not np.allclose(compute_df(alpha,gamma),0,atol=tol):
    #     df = compute_df(alpha,gamma)
    #     d2f = compute_d2f(alpha,m,k)
    #     alpha = alpha - df / d2f
    #########################################
    return alpha


#--------------------------
def M_step(W,phi,p,alpha,gamma,tol_alpha=1e-2):
    '''
        M step of the EM algorithm. 
        Input:
            W:  the document matrix, a float numpy matrix of shape m by n. 
                Here m is the number of text documents, an integer scalar.
            phi:  the optimal phi values for all documents, a numpy float tensor of shape m by n by k. 
            p: the number of all possible words in the vocabulary.
            alpha: the current estimate of parameter alpha in the Dirichlet distribution for generating topic-mixture for each document, a float scalar (alpha>0).
                    Here we assume alpha is a symmetric prior, where alpha in all topics have the same value.
            gamma:  the gamma values for all documents, a numpy float matrix of shape m by k. 
            tol_alpha: the tolerance of error for Newton's method, a float scalar. 
                If the function values |df(x)| < tol  then stop the iteration. 
        Output:
            alpha: the updated estimation of parameters alpha, a float scalar
            beta: the updated estimation of parameters for word distribution on k topics, a numpy float matrix of shape k by p. 
                Each element beta[i] represents the vector of word probabilitis in the i-th topic. 
    '''
    alpha= update_alpha(alpha,gamma,tol_alpha)
    beta = update_beta(W,phi,p)
    return alpha,beta



#--------------------------
def LDA(W,k=3,p=100,alpha=.1,n_iter_var=100,n_iter_em=10,tol_alpha=1e-2):
    '''
        Variational EM algorithm for LDA. 
        Input:
            W:  the document matrix, a float numpy matrix of shape m by n. 
                Here m is the number of text documents, an integer scalar.
            p: the number of all possible words (the size of the vocabulary), an integer scalar.
            k: the number of topics, an integer scalar
            alpha: the initial value of the alpha parameter of the Dirichlet distribution for generating topic-mixture for each document, a float scalar (alpha>0).
            n_iter_var: the number of iterations in variational inference (E-step). 
            n_iter_em: the number of iterations in EM
            tol_alpha: the tolerance of error for Newton's method, a float scalar. 
                If the function values |df(x)| < tol  then stop the iteration. 
        Output:
            alpha: the updated estimation of parameters alpha, a float scalar
            beta: the updated estimation of parameters for word distribution on k topics, a numpy float matrix of shape k by p. 
                Each element beta[i] represents the vector of word probabilitis in the i-th topic. 
            gamma:  the optimal value for gamma, a numpy float vector of length k. 
            phi:  the optimal values for phi, a numpy float matrix of shape n by k.
    '''
    # initialize beta (for testing purpose)
    beta = np.arange(float(k*p)).reshape((k,p))+1.
    for i in range(k):
        beta[i] = beta[i] /sum(beta[i])

    for _ in range(n_iter_em):
        #########################################
        ## INSERT YOUR CODE HERE
        gamma,phi = E_step(W,beta,alpha,n_iter_var)
        alpha,beta = M_step(W,phi,p,alpha,gamma,tol_alpha)
        #########################################
    return alpha,beta,gamma, phi 


