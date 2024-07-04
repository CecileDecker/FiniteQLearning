############################################################################################

import numpy as np
from tqdm import tqdm 
from scipy.optimize import minimize
import copy

# Classical Q learning #

def q_learning(X, A, r, P_0, alpha, x_0, eps_greedy = 0.05, Nr_iter = 1000, gamma_t_tilde = lambda t: 1/(t+1), Q_0 = None):
    """
    Parameters
    ----------
    X : numpy.ndarray
        A list or numpy array containing all states
    A : numpy.ndarray
        A list or numpy array containing all actions
    r : function
        Reward function r(x,a,y) depending on state-action-state.
    P_0 : function
        function P_0(x,a) that creates a new random variabe in dependence of state and action
    alpha : float
        Discounting rate.
    x_0 : numpy.ndarray
        the initial state.
    eps_greedy : float, optional
        Parameter for the epsilon greedy policy. The default is 0.05.
    Nr_iter : int, optional
        Number of Iterations. The default is 1000.
    gamma_t_tilde : function, optional
        learning rate. The default is lambda t: 1/(t+1).
    Q_0 : matrix, optional
        Initial value for the Q-value matrix. The default is None.

    Returns
    -------
    matrix
        The Q value matrix.
    """

    rng = np.random.default_rng()

    # Initialize with Q_0 (if any)
    Q = np.zeros([len(X), len(A)])
    if Q_0 is not None:
        Q = Q_0
    
    # Initialize the Visits matrix
    Visits = np.zeros([len(X), len(A)])

    # Catch A and X as lists type
    if np.ndim(A) > 1:
        A_list = A
    else:
        A_list = np.array([[a] for a in A])
    if np.ndim(X) > 1:
        X_list = X
    else:
        X_list = np.array([[x] for x in X])

    # Functions that catch the index of a in A
    def a_index(a):
        return np.flatnonzero((a == A_list).all(1))[0]
    def x_index(x):
        return np.flatnonzero((x == X_list).all(1))[0]
    
    # Define the f function
    def f(x, a, y):
        return r(x, a, y) + alpha * np.max(Q[x_index(y), :])

    # Define the a_t function
    def a_t(y):
        # eps_bound = 1-(t/Nr_iter)*(eps_greedy)
        eps_bound = eps_greedy
        unif      = np.random.uniform(0)
        return (unif > eps_bound) * A[np.argmax(Q[x_index(y), :])] + (unif <= eps_bound) * rng.choice(A)
    
    # Define the gamma_t function
    def gamma_t(j, v):
        return gamma_t_tilde(v) * j 
    
    # Set initial value
    Y_0 = x_0
    # Iterations of Q and Visits and States
    for t in tqdm(range(Nr_iter)):

        
        x , a        = Y_0, a_t(Y_0)
        Q_old        = copy.deepcopy(Q)
        j_xa         = 1                     # Because in the non-robust case, (x, a) is literally chosen to be equal to (Y_t, a_t(Y_))
        x_ind, a_ind = x_index(x), a_index(a)
        Y_1          = P_0(x, a) # Get the next state, with a bit of randomness

        # Do the update of Q
        Q[x_ind, a_ind] = Q_old[x_ind, a_ind] + gamma_t(j_xa, Visits[x_ind, a_ind]) * (f(x, a, Y_1) - Q_old[x_ind, a_ind])
        Visits[x_ind, a_ind] += 1            # Update the visits matrix
        Y_0 = Y_1                            # Update the state
    
    return Q, Visits
