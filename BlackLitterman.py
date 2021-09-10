import numpy as np


def BlackLitterman(weights,delta,tau, sigmaR, P_matrix, Q_matrix, Omega_matrix):
    # Find the equilibrium expected return
    weights = weights.T
    expected_return = delta * np.dot(sigmaR, weights) 
    
    # Formula to calculate the new expected return with views incorporated
    r_1 = np.dot(tau * np.dot(sigmaR,np.transpose(P_matrix)),np.linalg.inv(tau * np.dot(np.dot(P_matrix,sigmaR),np.transpose(P_matrix))\
                                                                    +Omega_matrix))
    r_2 = Q_matrix - np.dot(P_matrix, expected_return)
    view_term = np.dot(r_1, r_2)
    
    # Incorporate view into expected return
    BL_return = expected_return + view_term
    
    # updated sigmaR with view
    s_1 = tau * sigmaR
    s_2 = np.dot(r_1, tau * np.dot(P_matrix, sigmaR))
    updated_sigmaR = sigmaR + s_1 - s_2
    
    # updated weights
    updated_weight = np.dot(np.linalg.inv(delta * updated_sigmaR), BL_return)
    
    return expected_return, BL_return, updated_sigmaR, updated_weight