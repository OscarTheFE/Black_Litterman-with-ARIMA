import numpy as np

def Portfolio_returns(expected_return,r,weight):
    excess_return = expected_return - r
    excess_return = excess_return.T
    return r + np.dot(excess_return, weight)

def Portfolio_std(sigmaR, weight):
    return np.sqrt(weight.T @ sigmaR @ weight)


def Tangency_Portfolio(expected_return,r,sigmaR):
    excess_return = expected_return - r
    excess_return = excess_return.T
    H_matrix = np.linalg.solve(sigmaR,excess_return)
    one_matrix = np.ones(np.size(sigmaR,1))
    tangency_weight = (1 / (np.dot(one_matrix.T,H_matrix))) * H_matrix
    # tangency_weight should be a vector with the tangency weights for each assets
    return tangency_weight
    
def Min_Var_portfolio(expected_return, r, sigmaR, portfolio_return):
    # call tangency weight
    tangency_weight = Tangency_Portfolio(expected_return, r, sigmaR)
    excess_return = expected_return - r
    excess_return = excess_return.T
    mv_cash = 1 - (portfolio_return - r)/(np.dot(excess_return, tangency_weight))
    mv_weight = (1 - mv_cash) * tangency_weight
    return mv_cash, mv_weight
                                          
def Maximum_Return_Portfolio(expected_return, r, sigmaR, portfolio_sigma):
    # Call the tangency portfolio
    tangency_weight = Tangency_Portfolio(expected_return, r, sigmaR)
    # Calculate the cash portion of maximum return portfolio
    excess_return = expected_return - r
    excess_return = np.transpose(excess_return)
    H_matrix = np.linalg.solve(sigmaR, excess_return)
    one_matrix = np.ones(np.size(sigmaR,1))
    sign = np.sign(np.dot(one_matrix.T,H_matrix))
    G_matrix = tangency_weight.T @ sigmaR @ tangency_weight
    mr_cash = 1 - ((portfolio_sigma * sign) / np.sqrt(G_matrix))
    mr_weight = (1 - mr_cash) * tangency_weight
    # mr_weight should be a vector while mr_cash is just a percentage
    return mr_cash, mr_weight