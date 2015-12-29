##### Libraries #####
import numpy as np
from cvxopt import matrix, solvers

##### Functions #####
### Quadratic program using interior point method
def coefficient_update_interior_point(coef_matrix, data, params):

    ## Compute matrices of quadratic optimization argmin formula
    # initialize train_signals
    Y = data.train_signals
    # initialize matrices bounds and iterators
    N = params.nb_nodes #nb_nodes
    N_list = range(N)
    tmp, M = Y.shape #nb_signals
    M_list = range(M)
    S = params.nb_subdicos #nb_subdicos
    S_list = range(S)
    K_list = [range(K) for K in params.deg_subdicos]
    maxK_list = params.deg_subdicos
    SK = sum(maxK_list)
    # initialize laplacian relatives
    laplacian_powers = data.laplacian_powers
    # initialize matrices to zeros
    PPT = np.zeros((SK,SK))
    YPT = np.zeros((1,SK))
    # loop over n, m, and s
    for n in N_list:
        for m in M_list:
            Pnm = np.zeros((SK,1))
            sk = 0
            for s in S_list:
                Pnm[sk:sk+maxK_list[s],:] = [laplacian_powers[k][n,:].dot(coef_matrix[s*N:(s+1)*N,m]) for k in K_list[s]]
                sk += maxK_list[s]
            PPTnm = Pnm.dot(Pnm.T)
            YPTnm = Y[n,m]*Pnm.T
            PPT += PPTnm
            YPT += YPTnm
    # here they are
    P_mat = PPT + params.mu_tradeoff*np.identity(SK)
    q_mat = YPT.T

    ## Compute matrices of quadratic optimization constraints
    B = data.eigval_power_mat
    B1 = np.kron(np.identity(S),B)
    B2 = np.kron(np.ones((1,S)),B)
    O1 = np.ones((B1.shape[0],1))
    O2 = np.ones((B2.shape[0],1))
    # bounds params
    c = params.spectral_cst
    epsilon1 = params.epsilon1
    epsilon2 = params.epsilon2
    # here they are
    G_mat = np.vstack((B1,-B1,B2,-B2))
    #G_mat = np.vstack((B1,B2,-B2))
    h_mat = np.vstack((c*O1,-0.000001*O1,(c+epsilon2)*O2,-(c-epsilon1)*O2))
    #h_mat = np.vstack((c*O1,(c+epsilon2)*O2,-(c-epsilon1)*O2))

    ## Set matrices for cvxopt
    P_matd = matrix(P_mat, tc='d')
    q_matd = matrix(q_mat, tc='d')
    G_matd = matrix(G_mat, tc='d')
    h_matd = matrix(h_mat, tc='d')

    ## run interior point method cvxopt solver ('mosek' solver can be used if installed)
    sol = solvers.qp(P_matd,q_matd,G_matd,h_matd)#,solver='mosek')
    alpha = sol['x']

    return alpha
