##### Libraries #####
import numpy as np
##### Modules #####
from omp import omp_non_normalized_atoms
from interior_point import coefficient_update_interior_point
from plot_functions import plot_kernels

##### Functions #####
### learn a polynomial dictionary with parameters params from data data
def polynomial_dictionary_learning(data, params):
    # initialize lists
    result = []
    total_error = []
    # initialize dictionary
    if params.init_method == 'random_kernels':
        dictionary = initialize_dictionary(data, params)
    # plot intial kernels
    if params.plot_kernels:
        pass#plot_kernels(data.init_alpha, data, params)
    # main loop of algorithm
    for iter in range(params.nb_iter):
        # sparse coding step (OMP)
        coef_matrix = omp_non_normalized_atoms(dictionary, data.train_signals, params.sparsity)
        # dictionary update step (interior point)
        alpha = coefficient_update_interior_point(coef_matrix, data, params)
        result.append(alpha)
        # plot kernels
        if params.plot_kernels:
            plot_kernels(alpha, data, params)
        # update dictionary
        update_dictionary(dictionary, alpha, data, params)
        # show progress
        if iter>0:
            current_error = np.sqrt(np.sum(np.power((data.train_signals-dictionary.dot(coef_matrix)),2))/data.train_signals.size)
            total_error.append(current_error)
    return dictionary, coef_matrix, alpha, result


### initialize a dictionary from parameters params and data data
def initialize_dictionary(data, params):
    dictionary = np.zeros((params.nb_nodes, params.nb_atoms))
    for s in range(params.nb_subdicos):
        lambdas = params.spectral_cst*np.random.rand(params.nb_nodes)
        lambdas_mat = np.diag(lambdas)
        dictionary[:,s*params.nb_nodes:(s+1)*params.nb_nodes] = data.eigvect_laplacian_mat.dot(lambdas_mat.dot(np.matrix(data.eigvect_laplacian_mat).getH()))
    return dictionary

### update the dictionary from newly computed alphas
def update_dictionary(dictionary, alpha, data, params):
    r = 0
    for s in range(params.nb_subdicos):
        Ds = np.zeros(params.nb_nodes)
        for k in range(params.deg_subdicos[s]):
            Ds = Ds + alpha[s+r]*data.laplacian_powers[s]
        r = sum(params.deg_subdicos[:s+1]) + s+1
        dictionary[:,s*params.nb_nodes:(s+1)*params.nb_nodes] = Ds
