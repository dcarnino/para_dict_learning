##### Libraries #####
import numpy as np
import sklearn.linear_model

##### Functions #####
### Orthogonal Matching Pursuit for sparse coding
def omp_non_normalized_atoms(dictionary, train_signals, sparsity):
    tmp, nb_signals = train_signals.shape
    nb_nodes, nb_atoms = dictionary.shape
    # normalize dictionary atoms
    norm_atoms = np.sqrt(np.sum(dictionary**2,axis=0))
    norm_atoms[norm_atoms==0] = 1
    norm_dictionary = dictionary/np.tile(norm_atoms,(nb_nodes,1))
    # compute sparse representation using OMP
    normed_coef_matrix = sklearn.linear_model.orthogonal_mp(norm_dictionary, train_signals, n_nonzero_coefs=sparsity, precompute=True)
    # renormalize coefficients
    coef_matrix = normed_coef_matrix/np.tile(np.matrix(norm_atoms).getH(),(1,nb_signals))
    return coef_matrix
