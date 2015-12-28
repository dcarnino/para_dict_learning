##### Libraries #####


##### Main #####

### Parameters ###
nb_nodes = 100 #N
nb_subdicos = 4 #S
nb_atoms = nb_nodes * nb_subdicos #J
deg_subdicos = [20]*nb_subdicos #K
sparsity = 4 #T0
spectral_cst = 1 #c
epsilon1 = 0.02
epsilon2 = 0.02 #we assume both epsilons have the same value
mu_tradeoff = 10**(-4) #polynomial regularizer parameter
