class parameters:
    def __init__(self):
        self.nb_nodes = 100 #N
        self.nb_subdicos = 4 #S
        self.nb_atoms = self.nb_nodes * self.nb_subdicos #J
        self.deg_subdicos = [20]*self.nb_subdicos #K
        self.sparsity = 4 #T0
        self.spectral_cst = 1 #c
        self.epsilon1 = 0.02
        self.epsilon2 = 0.02 #we assume both epsilons have the same value
        self.mu_tradeoff = 10**(-4) #polynomial regularizer parameter

        self.init_method = 'random_kernels'
        self.verbose = True
        self.nb_iter = 25
        self.plot_kernels = True
