class parameters:
    def __init__(self):
        self.file_name = 'testdata.mat'
        #self.file_name = ''
        self.nb_nodes = 100 #N
        self.nb_subdicos = 4 #S
        self.nb_atoms = self.nb_nodes * self.nb_subdicos #J
        self.deg_subdicos = [20+1]*self.nb_subdicos #K
        self.sparsity = 4 #T0
        self.spectral_cst = 1 #c
        self.epsilon1 = 0.02
        self.epsilon2 = 0.02 #we assume both epsilons have the same value
        self.mu_tradeoff = 10**(-4) #polynomial regularizer parameter

        self.init_method = 'random_kernels'
        self.verbose = True
        self.nb_iter = 5
        self.plot_kernels = True

        self.theta = 0.9
        self.kappa = 0.5
        self.deg_kernels = 6
        self.mean_alpha = 0
        self.sd_alpha = 0.2
        self.nb_train_signals = 600
        self.nb_test_signals = 600
        self.nb_total_signals = self.nb_train_signals + self.nb_test_signals
