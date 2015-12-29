##### Libraries #####
import scipy.io
import scipy.linalg
import numpy as np
import matplotlib.pyplot as plt
##### Modules #####
from synthetic_graph import synthetic_graph

##### Classes #####
### Class describing the data
class data:

    ### load the data
    def __init__(self, params, mat_file=''):
        # get initial data
        if mat_file != '':
            self.get_info_from_mat_file(mat_file)
            # get parameters relative to laplacian matrix
            self.get_info_laplacian()
            self.get_power_laplacian(max(params.deg_subdicos))
        else:
            graph = synthetic_graph(params.nb_nodes, params.theta, params.kappa)
            self.get_info_from_graph(graph)
            # get parameters relative to laplacian matrix
            self.get_info_laplacian()
            self.get_power_laplacian(max(params.deg_subdicos))
            # draw alpha randomly in a gaussian
            self.init_alpha = np.random.normal(params.mean_alpha, params.sd_alpha, params.deg_kernels*params.nb_subdicos)
            self.generate_dictionary(params)
            self.generate_signals(params)
        if params.plot_kernels:
            self.plot_kernels(params)

    ### set attributes to their values in dictionary from mat_file
    def get_info_from_mat_file(self, mat_file):
        dictio = scipy.io.loadmat(mat_file)
        self.adjacency_mat = dictio['A']
        self.init_alpha = dictio['C']
        self.poly_dico = dictio['D']
        self.weight_mat = dictio['W']
        self.x_coords = dictio['XCoords']
        self.y_coords = dictio['YCoords']
        self.train_signals = dictio['TrainSignal']
        self.test_signals = dictio['TestSignal']

    ### set attributes relative to laplacian matrix
    def get_info_laplacian(self):
        deg_mat = np.diag(np.sum(self.weight_mat,axis=1))
        half_pow_deg_mat = scipy.linalg.sqrtm(np.linalg.inv(deg_mat))
        raw_laplacian_mat = deg_mat - self.weight_mat
        self.laplacian_mat = half_pow_deg_mat.dot(raw_laplacian_mat.dot(half_pow_deg_mat))
        eigvals, eigvects = np.linalg.eig(self.laplacian_mat)
        self.eigval_laplacian_mat = np.diag(eigvals)
        self.eigvect_laplacian_mat = eigvects
        self.sorted_laplacian_eigvals = np.sort(eigvals)[::-1]

    ### store attributes relative powers of the laplacian
    def get_power_laplacian(self, max_power):
        self.laplacian_powers = [np.linalg.matrix_power(self.laplacian_mat, k) for k in range(max_power)]
        self.eigval_powers = [self.sorted_laplacian_eigvals**k for k in range(max_power)]
        self.eigval_power_mat = np.array(self.eigval_powers).T

    ### get attributes from a graph
    def get_info_from_graph(self, graph):
        self.weight_mat = graph.weight_mat
        x_coords, y_coords = zip(*graph.coordinates)
        self.x_coords = np.array(x_coords)
        self.y_coords = np.array(y_coords)

    ### generate polynomial dictionary from graph and alphas
    def generate_dictionary(self, params):
        self.poly_dico = np.zeros((params.nb_nodes,params.nb_nodes*params.nb_subdicos))
        r = 0
        for s in range(params.nb_subdicos):
            Ds = np.zeros((params.nb_nodes,params.nb_nodes))
            for k in range(params.deg_kernels):
                Ds += self.init_alpha[k+r]*self.laplacian_powers[k]
            r += params.deg_kernels
            self.poly_dico[:,s*params.nb_nodes:(s+1)*params.nb_nodes] = Ds

    ### generate signals by combining randomly T0 or less atoms from dictionary
    def generate_signals(self, params):
        Y = np.zeros((params.nb_nodes,params.nb_total_signals))
        T0_list = range(params.sparsity)
        NS_list = range(params.nb_nodes*params.nb_subdicos)
        for m in range(params.nb_total_signals):
            T0_selected = np.random.choice(T0_list)
            atoms_selected = self.poly_dico[:,np.random.choice(NS_list,T0_selected+1)]
            linear_combination = np.random.choice(range(-10,11),(T0_selected+1,))
            Y[:,m] = atoms_selected.dot(linear_combination)
        self.train_signals = Y[:,0:params.nb_train_signals]
        self.test_signals = Y[:,params.nb_train_signals:params.nb_total_signals]

    def plot_kernels(self, params):
        # kernels computation
        g_ker = self.generating_kernels(params)
        # display plot
        fig, ax = plt.subplots()
        for s in range(params.nb_subdicos):
            scatter, = ax.plot(self.sorted_laplacian_eigvals, g_ker[:,s], '-o', alpha=0.5)
        ax.set_xlim((0,1.5))
        ax.set_xlabel('Eigenvalues of the Laplacian')
        ax.set_ylabel('Generating kernels')
        plt.title('Initial kernels')
        plt.show()

    ### compute generating kernels from alpha
    def generating_kernels(self, params):
        g_ker = np.zeros((params.nb_nodes, params.nb_subdicos))
        N_list = range(params.nb_nodes)
        K_list = range(params.deg_kernels)
        r = 0
        for s in range(params.nb_subdicos):
            for n in N_list:
                p = 0
                for k in K_list:
                    p += self.init_alpha[k+r]*self.eigval_powers[k][n]
                g_ker[n,s] = p
            r += params.deg_kernels
        return g_ker
