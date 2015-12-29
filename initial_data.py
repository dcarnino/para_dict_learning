##### Libraries #####
import scipy.io
import scipy.linalg
import numpy as np

##### Classes #####
### Class describing the data
class data:

    ### load the data
    def __init__(self, mat_file='', max_power=20):
        # get initial data
        if mat_file != '':
            self.get_info_from_mat_file(mat_file)
        else:
            # do everything with synthetic graph and signal to match get_info_from_dictio format
            pass
        # get parameters relative to laplacian matrix
        self.get_info_laplacian()
        self.get_power_laplacian(max_power)



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

##### Main #####
if __name__ == "__main__":
    deg_subdicos = [20]*4
    example = data('testdata.mat',max(deg_subdicos))
    print example.sorted_laplacian_eigvals
