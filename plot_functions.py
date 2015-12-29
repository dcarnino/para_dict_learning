##### Libraries #####
import numpy as np
import matplotlib.pyplot as plt

##### Functions #####
### plot current kernels corresponding to alpha
def plot_kernels(alpha, data, params):
    # kernels computation
    g_ker = generating_kernels(alpha, data, params)
    # display plot
    fig, ax = plt.subplots()
    for s in range(params.nb_subdicos):
        scatter, = ax.plot(data.sorted_laplacian_eigvals, g_ker[:,s], '-o', alpha=0.5)
    ax.set_xlim((0,1.5))
    ax.set_xlabel('Eigenvalues of the Laplacian')
    ax.set_ylabel('Generating kernels')
    plt.title('Learned kernels')
    plt.show()

### compute generating kernels from alpha
def generating_kernels(alpha, data, params):
    g_ker = np.zeros((params.nb_nodes, params.nb_subdicos))
    N_list = range(params.nb_nodes)
    r = 0
    for s in range(params.nb_subdicos):
        for n in N_list:
            p = 0
            for k in range(params.deg_subdicos[s]):
                p += alpha[k+r]*data.eigval_powers[k][n]
            g_ker[n,s] = p
        r += params.deg_subdicos[s]
    return g_ker
