##### Libraries #####
import numpy as np
import random as rd
##### Modules #####
from initial_data import data
from parameters import parameters
from polynomial_dictionary_learning import polynomial_dictionary_learning
##### Seeds #####
np.random.seed(1)
rd.seed(1)

##### Main #####
### Parameters ###
params = parameters()
### Data importation ###
signal = data('testdata.mat',max(params.deg_subdicos))
### Polynomial Dictionary Learning ###
dictionary, coef_matrix, alpha, result = polynomial_dictionary_learning(signal, params)
print alpha
