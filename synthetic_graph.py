##### Libraries #####
import numpy as np
import networkx as nx

##### Classes #####
### Class describing a synthetic graph
class synthetic_graph:

    ### initialize a graph randomly embedded in unit square
    def __init__(self, nb_nodes, theta, kappa):
        self.graph = nx.Graph()
        self.nodes = range(nb_nodes)
        self.coordinates = np.random.sample(size=(nb_nodes,2))
        for inode in self.nodes:
            for jnode in self.nodes:
                if inode != jnode:
                    if self.distance(inode, jnode) <= kappa:
                        self.graph.add_edge(inode, jnode, weight=self.weight_func(inode, jnode, theta))
        assert (nx.is_connected(self.graph)), "Graph carrying signal is not connected !"
        self.embed_signal()

    ### compute and return euclidean distance between inode and jnode
    def distance(self, inode, jnode):
        ix, iy = self.coordinates[inode]
        jx, jy = self.coordinates[jnode]
        return np.sqrt((jx-ix)**2+(jy-iy)**2)

    ### compute and return weight of the edge between inode and jnode
    def weight_func(self, inode, jnode, theta):
        return np.exp(-(self.distance(inode,jnode)**2)/(2.*theta**2))


##### Main #####
if __name__ == "__main__":
    nb_nodes, theta, kappa = 100, 0.9, 0.5
    synth_graph = synthetic_graph(nb_nodes, theta, kappa)
