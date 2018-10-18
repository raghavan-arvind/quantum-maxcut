from random import randint, choice
from math import ceil
from scipy.optimize import minimize
import numpy as np
from qiskit import register, available_backends, QuantumCircuit, QuantumRegister, \
                    ClassicalRegister, execute

class Graph():
    def __init__(self, N, randomize=True):
        ''' Initialize a random graph with N vertices. '''
        self.N = N
        self.E = 0
        self.adj = {n:dict() for n in range(N)}
        
        # Randomly generate edges
        if randomize:
            self.randomize()

    def randomize(self):
        ''' Randomly generate edges for this graph. '''

        # Generate list of tuples for all possible directed edges.
        all_possible_edges = set([(x,y) for x in range(self.N) for y in range(self.N) if x != y])

        # Sanity check, ensuring we generated the correct number of edges.
        e_gen = len(all_possible_edges) / 2
        e_shd = self.N * (self.N-1) / 2
        assert e_gen == e_shd , "%d != %d" % (e_gen, e_shd)

        # Choose a random number of edges for this graph to have. 
        # Note, we stop at len/2 because we generated directed edges,
        # so each edge counts twice.
        num_edges = randint(1, len(all_possible_edges)/2)
        for i in range(num_edges):
            # Choose an edge, remove it and its directed complement from the list.
            e = choice(list(all_possible_edges))
            all_possible_edges.remove(e)
            all_possible_edges.remove(e[::-1])

            # Unpack tuple into vertex ints.
            u, v = int(e[0]), int(e[1])

            # Choose a random weight for each edge.
            weight = randint(1, 100)
            self.add_edge(u, v, weight)


    def add_edge(self, u, v, weight):
        ''' Add an edge to the graph. '''
        self.E += 1
        self.adj[u][v] = weight

    def get_edges():
        ''' Get a list of all edges. '''
        edges = []
        for u in adj:
            for v in adj[u]:
                edges.append((u, v, adj[u][v]))
        return edges

    def get_score(self,bitstring):
        ''' Score a candidate solution. '''
        assert len(bitstring) == self.N

        score = 0

        # For every edge u,v in the graph, add the weight
        # of the edge if u,v belong to different cuts
        # given this canddiate solution.

        for u in self.adj:
            for v in self.adj[u]:
                if bitstring[u] != bitstring[v]:
                    score += self.adj[u][v]
        return score

    def optimal_score(self):
        ''' 
        Returns (score, solutions) holding the best possible solution to the 
        MaxCut problem with this graph.
        '''
            
        best = 0
        best_val = []

        # Iterate over all possible candidate bitstrings
        # Note: the bitstrings from 0 - N/2 are symmetrically
        # equivalent to those above
        for i in range(ceil((2 ** self.N)/2)):
            # Convert number to 0-padded bitstring.
            bitstring = bin(i)[2:]
            bitstring = (self.N - len(bitstring)) * "0" + bitstring

            sc = self.get_score(bitstring)
            if sc > best:
                best = sc
                best_val = [bitstring]
            elif sc == best:
                best_val.append(bitstring)
        return best, best_val

    def edges_cut(self, bitstring):
        ''' Given a candidate solution, return the number of edges that this solution cuts. '''
        num = 0
        for u in self.adj:
            for v in self.adj[u]:
                if bitstring[u] != bitstring[v]:
                    num += 1
        return num
    
    def __str__(self):
        return "Graph with %d vertices %d edges.\nAdjacency List: %s" % (self.N, self.E, self.adj)

def get_expectation(x, graph):
    gamma, beta = x

    # Construct quantum circuit.
    q = QuantumRegister(graph.N + graph.E)
    c = ClassicalRegister(graph.N + graph.E)

    qc = QuantumCircuit(q, c)


    qc.measure(q, c)

    job = execute(qc, backend='ibmq_qasm_simulator', shots=1024)
    results = job.result()

    print(results.get_counts(qc))


if __name__ == '__main__':
    g = Graph(10)
    get_expectation([np.pi, np.pi], g)

    '''
    for _ in range(100):
        g = Graph(15)
        edges = g.E

        best_solution = g.optimal_score()[1][0]
        print(1.0 * g.edges_cut(best_solution) / edges)
    '''

    '''
    g = Graph(6, randomize=False)

    g.add_edge(0, 3, 100)
    g.add_edge(1, 4, 100)
    g.add_edge(2, 5, 100)

    g.add_edge(0, 1, 1)
    g.add_edge(0, 2, 1)
    g.add_edge(1, 2, 1)

    g.add_edge(3, 4, 1)
    g.add_edge(3, 5, 1)
    g.add_edge(4, 5, 1)

    print(g.optimal_score())
    best_solution = g.optimal_score()[1][0]
    print(1.0 * g.edges_cut(best_solution) / g.E)
    '''


