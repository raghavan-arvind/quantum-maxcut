from random import randint, choice
from math import ceil
from scipy.optimize import minimize
import numpy as np
import sys
import pandas as pd
from qiskit import register, available_backends, QuantumCircuit, QuantumRegister, \
                    ClassicalRegister, execute
import Qconfig

register(Qconfig.APItoken, Qconfig.config["url"])

DEBUG = True
def debug(string):
    if DEBUG:
        sys.stdout.write(string)
        sys.stdout.flush()


class Graph():
    def __init__(self, N, randomize=True):
        ''' Initialize a random graph with N vertices. '''
        self.N = N
        self.E = 0
        self.adj = {n:dict() for n in range(N)}

        # For storing information about each run.
        self.currentScore = float('-inf')
        self.currentBest = ""
        self.runs = []
        
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

    def get_edges(self):
        ''' Get a list of all edges. '''
        edges = []
        for u in self.adj:
            for v in self.adj[u]:
                edges.append((u, v, self.adj[u][v]))
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

    def update_score(self, bitstring):
        ''' Scores the given bitstring and keeps track of best. '''
        score = self.get_score(bitstring)
        if score > self.currentScore:
            self.currentScore = score
            self.currentBest = bitstring
        return score

    def add_run(self, gamma, beta, expected_value):
        ''' Save the data from each run iteration. '''
        self.runs.append([len(self.runs), gamma, beta, expected_value])

    def save_results(self, filename):
        df = pd.DataFrame(self.runs, 
                columns=['Iter','Gamma', 'Beta', 'Expected Value']).set_index('Iter')
        df.to_csv(filename)

    def __str__(self):
        return "Graph with %d vertices %d edges.\nAdjacency List: %s" % (self.N, self.E, self.adj)

def get_expectation(x, graph, NUM_SHOTS=10):
    gamma, beta = x

    debug("Cost of Gamma: %s, beta: %s... " % (gamma, beta))

    # Construct quantum circuit.
    q = QuantumRegister(graph.N + graph.E)
    c = ClassicalRegister(graph.N)
    qc = QuantumCircuit(q, c)

    # Apply hadamard to all inputs.
    for i in range(graph.N):
        qc.h(q[i])

    # Apply V for all edges.
    for edge_ind, edge in enumerate(g.get_edges()):
        u, v, w = edge
        edge_ind += graph.N

        # Apply CNots.
        qc.cx(q[u], q[edge_ind])
        qc.cx(q[v], q[edge_ind]) 
        qc.u1(gamma*w, q[edge_ind]) 

        # Apply CNots.
        qc.cx(q[v], q[edge_ind])
        qc.cx(q[u], q[edge_ind])

    # Apply W to all vertices.
    for i in range(graph.N):
        qc.h(q[i])
        qc.u1(-2*beta, q[i])
        qc.h(q[i])


    # Measure the qubits.
    for i in range(graph.N):
        qc.measure(q[i], c[i])

    # Run the simluator.
    job = execute(qc, backend='ibmq_qasm_simulator', shots=NUM_SHOTS)
    results = job.result()
    result_dict = results.get_counts(qc)

    debug("done!\n")

    # Calculate the expected value of the candidate bitstrings.
    exp = 0
    for bitstring in result_dict:
        prob = np.float(result_dict[bitstring]) / NUM_SHOTS
        score = g.update_score(bitstring)

        debug("\t\t%s: %s\n" % (bitstring, score))

        # Expected value is the score of each bitstring times
        # probability of it occuring.
        exp += score * prob

    debug("\tExpected Value: %s\n" % (exp))
    debug("\tBest Found Solution: %s, %s\n" % (g.currentScore, g.currentBest))

    g.add_run(gamma, beta, exp)

    return -1*exp # -1 bc we want to minimize


if __name__ == '__main__':
    filename = "results.csv"

    print("-- Building Graph--")
    g = Graph(5)
    print(g)

    best, best_val = g.optimal_score()
    print("Optimal Solution: %s, %s" % (best, best_val[0]))
    #print("Best Found Solution: %s, %s" % (g.currentScore, g.currentBest))

    # do the thing
    gamma_start = np.pi
    beta_start = np.pi/2

    print("\n-- Starting optimization --")
    try:
        res = minimize(get_expectation, [gamma_start, beta_start], args=(g),
                options=dict(maxiter=2,disp=True), bounds=[(0, 2*np.pi), (0,np.pi)])
    except KeyboardInterrupt:
        debug("\nWriting to %s\n" % (filename))
        g.save_results(filename)
    finally:
        exit()

    print("-- Finished optimization  --\n")
    print("Gamma: %s, Beta: %s" % (res.x[0], res.x[1]))
    print("Final cost: %s" % (res.maxcv))

    best, best_val = g.optimal_score()
    print("Optimal Solution: %s, %s" % (best, best_val[0]))
    print("Best Found Solution: %s, %s" % (g.currentScore, g.currentBest))

    debug("\nWriting to %s\n" % (filename))
    g.save_results(filename)
    

