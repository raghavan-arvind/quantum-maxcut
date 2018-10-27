import time
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import Qconfig
from tqdm import tqdm
from random import randint, choice, uniform
from math import ceil
from scipy.optimize import minimize
from statistics import stdev, mean
from qiskit import register, available_backends, QuantumCircuit, QuantumRegister, \
        ClassicalRegister, execute

register(Qconfig.APItoken, Qconfig.config["url"])

DEBUG = False
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
            #weight = randint(1, 100)
            weight = 1
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

def get_expectation(x, g, NUM_SHOTS=1024):
    gamma, beta = x

    debug("Cost of Gamma: %s, beta: %s... " % (gamma, beta))

    # Construct quantum circuit.
    q = QuantumRegister(g.N)
    c = ClassicalRegister(g.N)
    qc = QuantumCircuit(q, c)

    # Apply hadamard to all inputs.
    for i in range(g.N):
        qc.h(q[i])

    # Apply V for all edges.
    for edge in g.get_edges():
        u, v, w = edge

        # Apply CNots.
        qc.cx(q[u], q[v])

        qc.u1(gamma*w, q[v])

        # Apply CNots.
        qc.cx(q[u], q[v])

    # Apply W to all vertices.
    for i in range(g.N):
        qc.h(q[i])
        qc.u1(-2*beta, q[i])
        qc.h(q[i])


    # Measure the qubits (avoiding ancilla).
    for i in range(g.N):
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

        #debug("\t\t%s: %s\n" % (bitstring, score))

        # Expected value is the score of each bitstring times
        # probability of it occuring.
        exp += score * prob

    debug("\tExpected Value: %s\n" % (exp))
    debug("\tBest Found Solution: %s, %s\n" % (g.currentScore, g.currentBest))

    g.add_run(gamma, beta, exp)

    return -1*exp # -1 bc we want to minimize

def run_optimizer(num_nodes, filename="results.csv"):
    debug("-- Building Graph--\n")
    g = Graph(num_nodes)
    debug(str(g) + "\n")

    best, best_val = g.optimal_score()
    debug("Optimal Solution: %s, %s\n" % (best, best_val[0]))

    # Initialize and run the algorithm.
    gamma_start = uniform(0, 2*np.pi)
    beta_start = uniform(0, np.pi)

    print("\n-- Starting optimization --")
    try:
        res = minimize(get_expectation, [gamma_start, beta_start], args=(g),
                options=dict(maxiter=2,disp=True), bounds=[(0, 2*np.pi), (0,np.pi)])
    except KeyboardInterrupt:
        debug("\nWriting to %s\n" % (filename))
        g.save_results(filename)
    finally:
        exit()

    debug("-- Finished optimization  --\n")
    debug("Gamma: %s, Beta: %s\n" % (res.x[0], res.x[1]))
    debug("Final cost: %s\n" % (res.maxcv))

    best, best_val = g.optimal_score()
    debug("Optimal Solution: %s, %s\n" % (best, best_val[0]))
    debug("Best Found Solution: %s, %s\n" % (g.currentScore, g.currentBest))

    debug("\nWriting to %s\n" % (filename))
    g.save_results(filename)



if __name__ == '__main__':
    # Choose some random starting beta and graph.
    beta = 0.79
    g = Graph(7)

    # RUNS # of runs at each gamma for error bars.
    RUNS = 3

    # Keep track of gammas, expected values, for plotting.
    gammas = []
    exps = [[] for i in range(RUNS)]

    # The maximum possible expected value is the maximum possible weighted cut.
    opt = g.optimal_score()[0]
    print("Optimal score: %s" % (opt))

    STEP = 0.02
    MIN = 0
    MAX = 2*np.pi
    #MAX = 1

    gams = np.linspace(MIN, MAX, int((MAX-MIN)/STEP))
    for gamma in tqdm(gams):
        gammas.append(gamma)

        # Calculate expected values.
        for i in range(RUNS):
            exps[i].append(-1 * get_expectation([gamma, beta], g))

        print("gamma: %s" % (gamma))

    '''
    fig, ax = plt.subplots()
    for i in range(RUNS):
        ax.plot(x=gammas, y=exps[i])
    '''

    #ax.errorbar(x=gammas, y=exp, yerr=std, marker='+', markersize=10)
    #ax.axhline(y=opt, xmin=0, xmax=2*np.pi, color='r', label="Max Exp. Value")
    #ax.legend(loc=2)

    plt.title("Effect of Varying Gamma")
    plt.xlabel("Gamma")
    plt.ylabel("Expected Value")

    for i in range(RUNS):
        plt.plot(gammas, exps[i])

    plt.show()


    '''
    For several random problem instances, plot the cost of the output state.
    Plot average, maximum and minimum cost.  How do these compare
    '''
    num_verts = [5,10,15,20]
    instances = [Graph(v) for v in num_verts]
    start_gamma = .5
    start_beta = .5
    RUNS = 3
    exps = [[] for i in range(len(num_verts))]

    for g, graph in enumerate(instances):
        for r in range(RUNS):
            exps[g].append(-1 * get_expectation([start_gamma, start_beta], graph))

    plt.title("Cost Distribution across Varying Graph Sizes")
    plt.xlabel("Number of Vertices")
    plt.ylabel("Cost")

    averages = [mean(instance) for instance in exps]
    lows = [min(instance) for instance in exps]
    highs = [max(instance) for instance in exps]

    plt.plot(num_verts, averages)
    plt.plot(num_verts, lows)
    plt.plot(num_verts, highs)

    plt.show()
