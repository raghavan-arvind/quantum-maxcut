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
            weight = randint(-100, 100)

            #weight = 1
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
    job = execute(qc, backend='local_qasm_simulator', shots=NUM_SHOTS)
    results = job.result()
    result_dict = results.get_counts(qc)

    debug("done!\n")

    # Calculate the expected value of the candidate bitstrings.
    exp = 0
    for bitstring in result_dict:
        prob = np.float(result_dict[bitstring]) / NUM_SHOTS
        score = g.update_score(bitstring)

        # Expected value is the score of each bitstring times
        # probability of it occuring.
        exp += score * prob

    debug("\tExpected Value: %s\n" % (exp))
    debug("\tBest Found Solution: %s, %s\n" % (g.currentScore, g.currentBest))

    g.add_run(gamma, beta, exp)

    return exp # bc we want to minimize

def run_optimizer(num_nodes, filename="results.csv"):
    debug("-- Building Graph--\n")
    g = Graph(num_nodes)
    debug(str(g) + "\n")

    best, best_val = g.optimal_score()
    debug("Optimal Solution: %s, %s\n" % (best, best_val[0]))

    # Initialize and run the algorithm.
    gamma_start = uniform(0, 2*np.pi)
    beta_start = uniform(0, np.pi)

    # minimize wants lower values, so negate expectation.
    neg_get_expectatation = lambda x, y: -1 * get_expectation(x, y)

    debug("\n-- Starting optimization --\n")
    try:
        res = minimize(neg_get_expectation, [gamma_start, beta_start], args=(g),
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


#def instance_cost(num_instances=30, num_vert=10, num_runs=5):
def instance_cost(num_instances=5, num_vert=5, num_runs=5):
    '''
    For several random problem instances, plot the cost of the output state.
    Plot average, maximum and minimum cost.
    '''

    # Prepare several random instances of the problem.
    instances = [Graph(num_vert) for _ in range(num_instances)]

    # Choose starting values for gamma and beta.

    # For holding iteration number and expected values.
    its, exps, opt = [], [], []

    # Calculate expected values.
    for it, graph in tqdm(enumerate(instances)):

        vals = []
        for _ in range(num_runs):
            # Use random gamma, beta for each run.
            gamma = uniform(0, 2*np.pi)
            beta = uniform(0, np.pi)

            vals.append(get_expectation([gamma, beta], graph))

        # Save results.
        its.append(it+1)
        exps.append(vals)
        opt.append(graph.optimal_score()[0])


    plt.title("Costs of Random Instances")
    plt.xlabel("Iteration Number")
    plt.ylabel("Cost")

    # Sort by optimal value just so it's pleasant to look at.
    exps = [x for _,x in sorted(zip(opt,exps), key=lambda pair: pair[0])]
    opt = sorted(opt)

    averages = [mean(ex) for ex in exps]
    lows = [min(ex) for ex in exps]
    highs = [max(ex) for ex in exps]

    plt.plot(its, averages, color='blue', label='Average Cost')
    plt.plot(its, lows, color='green', label='Minimum Cost')
    plt.plot(its, highs, color='orange', label='Maximum Cost')
    plt.plot(its, opt, color='red', label='Optimal Cost')

    plt.legend()

    plt.show()

def hold_constant():
    ''' Plots expected value vs. gamma/beta, holding the rest of the variables constant.'''

    # Choose some random starting gamma and graph.
    gamma = 2.251
    g = Graph(5)

    # RUNS # of runs at each gamma for error bars.
    RUNS = 3

    # Keep track of gammas, expected values, for plotting.
    betas, exp, std = [], [], []

    # The maximum possible expected value is the maximum possible weighted cut.
    opt = g.optimal_score()[0]
    debug("Optimal score: %s\n" % (opt))

    NUM_RUNS = 32
    MIN = 0
    MAX = np.pi
    #MAX = 1

    bets = np.linspace(MIN, MAX, NUM_RUNS)
    for beta in tqdm(bets):
        betas.append(beta)

        # Calculate expected values.
        vals = []
        for i in range(RUNS):
            vals.append(get_expectation([gamma, beta], g))

        # Calculate mean, standard deviation.
        exp.append(mean(vals))
        std.append(stdev(vals))


    fig, ax = plt.subplots()
    '''
    for i in range(RUNS):
        ax.plot(x=gammas, y=exps[i])
    '''

    ax.errorbar(x=betas, y=exp, yerr=std, fmt='o-', markersize=10)
    #ax.axhline(y=opt, xmin=0, xmax=2*np.pi, color='r', label="Max Exp. Value")
    ax.legend(loc=2)

    ax.set_title("Effect of Varying Beta with Gamma = %s" % (gamma))
    ax.set_xlabel("Beta")
    ax.set_ylabel("Expected Value")

    '''
    for i in range(RUNS):
        plt.plot(gammas, exps[i])
    '''

    plt.show()

if __name__ == '__main__':
    instance_cost()

