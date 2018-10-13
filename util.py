from random import randint, choice
from math import ceil

class Graph():
    def __init__(self, N, init=True):
        ''' Initialize a random graph with N vertices. '''
        self.N = N
        self.E = 0
        self.adj = {n:dict() for n in range(N)}
        
        # Randomly generate edges
        if init:
            self.randomize()

    def randomize(self):
        ''' Randomly generate edges for this graph. '''
        # all possible edges
        all_possible_edges = set([(x,y) for x in range(self.N) for y in range(self.N) if x != y])

        # sanity check, ensuring we generated the correct number of edges
        e_gen = len(all_possible_edges) / 2
        e_shd = self.N * (self.N-1) / 2
        assert e_gen == e_shd , "%d != %d" % (e_gen, e_shd)

        # can have 1 - N(N-1)/2 edges
        num_edges = randint(1, len(all_possible_edges)/2)
        for i in range(num_edges):
            # choose an edge, remove it and its complement from list
            e = choice(list(all_possible_edges))
            all_possible_edges.remove(e)
            all_possible_edges.remove(e[::-1])

            # unpack string edge into int vertices
            u, v = int(e[0]), int(e[1])

            # generate random weight
            weight = randint(1, 100)
            self.add_edge(u, v, weight)


    def add_edge(self, u, v, weight):
        ''' Add an edge to the graph. '''
        self.E += 1
        self.adj[u][v] = weight

    def get_score(self,bitstring):
        ''' Score a candidate solution. '''
        assert len(bitstring) == self.N

        score = 0
        # for every edge u,v in the graph
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
        for i in range(ceil((2 ** self.N)/2)):
            # 0-padded bitstring repr of i
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
        num = 0
        for u in self.adj:
            for v in self.adj[u]:
                if bitstring[u] != bitstring[v]:
                    num += 1
        return num
    
    def __str__(self):
        return "Graph with %d vertices %d edges.\nAdjacency List: %s" % (self.N, self.E, self.adj)

if __name__ == '__main__':
    '''
    for _ in range(100):
        g = Graph(15)
        edges = g.E

        best_solution = g.optimal_score()[1][0]
        print(1.0 * g.edges_cut(best_solution) / edges)
    '''
    
    '''
    g = Graph(6, init=False)

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
    pass

