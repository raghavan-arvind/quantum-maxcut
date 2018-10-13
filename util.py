from random import randint, choice
from math import ceil

class Graph():
    def __init__(self, N):
        ''' Initialize a random graph with N vertices. '''
        self.N = N
        self.E = 0
        self.adj = {n:dict() for n in range(N)}
        
        self.randomize()

    def randomize(self):
        # all possible edges
        all_possible_edges = set([str(x)+str(y) for x in range(self.N) for y in range(self.N) if x != y])
        assert len(all_possible_edges) / 2== self.N * (self.N-1) / 2

        num_edges = randint(1, len(all_possible_edges)/2)
        self.E = num_edges
        for i in range(num_edges):
            # choose an edge, remove it and its complement from list
            e = choice(list(all_possible_edges))

            all_possible_edges.remove(e)
            all_possible_edges.remove(e[::-1])

            # unpack string into int vertexes
            u, v = int(e[0]), int(e[1])

            # generate random weight
            weight = randint(1, 100)
            self.add_edge(u, v, weight)


    def add_edge(self, u, v, weight):
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
        best = 0
        best_val = []
        for i in range(ceil((2 ** self.N) / 2)):
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


    
    def __str__(self):
        return "Graph with %d vertices %d edges.\nAdjacency List: %s" % (self.N, self.E, self.adj)

if __name__ == '__main__':
    g = Graph(4)
    print(g)
    print(g.optimal_score())
