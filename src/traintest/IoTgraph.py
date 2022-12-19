import random
import networkx as nx
import numpy as np
import gym

from gym import spaces

class IoTgraph(gym.Env):


    def random_edge(self):
        """Add random edges to the graph until it is a complete graph"""
        edges = list(self.graph.edges)
        nonedges = list(nx.non_edges(self.graph))

        if len(edges) > 0:
            chosen_edge = random.choice(edges)
            chosen_nonedge = random.choice([x for x in nonedges if chosen_edge[0] == x[0] or chosen_edge[0] == x[1]])
        else:
            chosen_nonedge = random.choice(nonedges)

        self.graph.add_edge(chosen_nonedge[0], chosen_nonedge[1])
       

    def is_corrupted(self, path, verbose=True):
        """Determine if a path is corrupted 
        based on the attack probabilities of the nodes that make it up"""
        for node in path:
            attack_prob = self.devices[node].attack_prob
            attacked = random.uniform(0,1) < attack_prob

            if attacked:
                if verbose:
                    print(f'Path = {path} is corrupted')
                    print(f'Node {node} attacked this path')
                return True

        return False


    def __init__(self, fname=None, network_size=10, edge_prob=1, percent_mal=0, attack_probs=[0, 0]):
        """Initialize the graph"""
        self.devices = []
        self.mal_nodes = []

        class device:
            pass

        # Read in a custom environment from a file and create the graph that way
        if fname != None:
            with open(fname) as f:
                lines = [line.rstrip() for line in f]

                self.network_size = int(lines[0])
                self.src = 0
                self.dst = self.network_size - 1

                edges = [eval(x) for x in lines[1:-1]]
                self.graph = nx.Graph()
                self.graph.add_nodes_from(list(range(self.network_size)))
                self.graph.add_edges_from(edges)

                attack_probs = [float(x) for x in lines[-1].split(',')]

                nodes = list(range(0, self.network_size))

                for node, prob in zip(nodes, attack_probs):
                    a = device()
                    a.node = node
                    a.attack_prob = prob

                    if prob > 0:
                        self.mal_nodes.append(node)
                        a.mal = True

                    else:
                        a.mal = False

                    self.devices.append(a)

        # Create a random graph based on a set of parameters
        else:
            self.network_size = network_size
            self.src = 0
            self.dst = network_size - 1

            self.graph = nx.gnp_random_graph(network_size, edge_prob)
            while not nx.is_connected(self.graph):
                self.random_edge()

            num_mal = network_size * percent_mal

            while num_mal > 0:
                rand = np.random.randint(0, network_size)
                if rand != self.src and rand != self.dst and rand not in self.mal_nodes:
                    self.mal_nodes.append(rand)
                    num_mal -= 1

            nodes = list(range(0, network_size))

            for node in nodes:
                a = device()
                a.node = node

                if node in self.mal_nodes:
                    a.mal = True
                    a.attack_prob = np.random.uniform(attack_probs[0], attack_probs[1])
                else:
                    a.mal = False
                    a.attack_prob = 0

                self.devices.append(a)

        self.num_actions = self.network_size
        self.num_states  = self.network_size

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_states)


    def render(self):
        """Draw the graph"""
        nx.draw(self.graph, with_labels=True)
