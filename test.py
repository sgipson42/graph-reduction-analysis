import networkx as nx
import matplotlib.pyplot as plt

file_path = "congress_network/congress.edgelist"
G = nx.read_edgelist(file_path, create_using=nx.DiGraph)
