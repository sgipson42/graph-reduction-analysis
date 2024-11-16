import networkx as nx
import matplotlib.pyplot as plt

file_path = "congress_network/congress.edgelist"
G = nx.read_edgelist(file_path, create_using=nx.DiGraph)

# Visualize the graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)
plt.title("Congressional Twitter Interaction Network")
nx.draw(G, pos, with_labels=True)
plt.savefig("twitter_network.png", dpi=300)
plt.show()
