import networkx as nx
import pandas as pd
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Load or Create the Graph
G = nx.DiGraph()
G.add_weighted_edges_from([
    ("A", "B", 3), ("A", "C", 1), ("C", "A", 2),
    ("B", "C", 5), ("C", "D", 4), ("D", "A", 2)
])

# Step 2: Compute Centrality Measures
def compute_centrality(graph):
    return {
        "Degree": nx.degree_centrality(graph),
        "Betweenness": nx.betweenness_centrality(graph, weight="weight", normalized=True),
        "Closeness": nx.closeness_centrality(graph, distance="weight"),
        "Eigenvector": nx.eigenvector_centrality_numpy(graph, weight="weight")
    }

start_time = time.time()
centrality = compute_centrality(G)
print(f"Centrality Computation Time: {time.time() - start_time:.4f} seconds")

# Step 3: Normalize Centrality Scores
centrality_df = pd.DataFrame(centrality)
scaler = StandardScaler()
centrality_normalized = scaler.fit_transform(centrality_df)

# Step 4: PCA for Composite Score
pca = PCA(n_components=1)
centrality_df["Composite"] = pca.fit_transform(centrality_normalized)

# Step 5: Graph Reduction Based on Composite Centrality
threshold = centrality_df["Composite"].quantile(0.8)  # Top 20% of nodes
nodes_to_keep = centrality_df[centrality_df["Composite"] >= threshold].index
reduced_graph = G.subgraph(nodes_to_keep).copy()

# Step 6: Evaluate the Reduced Graph
def evaluate_graph(original, reduced):
    metrics = {
        "Nodes": (original.number_of_nodes(), reduced.number_of_nodes()),
        "Edges": (original.number_of_edges(), reduced.number_of_edges()),
        "Connectivity (Components)": (
            nx.number_strongly_connected_components(original),
            nx.number_strongly_connected_components(reduced)
        ),
    }
    return metrics

evaluation = evaluate_graph(G, reduced_graph)

# Step 7: Visualize the Results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Graph")
nx.draw(G, with_labels=True, node_size=500, node_color="lightblue", edge_color="gray")

plt.subplot(1, 2, 2)
plt.title("Reduced Graph")
nx.draw(reduced_graph, with_labels=True, node_size=500, node_color="lightgreen", edge_color="gray")
plt.show()

# Step 8: Print Results
print("Graph Reduction Metrics:")
for metric, values in evaluation.items():
    print(f"{metric}: Original = {values[0]}, Reduced = {values[1]}")

print("\nExplained Variance by PCA:")
print(pca.explained_variance_ratio_)

