import networkx as nx
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import getdataset
from rich.console import Console
from runwithstats import statrun
import pandas as pd

def prune(G, nodeskept):
    G2 = G.subgraph(nodeskept).copy()
    return G2, G2.number_of_nodes(), G2.number_of_edges()

def evaluate_graph(original, reduced):
    metrics = {
        "Nodes": (original.number_of_nodes(), reduced.number_of_nodes()),
        "Edges": (original.number_of_edges(), reduced.number_of_edges()),
        "Connectivity (Components)": (
            nx.number_strongly_connected_components(original),
            nx.number_strongly_connected_components(reduced)
        ),
	"Node Retention": reduced.number_of_nodes() / original.number_of_nodes(),
        "Edge Retention": reduced.number_of_edges() / original.number_of_edges()
    }
    return metrics

def visualize_graph(G, metric):
	plt.figure(figsize=(12, 8))
	pos = nx.spring_layout(G, seed=42)
	plt.title(metric)
	nx.draw(G, pos, with_labels=True)
	plt.savefig(f"./visualizations/{metric}.png", dpi=300)
	#plt.show()


console = Console()
df = pd.DataFrame(columns=['time_taken', 'current_memory', 'peak_memory', 'centrality_scores', 'metric'])
scores_df = pd.DataFrame()
reduction_df = pd.DataFrame()
G = getdataset.get_dataset()

#funcs = [nx.in_degree_centrality, nx.out_degree_centrality, nx.closeness_centrality, nx.betweenness_centrality, nx.eigenvector_centrality]
funcs = [
    (nx.degree_centrality, {}),
    (nx.betweenness_centrality, {"weight": "weight", "normalized":True}),
    (nx.closeness_centrality, {"distance": "weight"})
]
results_file = 'all_results.csv'
columns = ['time_taken', 'current_memory', 'peak_memory', 'centrality_scores', 'metric', 'trial', 
           'threshold', 'reduction_level', 'node_count', 'edge_count', 
           'node_retention', 'edge_retention', 'connectivity']
pd.DataFrame(columns=columns).to_csv(results_file, index=False)
trials =  100 
num_reductions = 10 
#thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
thresholds = [0.2, 0.4, 0.6, 0.8]
for idx, (func, kwargs) in enumerate(funcs):
	for threshold_score in thresholds:
		for trial in range(trials):
			G_copy = G
			rerun = True
			for i in range(num_reductions):
				if rerun: 
					# make sure that you have weights being used in each function
					time_taken, current_memory, peak_memory, centrality_scores = statrun(func, G_copy, **kwargs)
					scores_df[func.__name__] = centrality_scores
					if isinstance(func, str):
						col = func
					else:
						col = func.__name__
					# take the remaining amount as quality
					threshold = scores_df[col].quantile(threshold_score) 
					nodes_to_keep = scores_df[scores_df[col] >= threshold].index
					#time_taken, current_memory, peak_memory, (reduced_graph, nodes, edges) = statrun(prune, G_copy, nodes_to_keep)
					reduced_graph, nodes, edges = prune(G_copy, nodes_to_keep)
					evaluation = evaluate_graph(G_copy, reduced_graph)
					console.print(col)
					console.print(evaluation)
					row = {
						'time_taken': time_taken,
						'current_memory': current_memory,
						'peak_memory': peak_memory,
						'centrality_scores': centrality_scores,
						'metric':func.__name__,
						'trial':trial,
						'threshold':threshold_score,
						'reduction_level':i,
						'node_count': nodes,
						'edge_count': edges,
						'node_retention': evaluation["Node Retention"],
						'edge_retention':evaluation["Edge Retention"],
						'connectivity':evaluation["Connectivity (Components)"][1],
					}
					pd.DataFrame([row]).to_csv(results_file, mode='a', header=False, index=False)
						
					"""
					reduction_df = reduction_df._append({
							'time_taken': time_taken,
							'current_memory': current_memory,
							'peak_memory': peak_memory,
							'nodes': nodes,
							'edges':edges,
							'connectivity':evaluation["Connectivity (Components)"][1],
							'metric':col,
							'trial':trial,
							'threshold':threshold_score,
							'reduction_level':i
						}, ignore_index=True)
						
					"""
					#visualize_graph(reduced_graph, col)
					G_copy = reduced_graph
					if G_copy.number_of_nodes()<2 or evaluation["Connectivity (Components)"][1] > evaluation["Connectivity (Components)"][0] or (evaluation["Node Retention"]==1.0) or (evaluation["Edge Retention"]==1.0):
						rerun=False
					else: 
						rerun=True
			

console.print('loop complete.')
df.to_csv('centrality_results.csv')
scores_df.to_csv('centrality_scores.csv')

# first evaluate what each reduction looks like just based on these scores
"""
scores_df['composite_score'] = scores_df.mean(axis=1)
funcs.append('composite_score')
all_nodes = []
composite = set()
for func in funcs:
	if isinstance(func, str):
		col = func
	else:
		col = func.__name__
	# take the remaining amount as quality
	threshold = scores_df[col].quantile(0.8) 
	nodes_to_keep = scores_df[scores_df[col] >= threshold].index
	time_taken, current_memory, peak_memory, (reduced_graph, nodes, edges) = statrun(prune, G, nodes_to_keep)
	evaluation = evaluate_graph(G, reduced_graph)
	console.print(col)
	console.print(evaluation)
	reduction_df = reduction_df._append({
			'time_taken': time_taken,
			'current_memory': current_memory,
			'peak_memory': peak_memory,
			'nodes': nodes,
			'edges':edges,
			'connectivity':evaluation["Connectivity (Components)"][1],
			'metric':col
		}, ignore_index=True)
		
	visualize_graph(reduced_graph, col)
	node_list = list(reduced_graph.nodes())
	if isinstance(func, str):
		composite_set = set(node_list)
		composite = composite_set
	else:
		all_nodes.append(node_list)
"""		
"""		
# now want to see number of nodes shared between the graphs
intersection = set(all_nodes[0]).intersection(*all_nodes[1:])
console.print(len(intersection))
console.print(len(composite))
comp_intersection = intersection.intersection(composite)
console.print(len(comp_intersection))
reduction_df.to_csv('reduction_results.csv')
"""		



# then compare the composite to each individual

# need to evaluate how similar the scores for each node are
# get a composite score per node
