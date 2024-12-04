import json
import networkx as nx


with open('congress_network/congress_network_data.json', 'r') as file:
    data = json.load(file)

data = data[0]

in_list = data['inList']
in_weight = data['inWeight']
out_list = data['outList']
out_weight = data['outWeight']
username_list = data['usernameList']

G = nx.DiGraph()

for i, username in enumerate(username_list):
    G.add_node(i, username=username)

for target_node, sources in enumerate(in_list):
    for source_node, weight in zip(sources, in_weight[target_node]):
        G.add_edge(source_node, target_node, weight=weight)


def get_dataset():
    return G
