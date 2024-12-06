import networkx as nx
import community
import igraph
from collections import defaultdict
import sklearn
import numpy as np
from infomap import Infomap
import getdataset
import runwithstats

G = getdataset.get_dataset()


def imap(G):
    infomap = Infomap(directed=True)
    # add weighted edges to infomap
    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 0.0005)  # default weight is 0.0005 in case of unspecified
        infomap.add_link(u, v, weight)
    infomap.run()
    communities = {}
    for node_id, module_id in infomap.modules:
        if module_id not in communities:
            communities[module_id] = []
        communities[module_id].append(node_id)
    return communities


def labelprop(G):
    labels = {node: node for node in G.nodes()}
    for _ in range(100):  # limit iterations to avoid infinite loops
        updated = False
        for node in G.nodes():
            # collect weighted labels from incoming neighbors
            neighbors = list(G.predecessors(node))
            if not neighbors:
                continue
            neighbor_labels = {}
            for neighbor in neighbors:
                weight = G[neighbor][node]['weight']
                neighbor_label = labels[neighbor]
                neighbor_labels[neighbor_label] = neighbor_labels.get(neighbor_label, 0) + weight
            # update the label to most frequent
            if neighbor_labels:
                new_label = max(neighbor_labels, key=neighbor_labels.get)
                if labels[node] != new_label:
                    labels[node] = new_label
                    updated = True
        if not updated:  # stop if labels converge
            break
    # group nodes by labels
    communities = defaultdict(list)
    for node, label in labels.items():
        communities[label].append(node)
    return communities

def prune(G, communities, mincommsize):
    significant_communities = [nodes for nodes in communities.values() if len(nodes) >= mincommsize]
    keep = set(node for comm in significant_communities for node in comm)
    G2 = G.subgraph(keep).copy()
    return G2, G2.number_of_nodes(), G2.number_of_edges()


def analyze_communities(communities):
    # print(f"Total number of communities detected: {len(communities)}")
    avgsize = 0
    # print("Community sizes:")
    for community_id, members in communities.items():
        # print(f"  Community {community_id}: {len(members)} nodes")
        avgsize = avgsize + len(members)
    avgsize = avgsize / len(communities)
    return len(communities), avgsize


# imap initial prune size must be between 25 and 200
# labelprop prune sizes finnicky (100 does minimal pruning, 200 prunes almost half)


def statrunim(G, n):
    timetaken, finalmem, peakmem, comms = runwithstats.statrun(imap, G)
    G1, nodesrem, edgesrem = prune(G, comms, n)
    commnum, avgsize = analyze_communities(comms)
    return G1, nodesrem, edgesrem, timetaken, finalmem, peakmem, commnum, avgsize

def statrunlp(G, n):
    timetaken, finalmem, peakmem, comms = runwithstats.statrun(labelprop, G)
    G1, nodesrem, edgesrem = prune(G, comms, n)
    commnum, avgsize = analyze_communities(comms)
    return G1, nodesrem, edgesrem, timetaken, finalmem, peakmem, commnum, avgsize


def massrunim(G, runs):
    # initialize iteration variables so they aren't exclusive to the loop
    nodes1, nodes2, nodes3, nodes4, nodes5 = 0,0,0,0,0
    edges1, edges2, edges3, edges4, edges5 = 0,0,0,0,0
    time1, time2, time3, time4, time5 = 0,0,0,0,0
    final1, final2, final3, final4, final5 = 0,0,0,0,0
    peak1, peak2, peak3, peak4, peak5 = 0,0,0,0,0
    commnum1, commnum2, commnum3, commnum4, commnum5 = 0,0,0,0,0
    commsize1, commsize1, commsize3, commsize4, commsuze5 = 0,0,0,0,0
    # initialize total-tracking variables
    tn1, tn2, tn3, tn4, tn5 = 0,0,0,0,0
    te1, te2, te3, te4, te5 = 0,0,0,0,0
    tt1, tt2, tt3, tt4, tt5 = 0,0,0,0,0
    tf1, tf2, tf3, tf4, tf5 = 0,0,0,0,0
    tp1, tp2, tp3, tp4, tp5 = 0,0,0,0,0
    tcn1, tcn2, tcn3, tcn4, tcn5 = 0,0,0,0,0
    tcs1, tcs2, tcs3, tcs4, tcs5 = 0,0,0,0,0
    run = 0
    while run < runs:
        G1, nodes1, edges1, time1, final1, peak1, commnum1, commsize1 = statrunim(G, 150)
        G2, nodes2, edges2, time2, final2, peak2, commnum2, commsize2 = statrunim(G1, 150)
        G3, nodes3, edges3, time3, final3, peak3, commnum3, commsize3 = statrunim(G2, 150)
        G4, nodes4, edges4, time4, final4, peak4, commnum4, commsize4 = statrunim(G3, 150)
        G5, nodes5, edges5, time5, final5, peak5, commnum5, commsize5 = statrunim(G4, 150)
        tn1 = tn1 + nodes1
        tn2 = tn2 + nodes2
        tn3 = tn3 + nodes3
        tn4 = tn4 + nodes4
        tn5 = tn5 + nodes5
        te1 = te1 + edges1
        te2 = te2 + edges2
        te3 = te3 + edges3
        te4 = te4 + edges4
        te5 = te5 + edges5
        tt1 = tt1 + time1
        tt2 = tt2 + time2
        tt3 = tt3 + time3
        tt4 = tt4 + time4
        tt5 = tt5 + time5
        tf1 = tf1 + final1
        tf2 = tf2 + final2
        tf3 = tf3 + final3
        tf4 = tf4 + final4
        tf5 = tf5 + final5
        tp1 = tp1 + peak1
        tp2 = tp2 + peak2
        tp3 = tp3 + peak3
        tp4 = tp4 + peak4
        tp5 = tp5 + peak5
        tcn1 = tcn1 + commnum1
        tcn2 = tcn2 + commnum2
        tcn3 = tcn3 + commnum3
        tcn4 = tcn4 + commnum4
        tcn5 = tcn5 + commnum5
        tcs1 = tcs1 + commsize1
        tcs2 = tcs2 + commsize2
        tcs3 = tcs3 + commsize3
        tcs4 = tcs4 + commsize4
        tcs5 = tcs5 + commsize5
        run = run + 1
    # compile all averages
    # nodes
    an1 = tn1 / runs
    an2 = tn2 / runs
    an3 = tn3 / runs
    an4 = tn4 / runs
    an5 = tn5 / runs
    # edges
    ae1 = te1 / runs
    ae2 = te2 / runs
    ae3 = te3 / runs
    ae4 = te4 / runs
    ae5 = te5 / runs
    # time
    at1 = tt1 / runs
    at2 = tt2 / runs
    at3 = tt3 / runs
    at4 = tt4 / runs
    at5 = tt5 / runs
    # finalmem
    af1 = tf1 / runs
    af2 = tf2 / runs
    af3 = tf3 / runs
    af4 = tf4 / runs
    af5 = tf5 / runs
    # peakmem
    ap1 = tp1 / runs
    ap2 = tp2 / runs
    ap3 = tp3 / runs
    ap4 = tp4 / runs
    ap5 = tp5 / runs
    # number of communities
    acn1 = tcn1 / runs
    acn2 = tcn2 / runs
    acn3 = tcn3 / runs
    acn4 = tcn4 / runs
    acn5 = tcn5 / runs
    # size of communities
    acs1 = tcs1 / runs
    acs2 = tcs2 / runs
    acs3 = tcs3 / runs
    acs4 = tcs4 / runs
    acs5 = tcs5 / runs
    # massive print of all final average stats
    print("Average Run Stats by Iteration (Infomap):")
    print(f"  1: Reduced to {an1} nodes and {ae1} edges")
    print(f"  2: Reduced to {an2} nodes and {ae2} edges")
    print(f"  3: Reduced to {an3} nodes and {ae3} edges")
    print(f"  4: Reduced to {an4} nodes and {ae4} edges")
    print(f"  5: Reduced to {an5} nodes and {ae5} edges")
    print(f"  1: Took {at1} milliseconds, {af1} bytes upon completion, and {ap1} bytes at peak")
    print(f"  2: Took {at2} milliseconds, {af2} bytes upon completion, and {ap2} bytes at peak")
    print(f"  3: Took {at3} milliseconds, {af3} bytes upon completion, and {ap3} bytes at peak")
    print(f"  4: Took {at4} milliseconds, {af4} bytes upon completion, and {ap4} bytes at peak")
    print(f"  5: Took {at5} milliseconds, {af5} bytes upon completion, and {ap5} bytes at peak")
    print(f"  1: {acn1} communities, averaging size {acs1}")
    print(f"  2: {acn2} communities, averaging size {acs2}")
    print(f"  3: {acn3} communities, averaging size {acs3}")
    print(f"  4: {acn4} communities, averaging size {acs4}")
    print(f"  5: {acn5} communities, averaging size {acs5}")


def massrunlp(G, runs):
    # initialize iteration variables so they aren't exclusive to the loop
    nodes1, nodes2, nodes3, nodes4, nodes5 = 0,0,0,0,0
    edges1, edges2, edges3, edges4, edges5 = 0,0,0,0,0
    time1, time2, time3, time4, time5 = 0,0,0,0,0
    final1, final2, final3, final4, final5 = 0,0,0,0,0
    peak1, peak2, peak3, peak4, peak5 = 0,0,0,0,0
    commnum1, commnum2, commnum3, commnum4, commnum5 = 0,0,0,0,0
    commsize1, commsize1, commsize3, commsize4, commsize5 = 0,0,0,0,0
    # initialize total-tracking variables
    tn1, tn2, tn3, tn4, tn5 = 0,0,0,0,0
    te1, te2, te3, te4, te5 = 0,0,0,0,0
    tt1, tt2, tt3, tt4, tt5 = 0,0,0,0,0
    tf1, tf2, tf3, tf4, tf5 = 0,0,0,0,0
    tp1, tp2, tp3, tp4, tp5 = 0,0,0,0,0
    tcn1, tcn2, tcn3, tcn4, tcn5 = 0,0,0,0,0
    tcs1, tcs2, tcs3, tcs4, tcs5 = 0,0,0,0,0
    run = 0
    while run < runs:
        G1, nodes1, edges1, time1, final1, peak1, commnum1, commsize1 = statrunlp(G, 100)
        G2, nodes2, edges2, time2, final2, peak2, commnum2, commsize2 = statrunlp(G1, 125)
        G3, nodes3, edges3, time3, final3, peak3, commnum3, commsize3 = statrunlp(G2, 150)
        G4, nodes4, edges4, time4, final4, peak4, commnum4, commsize4 = statrunlp(G3, 175)
        G5, nodes5, edges5, time5, final5, peak5, commnum5, commsize5 = statrunlp(G4, 200)
        tn1 = tn1 + nodes1
        tn2 = tn2 + nodes2
        tn3 = tn3 + nodes3
        tn4 = tn4 + nodes4
        tn5 = tn5 + nodes5
        te1 = te1 + edges1
        te2 = te2 + edges2
        te3 = te3 + edges3
        te4 = te4 + edges4
        te5 = te5 + edges5
        tt1 = tt1 + time1
        tt2 = tt2 + time2
        tt3 = tt3 + time3
        tt4 = tt4 + time4
        tt5 = tt5 + time5
        tf1 = tf1 + final1
        tf2 = tf2 + final2
        tf3 = tf3 + final3
        tf4 = tf4 + final4
        tf5 = tf5 + final5
        tp1 = tp1 + peak1
        tp2 = tp2 + peak2
        tp3 = tp3 + peak3
        tp4 = tp4 + peak4
        tp5 = tp5 + peak5
        tcn1 = tcn1 + commnum1
        tcn2 = tcn2 + commnum2
        tcn3 = tcn3 + commnum3
        tcn4 = tcn4 + commnum4
        tcn5 = tcn5 + commnum5
        tcs1 = tcs1 + commsize1
        tcs2 = tcs2 + commsize2
        tcs3 = tcs3 + commsize3
        tcs4 = tcs4 + commsize4
        tcs5 = tcs5 + commsize5
        run = run + 1
    # compile all averages
    # nodes
    an1 = tn1 / runs
    an2 = tn2 / runs
    an3 = tn3 / runs
    an4 = tn4 / runs
    an5 = tn5 / runs
    # edges
    ae1 = te1 / runs
    ae2 = te2 / runs
    ae3 = te3 / runs
    ae4 = te4 / runs
    ae5 = te5 / runs
    # time
    at1 = tt1 / runs
    at2 = tt2 / runs
    at3 = tt3 / runs
    at4 = tt4 / runs
    at5 = tt5 / runs
    # finalmem
    af1 = tf1 / runs
    af2 = tf2 / runs
    af3 = tf3 / runs
    af4 = tf4 / runs
    af5 = tf5 / runs
    # peakmem
    ap1 = tp1 / runs
    ap2 = tp2 / runs
    ap3 = tp3 / runs
    ap4 = tp4 / runs
    ap5 = tp5 / runs
    # number of communities
    acn1 = tcn1 / runs
    acn2 = tcn2 / runs
    acn3 = tcn3 / runs
    acn4 = tcn4 / runs
    acn5 = tcn5 / runs
    # size of communities
    acs1 = tcs1 / runs
    acs2 = tcs2 / runs
    acs3 = tcs3 / runs
    acs4 = tcs4 / runs
    acs5 = tcs5 / runs
    # massive print of all final average stats
    print("Average Run Stats by Iteration (Label Propogation):")
    print(f"  1: Reduced to {an1} nodes and {ae1} edges")
    print(f"  2: Reduced to {an2} nodes and {ae2} edges")
    print(f"  3: Reduced to {an3} nodes and {ae3} edges")
    print(f"  4: Reduced to {an4} nodes and {ae4} edges")
    print(f"  5: Reduced to {an5} nodes and {ae5} edges")
    print(f"  1: Took {at1} milliseconds, {af1} bytes upon completion, and {ap1} bytes at peak")
    print(f"  2: Took {at2} milliseconds, {af2} bytes upon completion, and {ap2} bytes at peak")
    print(f"  3: Took {at3} milliseconds, {af3} bytes upon completion, and {ap3} bytes at peak")
    print(f"  4: Took {at4} milliseconds, {af4} bytes upon completion, and {ap4} bytes at peak")
    print(f"  5: Took {at5} milliseconds, {af5} bytes upon completion, and {ap5} bytes at peak")
    print(f"  1: {acn1} communities, averaging size {acs1}")
    print(f"  2: {acn2} communities, averaging size {acs2}")
    print(f"  3: {acn3} communities, averaging size {acs3}")
    print(f"  4: {acn4} communities, averaging size {acs4}")
    print(f"  5: {acn5} communities, averaging size {acs5}")

# test size 5 to check functionality, up to 100 for data collection
massrunim(G, 5)
massrunlp(G, 5)
