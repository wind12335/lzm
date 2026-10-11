import networkx as nx

DATA_NODE = ["Const"]
DATA_MANIPULATION_NODE = ["Reshape", "transpose", "concat"]

def merge_nodes(graph, nodes_to_merge, threshold):
    if nodes_to_merge[0] not in graph.nodes() or \
        nodes_to_merge[1] not in graph.nodes():
        return 

    if len(graph.nodes[nodes_to_merge[0]]['group']) + \
        len(graph.nodes[nodes_to_merge[1]]['group']) > threshold:
        return

    # Compute the new attributes for the merged node
    compute_cost = sum(graph.nodes[n]['compute_cost'] for n in nodes_to_merge)
    memory_cost = sum(graph.nodes[n]['memory'] for n in nodes_to_merge)
    device = graph.nodes[nodes_to_merge[0]]['device']  # Use the device of the first node
    group = graph.nodes[nodes_to_merge[0]]['group'] + graph.nodes[nodes_to_merge[1]]['group']

    # Create a new node with the combined attributes
    merged_node = f"merged_{'_'.join(nodes_to_merge)}"
    graph.add_node(merged_node, compute_cost=compute_cost, memory=memory_cost, device=device, group=group)

    # Transfer edges to the merged node and remove original nodes
    for n in nodes_to_merge:
        for neighbor in list(graph.neighbors(n)):
            if neighbor not in nodes_to_merge:
                weight = graph.edges[n, neighbor]['weight']
                if not graph.has_edge(merged_node, neighbor):
                    graph.add_edge(merged_node, neighbor, weight=weight)
                else:  # If the edge already exists, update the weight
                    graph.edges[merged_node, neighbor]['weight'] += weight
        graph.remove_node(n)

    # Remove edges between nodes in the group, if any exist
    for n1 in group:
        for n2 in group:
            if graph.has_edge(merged_node, f"merged_{n1}_{n2}"):
                graph.remove_edge(merged_node, f"merged_{n1}_{n2}")

    return graph


def twolayersmerge(G, threshold):
    if threshold == 1:
        print("threshold {} node num {}".format(threshold, len(G.graph.nodes())))
        return G
    
    nx_graph = G.graph
    for node in nx_graph.nodes():
        nx_graph.nodes[node]['group'] = [node]
    
    nodes_to_merges = []
    for node in nx_graph.nodes():
        succ_node = list(nx_graph.successors(node))
        if len(succ_node) != 0 and any(s in node for s in DATA_NODE) is True:
            nodes_to_merge = [node, succ_node[0]]
            # print(nodes_to_merge)
            nodes_to_merges.append(nodes_to_merge)
    for nodes_to_merge in nodes_to_merges:
        merge_nodes(nx_graph, nodes_to_merge, threshold)
    
    nodes_to_merges = []
    for node in nx_graph.nodes():
        pred_node = list(nx_graph.predecessors(node))
        if len(pred_node) != 0 and any(s in node for s in DATA_MANIPULATION_NODE) is True:
            nodes_to_merge = [pred_node[0], node]
            # print(nodes_to_merge)
            nodes_to_merges.append(nodes_to_merge)
    for nodes_to_merge in nodes_to_merges:
        merge_nodes(nx_graph, nodes_to_merge, threshold)

    prior_node_num = len(nx_graph.nodes())
    next_node_num = prior_node_num + 1
    nodes_to_merges = []
    while next_node_num != prior_node_num:
        prior_node_num = len(nx_graph.nodes())
        for node in nx_graph.nodes():
            succ_node = list(nx_graph.successors(node))
            if len(succ_node) == 1:
                if nx_graph.edges[node, succ_node[0]]['weight'] > \
                    min(nx_graph.nodes[node]['compute_cost'], nx_graph.nodes[succ_node[0]]['compute_cost']):
                    nodes_to_merge = [node, succ_node[0]]
                    nodes_to_merges.append(nodes_to_merge)
        for nodes_to_merge in nodes_to_merges:
            merge_nodes(nx_graph, nodes_to_merge, threshold)
        next_node_num = len(nx_graph.nodes())
                    
    print("threshold {} node num {}".format(threshold, len(nx_graph.nodes())))
