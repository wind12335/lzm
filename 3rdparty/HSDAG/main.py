import openvino.runtime as ov
import torch
import torch.nn as nn
from torch.distributions import Categorical
from config import Config
from layer import GNN, MLP, ParsingNet_GPU as ParsingNet
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import networkx as nx
import numpy as np
import time
from model import GPN
import json
import xml.etree.ElementTree as ET
from torch_sparse import SparseTensor
import torch_sparse
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from nfd import node_dimension



class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.entropys = []
        self.rewards = []
        self.is_terminals = []
        self.clusters = []
        self.node_ids = []


    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.entropys[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.clusters[:]
        del self.node_ids[:]

def xml2graph(ov_model,xml_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    G = nx.DiGraph()

    for layer in root.find('layers'):
        layer_id = layer.get('id')
        layer_name = layer.get('name')
        layer_type = layer.get('type')
        data_element = layer.find('data')
        # print(layer_name)
        if data_element is not None:
            # Get the 'shape' attribute from the 'data' element
            data_shape = data_element.get('shape')
        G.add_node(layer_id, name=layer_name, type=layer_type, shape=data_element)

    for edge in root.find('edges'):
        from_layer = edge.get('from-layer')
        to_layer = edge.get('to-layer')

        G.add_edge(from_layer, to_layer)
        
    # print(G.number_of_nodes())

    return G

# Function to find and merge nodes based on the heuristic
def merge_operations(graph):
    current_groups = []
    co_location_groups = []
    merged = set()
    
    for node in nx.topological_sort(graph):
        
        # Check if the node has exactly one successor and no other predecessors of successor
        successors = list(graph.successors(node))
        predecessors = list(graph.predecessors(node))
        # print(successors)
        if len(predecessors) == 1 and predecessors[0] in merged:
            for group in co_location_groups:
                if predecessors[0] in group:
                    current_groups = group
                    co_location_groups.remove(group)
        else:
            current_groups = []
            
        # print(current_groups)
        if node not in current_groups and node not in merged:
            current_groups.append(node)
            if len(successors) == 1:
                successor = successors[0]
                predecessors = list(graph.predecessors(successor))
                # print(predecessors)
                if len(predecessors) == 1 and predecessors[0] == node:
                    if successor not in merged:
                        current_groups.append(successor)
                        merged.add(successor)
            # print(current_groups)
            
        co_location_groups.append(current_groups)
        current_groups = []
    
        merged.add(node)
        # print(co_location_groups)
                
    return co_location_groups

# Function to merge nodes and create a new graph
def merge_nodes(graph, groups):
    new_graph = nx.DiGraph()
    group_map = {}

    # Map each node to its group identifier (new node)
    for idx, group in enumerate(groups):
        node_name = f"Group_{idx}"
        for node in group:
            group_map[node] = node_name
            if node_name not in new_graph:
                new_graph.add_node(node_name)
                
    # print(group_map)
    
    # Add edges with respect to new group nodes
    for u, v in graph.edges():
        new_u = group_map.get(u, u)
        new_v = group_map.get(v, v)
        if new_u != new_v:
            new_graph.add_edge(new_u, new_v)

    return new_graph, group_map

def get_final_graph(xml_file,Core):    
    # prepare input_data
    
    input_data = torch.rand(1, 3, 299, 299)

    ov_model = Core.read_model(model=xml_file)
    input_layer = ov_model.input()

    # compile model
    compiled_model = Core.compile_model(ov_model,device_name="MULTI:GPU,CPU")
    # compiled_model = ov.compile_model(ov_model,device_name="CPU")

    infer_request = compiled_model.create_infer_request()
    infer_request.infer(inputs={input_layer.any_name: input_data})
    # run inference
    result = infer_request.results
    cm = infer_request.get_compiled_model()
    # print(cm)
    runtime_model = cm.get_runtime_model()
    ops = ov_model.get_ordered_ops()
    # ops = runtime_model.get_ordered_ops()
    # print(len(ops))
    # Computation_G = xml2graph(runtime_model)
    Computation_G = xml2graph(ov_model,xml_file)
    # print(len(ops))
    ops_ov = ov_model.get_ordered_ops()
    
    op_type_name = []
    op_info = {}
    for op in ops_ov:
        op_attributes = op.get_attributes()
        # print(f"{op}: {op_attributes}")
        # print(op.get_friendly_name())
        # print(op.get_input_size)
        # print(op.get_type_name())
        op_type_name.append(op.get_type_name())
        op_info[op.get_friendly_name()] = op.get_type_name()
        # print(f"{op.get_friendly_name()}: {op.get_type_name()}")
    unique_list = list(set(op_type_name))
    # print(unique_list)
    co_location_groups = merge_operations(Computation_G)
    new_G,group_map = merge_nodes(Computation_G, co_location_groups)


    em = EmbeddingModel(unique_list)
    op_embeddings,group_op_map = em.forward(op_info,new_G,group_map,ops,Computation_G)
    for node in new_G.nodes():
        new_G.nodes[node]['x'] = op_embeddings[int(node[6:])]
    
    # node_mapping = {node: int(node[6:]) for i, node in enumerate(new_G.nodes)}
    # new_G = nx.relabel_nodes(new_G, node_mapping)
    # nfd_centrality = node_dimension(new_G,weight=None).values() #Peiyu: weight=None
    # nfd_centrality_dict = dict(zip(new_G.nodes(), nfd_centrality))    
    # nx.set_node_attributes(new_G, nfd_centrality_dict, 'nfd')

    # closeness_centrality = nx.closeness_centrality(new_G,distance=None).values()
    # closeness_centrality_dict = dict(zip(new_G.nodes(), closeness_centrality))    
    # nx.set_node_attributes(new_G, closeness_centrality_dict, 'closeness')
    # for node in new_G.nodes:
    #     new_G.nodes[node]['x'] = torch.cat((op_embeddings[int(node)], torch.tensor([new_G.nodes[node]['nfd']],dtype=op_embeddings.dtype), torch.tensor([new_G.nodes[node]['closeness']],dtype=op_embeddings.dtype), torch.tensor([node], dtype=op_embeddings.dtype)), 0)
    
    # node_mapping = {node: 'Group_'+str(node) for i, node in enumerate(new_G.nodes)}
    # new_G = nx.relabel_nodes(new_G, node_mapping)
        
    return new_G,group_op_map,Computation_G

# def get_final_graph(xml_file,Core):    
#     # prepare input_data
    
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     texts = ["Hello, my dog is cute"]
#     inputs = tokenizer(texts, return_tensors="pt", padding=True)
#     input_data = (inputs['input_ids'], inputs['attention_mask'])

#     ov_model = Core.read_model(model=xml_file)
#     input_layer_1 = ov_model.input(0)
#     input_layer_2 = ov_model.input(1)

#     # compile model
#     compiled_model = Core.compile_model(ov_model,device_name="MULTI:GPU,CPU")
#     # compiled_model = ov.compile_model(ov_model,device_name="CPU")

#     infer_request = compiled_model.create_infer_request()
#     infer_request.infer(inputs={input_layer_1.any_name: input_data[0], input_layer_2.any_name: input_data[1]})
#     # run inference
#     result = infer_request.results
#     cm = infer_request.get_compiled_model()
#     # print(cm)
#     runtime_model = cm.get_runtime_model()
#     ops = ov_model.get_ordered_ops()
#     # ops = runtime_model.get_ordered_ops()
#     # print(len(ops))
#     # Computation_G = xml2graph(runtime_model)
#     Computation_G = xml2graph(ov_model,xml_file)
#     # print(len(ops))
#     ops_ov = ov_model.get_ordered_ops()
    
#     op_type_name = []
#     op_info = {}
#     for op in ops_ov:
#         op_attributes = op.get_attributes()
#         # print(f"{op}: {op_attributes}")
#         # print(op.get_friendly_name())
#         # print(op.get_input_size)
#         # print(op.get_type_name())
#         op_type_name.append(op.get_type_name())
#         op_info[op.get_friendly_name()] = op.get_type_name()
#         # print(f"{op.get_friendly_name()}: {op.get_type_name()}")
#     unique_list = list(set(op_type_name))
#     # print(unique_list)
#     co_location_groups = merge_operations(Computation_G)
#     new_G,group_map = merge_nodes(Computation_G, co_location_groups)
#     em = EmbeddingModel(unique_list)
#     op_embeddings,group_op_map = em.forward(op_info,new_G,group_map,ops,Computation_G)
#     node_mapping = {node: int(node[6:]) for i, node in enumerate(new_G.nodes)}
#     new_G = nx.relabel_nodes(new_G, node_mapping)
#     nfd_centrality = node_dimension(new_G,weight=None).values() #Peiyu: weight=None
#     nfd_centrality_dict = dict(zip(new_G.nodes(), nfd_centrality))    
#     nx.set_node_attributes(new_G, nfd_centrality_dict, 'nfd')

#     closeness_centrality = nx.closeness_centrality(new_G,distance=None).values()
#     closeness_centrality_dict = dict(zip(new_G.nodes(), closeness_centrality))    
#     nx.set_node_attributes(new_G, closeness_centrality_dict, 'closeness')
#     for node in new_G.nodes:
#         new_G.nodes[node]['x'] = torch.cat((op_embeddings[int(node)], torch.tensor([new_G.nodes[node]['nfd']],dtype=op_embeddings.dtype), torch.tensor([new_G.nodes[node]['closeness']],dtype=op_embeddings.dtype), torch.tensor([node], dtype=op_embeddings.dtype)), 0)
    
#     node_mapping = {node: 'Group_'+str(node) for i, node in enumerate(new_G.nodes)}
#     new_G = nx.relabel_nodes(new_G, node_mapping)

#     return new_G,group_op_map,Computation_G


class EmbeddingModel():
    def __init__(self, unique_list):
        self.embedding_size = len(unique_list)
        
        # Create one-hot embeddings
        self.embeddings = nn.ParameterDict({
            op_type: nn.Parameter(self.one_hot_vector(i, self.embedding_size), requires_grad=False)
            for i, op_type in enumerate(unique_list)
        })
        
    def one_hot_vector(self, index, size):
        # Initialize a zero tensor
        one_hot = torch.zeros(size)
        # Set the corresponding index to 1
        one_hot[index] = 1.0
        return one_hot
        
    def forward(self, op_info,new_G,group_map,ops,Computation_G):
        # Example forward pass that aggregates embeddings based on input types
        # print(self.embeddings['Add'])
        aggregated_embedding = self.pre_process(op_info,new_G,group_map,ops,Computation_G)
        return aggregated_embedding
    
    def pre_process(self,op_info,new_G,group_map,ops,Computation_G):
        op_info_name = op_info.keys()
        # print(op_info_name)
        max_size = 4
        # num_nodes = Computation_G.number_of_nodes()
        num_nodes = new_G.number_of_nodes()
        # group_embedding_length = self.embedding_size+max_size+1
        group_embedding_length = self.embedding_size+1
        # group_embedding_length = self.embedding_size+max_size
        group_type_embeddings = {node: torch.zeros(self.embedding_size) for node in new_G.nodes()}  
        group_op_map = {}
        for key, value in group_map.items():
            if value in group_op_map:
                group_op_map[value].append(key)
            else:
                group_op_map[value] = [key]
        # print(group_op_map)
        op_embeddings = torch.zeros([num_nodes,group_embedding_length])
        # print(op_embeddings)

        # one_hot_adjacency = {}
        # for node in Computation_G.nodes(data=True):
        #     one_hot_adjacency[node[1].get("name")] = np.zeros(num_nodes, dtype=int)
        #     # print(type(node[1].get("name")))
        #     for neighbor in Computation_G.neighbors(node[0]):
        #         one_hot_adjacency[node[1].get("name")][int(neighbor) - 1] = 1 
                
        one_hot_adjacency = {}
        for node in new_G.nodes(data=True):
            one_hot_adjacency[node[0]] = np.zeros(num_nodes, dtype=int)
            # print(type(node[1].get("name")))
            for neighbor in new_G.neighbors(node[0]):
                # print(neighbor[5:])
                index_node = int(neighbor[6:])
                # print(node[0])
                one_hot_adjacency[node[0]][index_node] = 1 
                

        for op in ops:
            name = op.get_friendly_name()
            type_embedding = torch.zeros(self.embedding_size)
            # print(name)
            if name in op_info_name:
                type_embedding = self.embeddings[op_info[op.get_friendly_name()]]
                # print(type_embedding)
                try:
                    # print(op_info[op.get_friendly_name()])
                    for i in range(op.get_output_size()):
                        # print(op.get_output_shape(i))
                        shape = list(op.get_output_shape(i))
                        padded_shape = shape + [0] * (max_size - len(shape))
                        # print(padded_shape)
                except Exception as e1:
                    if str(e1) == "get_shape was called on a descriptor::Tensor with dynamic shape":
                        padded_shape = [1,100,100,100]
                        # print(e1)
            elif name[:8] == "Constant":
                # print("Run time model node:")
                # print(name)
                # print(op.get_type_name())
                type_embedding = self.embeddings['Constant']
                for i in range(op.get_output_size()):
                    # print(op.get_output_shape(i))
                    shape = list(op.get_output_shape(i))
                    padded_shape = shape + [0] * (max_size - len(shape))
                    # print(padded_shape)
            else:
                try:
                    shape = list(op.get_output_shape(i))
                    padded_shape = shape + [0] * (max_size - len(shape))
                    # print(padded_shape)
                except:
                    padded_shape = [1,100,100,100]
                
            padded_shape_tensor = torch.tensor(padded_shape, dtype=type_embedding.dtype).detach()
            node_with_feature = None
            for node in Computation_G.nodes(data=True):
                # print(node[1].get("name"))
                # print(name)
                if node[1].get("name") == name:
                    # print(node[1].get("name"))
                    node_with_feature = node[0]
                    # print("Find")
                    break
                
            # if node_with_feature == None:
            #     print("name not detected")
            #     return
            group_name = group_map[node_with_feature]
            # print(group_name)
            
            # if node_with_feature in group_op_map[group_name]:
            group_type_embeddings[group_name] = group_type_embeddings[group_name] + type_embedding
            
            if node_with_feature == group_op_map[group_name][-1]:
                # print(name)
                # print(type_embedding.shape)
                # print(padded_shape_tensor.shape)
                group_type_embeddings[group_name] = group_type_embeddings[group_name]/len(group_op_map[group_name])
                op_embedding = group_type_embeddings[group_name]
                # op_embedding = torch.cat((group_type_embeddings[group_name], padded_shape_tensor), 0)
                # print(op_embedding.shape)
                # print(op_embedding)
                group_index = int(group_name[6:])
                node_id = torch.tensor([group_index], dtype=type_embedding.dtype)
                op_embedding = torch.cat((op_embedding, node_id), 0)
                op_embeddings[group_index] += op_embedding
        return op_embeddings,group_op_map


class PPO(nn.Module):
    def __init__(self, gnn_params, action_dim):
        super(PPO, self).__init__()

        self.gpnn = GPN(gnn_params)
        self.classifier = nn.Sequential(
            nn.Linear(gnn_params['output_channel'], gnn_params['hidden_channel']),
            nn.ReLU(),
            nn.Linear(gnn_params['hidden_channel'], action_dim),
            nn.Softmax(dim=-1)
        )

    def act(self, state):
        clusters, node_ids = self.gpnn(state)
        action_probs = self.classifier(clusters)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action, action_logprob, dist_entropy, clusters, node_ids

    # def evaluate(self, state, action):
    #     clusters, node_ids = self.gpnn(state)
    #     action_probs = self.classifier(clusters)
    #     dist = Categorical(action_probs)
    #     action_logprobs = dist.log_prob(action)
    #     dist_entropy = dist.entropy()
    #
    #     return action_logprobs, dist_entropy

    def evaluate(self, state, action, clusters):
        action_probs = self.classifier(clusters)
        dist = Categorical(action_probs)
        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprob, dist_entropy

    def get_device_placement(self, clusters, node_ids, actions):
        device_placement = []
        for i in range(len(clusters)):
            cluster_device = actions[i].item()
            for node_id in node_ids[i]:
                device_placement.append((node_id, cluster_device))
        return dict(device_placement)

    def sim_runtime(self, p, G, xml_file, core):
        model = core.read_model(model=xml_file)
        ops = model.get_ops()
        # Check if all values are 1
        all_ones = all(value == 1 for value in p.values())
        # Check if all values are 0
        all_zeros = all(value == 0 for value in p.values())
        if all_ones or all_zeros:
            return 1e10
        else:
            for i, node in enumerate(G.nodes()):
                affinity = "CPU" if (p[node] == 0) else "GPU.1"
                ops[i].get_rt_info()["affinity"] = affinity
            try:
                compiled_model = core.compile_model(model, 'HETERO:GPU.1,CPU')
                input_layer = compiled_model.input(0)
                output_layer = compiled_model.output(0)
                input_data = np.random.randn(1, 3, 299, 299)
                run_times = []
                for i in range(10):
                    start_time = time.perf_counter()
                    output = compiled_model({input_layer.any_name: input_data})[output_layer]
                    end_time = time.perf_counter()
                    run_time = end_time - start_time
                    if i > 4:
                        run_times.append(run_time)
                run_time_mean = np.mean(run_times)
                return run_time_mean
            except Exception as e2:
                print(e2)
                return 1e10
            
    # def sim_runtime(self, p, G, xml_file, core):
    #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #     texts = ["Hello, my dog is cute"]
    #     inputs = tokenizer(texts, return_tensors="pt", padding=True)
    #     model = core.read_model(model=xml_file)
    #     ops = model.get_ops()
    #     # Check if all values are 1
    #     all_ones = all(value == 1 for value in p.values())
    #     # Check if all values are 0
    #     all_zeros = all(value == 0 for value in p.values())
    #     if all_ones or all_zeros:
    #         return 1e10
    #     else:
    #         for i, node in enumerate(G.nodes()):
    #             affinity = "CPU" if (p[node] == 0) else "GPU.1"
    #             ops[i].get_rt_info()["affinity"] = affinity
    #         try:
    #             compiled_model = core.compile_model(model, 'HETERO:GPU.1,CPU')
    #             input_layer_1 = compiled_model.input(0)
    #             input_layer_2 = compiled_model.input(1)
    #             output_layer = compiled_model.output(0)
    #             input_data = (inputs['input_ids'], inputs['attention_mask'])
    #             run_times = []
    #             for i in range(10):
    #                 start_time = time.perf_counter()
    #                 output = compiled_model({input_layer_1.any_name: input_data[0], input_layer_2.any_name: input_data[1]})[output_layer]
    #                 end_time = time.perf_counter()
    #                 run_time = end_time - start_time
    #                 if i > 4:
    #                     run_times.append(run_time)
    #             run_time_mean = np.mean(run_times)
    #             del core
    #             return run_time_mean
    #         except Exception as e2:
    #             print(e2)
    #             return 1e10

    def get_runtime(self, graph, device_placement, xml_file,core,group_op_map,Computation_G):
        operation_placement = {}
        for group_id,group_placement in device_placement.items():
            group_name = 'Group_' + str(group_id)
            for op_index in group_op_map[group_name]:
                operation_placement[op_index] = group_placement

        return self.sim_runtime(operation_placement, Computation_G, xml_file,core)

def train(graph, config, core, group_op_map,Computation_G):
    device = torch.device(config['device'])
    config.output_channel = graph.nodes()['Group_0']['x'].shape[0]
    model = PPO(config, config['num_devices']).to(device)

    # weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # PPO parameters
    max_eps = config['max_episodes']
    update_timestep = config['update_timestep']
    K_epochs = config['K_epochs']
    eps_clip = config['eps_clip']
    gamma = config['gamma']

    # Initialize PPO buffer
    buffer = RolloutBuffer()

    best_runtime = float('inf')
    best_placement = None
    best_model_state_dict = None

    training_records = []

    node_to_index = {node: i for i, node in enumerate(graph.nodes())}
    
    initial_graph = graph.copy()
    

    def node_match(n1, n2):
        # Ensure all attributes are checked, including tensors
        if set(n1.keys()) != set(n2.keys()):
            return False
        for key in n1:
            if isinstance(n1[key], torch.Tensor) and isinstance(n2[key], torch.Tensor):
                if not torch.equal(n1[key], n2[key]):
                    return False
            else:
                if n1[key] != n2[key]:
                    return False
        return True

    def edge_match(e1, e2):
        # Ensure all attributes are checked, including tensors
        if set(e1.keys()) != set(e2.keys()):
            return False
        for key in e1:
            if isinstance(e1[key], torch.Tensor) and isinstance(e2[key], torch.Tensor):
                if not torch.equal(e1[key], e2[key]):
                    return False
            else:
                if e1[key] != e2[key]:
                    return False
        return True

    # is_isomorphic = nx.is_isomorphic(graph, initial_graph, node_match=node_match, edge_match=edge_match)
    # print("Graphs are isomorphic:", is_isomorphic)

    state = graph
    for eps in tqdm((range(max_eps))):
        # print(state == initial_graph)
        # print(state == graph)
        state = initial_graph
        # print(state == initial_graph)
        # print(state == graph)
        # is_isomorphic = nx.is_isomorphic(graph, initial_graph, node_match=node_match, edge_match=edge_match)
        # print("Graphs are isomorphic:", is_isomorphic)
        ep_reward = 0
        ep_best_runtime = float('inf')
        ep_best_placement = None

        for t in range(update_timestep):
            node_features = torch.stack([state.nodes[node]['x'] for node in state.nodes()]).to(device)
            edge_index = torch.tensor([(node_to_index[u], node_to_index[v]) for u, v in state.edges()], dtype=torch.long).t().contiguous().to(device)
            batch = torch.zeros(len(state.nodes()), dtype=torch.long).to(device)

            if edge_index.size(1) == 0:
                adj_t = SparseTensor(row=torch.tensor([]), col=torch.tensor([]),
                                    sparse_sizes=(len(state.nodes()), len(state.nodes()))).to(device)
            else:
                dense_adj = to_dense_adj(edge_index).squeeze(0)
                adj_t = SparseTensor.from_dense(dense_adj).to(device)

            state_data = Data(x=node_features, edge_index=edge_index, batch=batch, adj_t=adj_t)

            actions, action_logprobs, dist_entropys, clusters, node_ids = model.act(state_data)
            
            actions_clone = actions.clone().detach()
            clusters_clone = clusters.clone().detach()

            actions_np = actions_clone.cpu().numpy()
            clusters_np = clusters_clone.cpu().numpy()

            device_placement = model.get_device_placement(clusters_np, node_ids, actions_np)
            runtime = model.get_runtime(graph, device_placement, config['graph_path'],core,group_op_map,Computation_G)
            # reward = np.log(model.get_cpu_runtime(config['graph_path'])) - np.log(runtime)
            reward =  1/(runtime)
            ep_reward += reward

            if runtime < ep_best_runtime:
                ep_best_runtime = runtime
                ep_best_placement = device_placement

            buffer.actions.append(actions)
            buffer.states.append(state_data)
            buffer.logprobs.append(action_logprobs)
            buffer.entropys.append(dist_entropys)
            buffer.rewards.append(reward)
            buffer.clusters.append(clusters)
            buffer.node_ids.append(node_ids)
            
            
            # Initialize an empty list to store the flat cluster assignments
            flat_cluster_assignments = []

            # Create a mapping from node indices to their respective group index
            node_to_group = {}
            for group_idx, group in enumerate(node_ids):
                for node in group:
                    node_to_group[node] = group_idx

            # Create the flat_cluster_assignments list by iterating over the sorted node indices
            sorted_nodes = sorted(node_to_group.keys())
            flat_cluster_assignments = [node_to_group[node] for node in sorted_nodes]
            cluster_assignments_tensor = torch.tensor(flat_cluster_assignments)
            final_tensor = clusters_clone[cluster_assignments_tensor]
            node_features = node_features + final_tensor
                        
            for i, node in enumerate(graph.nodes()):
                graph.nodes[node]['x'] = node_features[i]
            

            # Update state
            state = graph
            # is_isomorphic = nx.is_isomorphic(graph, initial_graph, node_match=node_match, edge_match=edge_match)
            # print("Graphs are isomorphic:", is_isomorphic)



        # Update policy
        cumulative_rewards = 0.0
        logprobs = buffer.logprobs
        logprobs = torch.cat(logprobs)
        for i, value in enumerate(buffer.rewards[::-1]):
            cumulative_rewards += value * (config['discount_rate'] ** i)
        
        policy_loss = (-logprobs.squeeze() * (cumulative_rewards)).sum()
        
        # Update policy
        optimizer.zero_grad()
        policy_loss.mean().backward()
        optimizer.step()

        # Clear PPO buffer
        buffer.clear()
        
        output_str = f"Episode {eps}: Average Reward = {ep_reward/update_timestep}, Best Runtime = {ep_best_runtime}"
        
        with open('training results.txt', 'a') as f:
            f.write(output_str + '\n')

        print(f"Episode {eps}: Average Reward = {ep_reward/update_timestep}, Best Runtime = {ep_best_runtime}")
        
        if ep_best_runtime < best_runtime:
            best_runtime = ep_best_runtime
            best_placement = ep_best_placement
            best_model_state_dict = model.state_dict()

        training_records.append({
            'episode': eps,
            'avg_reward': ep_reward/update_timestep,
            'best_runtime': ep_best_runtime,
            'best_placement': ep_best_placement
        })
        
        with open('training_records.json', 'w') as f:
            json.dump(training_records, f)

    print(f"Training completed. Best runtime: {best_runtime}, Best placement: {best_placement}")

    # with open('training_records.json', 'w') as f:
    #     json.dump(training_records, f)

    with open('best_result.json', 'w') as f:
        json.dump({'best_runtime': best_runtime, 'best_placement': best_placement}, f)

    torch.save(best_model_state_dict, 'best_model.pth')
    
def get_available_device(core):
    devices = core.available_devices
    number_of_device = 0
    for device in devices:
        device_name = core.get_property(device, "FULL_DEVICE_NAME")
        print(f"{device}: {device_name}")
        number_of_device += 1
        
    return devices, number_of_device

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # For all GPUs
    np.random.seed(seed)
    # If using other libraries that use randomness, set their seed here


def main():
    core = ov.Core()
    config = Config()
    # Set a seed
    set_seed(42)

    devices,number_of_device = get_available_device(core)
    devices[1] = devices[-1]
    graph,group_op_map,Computation_G = get_final_graph(config['graph_path'],core)

    train(graph, config, core, group_op_map,Computation_G)

if __name__ == "__main__":
    main()