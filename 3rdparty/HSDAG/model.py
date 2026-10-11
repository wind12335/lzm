import torch
from torch.nn import Module
from torch_sparse import SparseTensor
from layer import GNN, MLP
from pool import global_mean_pool
import torch_sparse


class GPN(Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.device = params['device']

        from layer import ParsingNet_GPU as ParsingNet


        gnn_model=params['gnn_model']

        self.input_trans = MLP(in_channel=params['in_channel'], hidden_channel=params['hidden_channel'], out_channel=params['hidden_channel'], num_layers=params['layer_trans'], dropout=params['dropout_network'], norm_mode='insert', act_final=params['act_final'])

        self.gnn1 = GNN(hidden_channel=params['hidden_channel'], num_layers=params['layer_gnn1'], dropout=params['dropout_network'], gnn_model=gnn_model, act_final=params['act_final'])

        self.parsing_net = ParsingNet(channel=params['hidden_channel'], dropout_network=params['dropout_network'], dropout_parsing=params['dropout_parsing'], layer_parsingnet=params['layer_parsingnet'], link_ignore_self_loop=params['link_ignore_self_loop'])

        if params['layer_gnn2']=="share":
            self.gnn2 = self.gnn1
            layer_gnn2 = params['layer_gnn1']
        else:
            if params['layer_gnn2']=="follow":
                layer_gnn2 = params['layer_gnn1']
            else:
                layer_gnn2 = params['layer_gnn2']
            self.gnn2 = GNN(hidden_channel=params['hidden_channel'], num_layers=layer_gnn2, dropout=params['dropout_network'], gnn_model=gnn_model, act_final=params['act_final'])

        self.deepsets_pre = MLP(in_channel=params['hidden_channel'], hidden_channel=params['hidden_channel'], out_channel=params['hidden_channel'], num_layers=params['layer_deepsets'], dropout=params['dropout_network'], norm_mode='post', act_final=params['act_final'])
        self.deepsets_post = MLP(in_channel=params['hidden_channel'], hidden_channel=params['hidden_channel'], out_channel=params['hidden_channel'], num_layers=params['layer_deepsets'], dropout=params['dropout_network'], norm_mode='post', act_final=params['act_final'])

        self.output_trans = MLP(in_channel=params['hidden_channel'], hidden_channel=params['hidden_channel'], out_channel=params['output_channel'], num_layers=params['layer_trans'], dropout=params['dropout_network'], norm_mode='None', act_final=params['act_final'])


    def forward(self, data):
        h = data.x
        adj_t = data.adj_t
        batch = data.batch
        batch_size = torch.max(batch).item()+1

        assignments = []

        h = self.input_trans(h)

        if not adj_t.is_symmetric():

            row, col, value = adj_t.coo()
            adj_t = SparseTensor(row=torch.cat([row, col]), col=torch.cat([col, row]),
                                           value=torch.cat([value, value]),
                                           sparse_sizes=adj_t.sparse_sizes()).to(self.device)

        assert adj_t.is_symmetric()

        node_ids = torch.arange(h.size(0), device=h.device)

        flag = True
        while flag:
            h_init = h
            adj_t_init = adj_t

            h_gnn1 = self.gnn1(h_init, adj_t_init)
            s, adj_t, batch, mask1, mask2, flag, node_score, link_counts = self.parsing_net(h_gnn1, adj_t, batch)

            if self.params['layer_gnn2']=="share":
                h_gnn2 = h_gnn1
            else:
                h_gnn2 = self.gnn2(h_init, adj_t_init)

            if flag==True or batch.shape[0]!=batch_size:
                h = self.deepsets_pre(h_gnn2)
                h = s.t() @ h
                h = self.deepsets_post(h)

                h = h * node_score.view(-1,1)
                h = h * link_counts.view(-1,1)

                h[mask2,:] = h_init[mask1,:]
                assignments.append(s)

        # node_ids = []
        # for assignment in assignments:
        #     node_ids.append(assignment.coo()[1])
            _, cluster_ids, _ = s.coo()
            node_ids = cluster_ids[node_ids]

        M = max(node_ids) + 1

        node_ids_list = [[] for _ in range(M)]

        for node_id, value in enumerate(node_ids):
            node_ids_list[value].append(node_id)

        h = self.output_trans(h)
        return h, node_ids_list