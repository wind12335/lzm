class Config:
    def __init__(self):
        self.graph_path = '/your/xml/file/location'
        self.num_devices = 2

        self.in_channel = 17
        self.output_channel = None
        self.hidden_channel = 128
        self.layer_trans = 2
        self.layer_gnn1 = 2
        self.layer_gnn2 = 1
        self.layer_deepsets = 1
        self.layer_parsingnet = 2
        self.gnn_model = 'GCN'

        self.dropout_network = 0.2
        self.dropout_parsing = 0.0
        self.link_ignore_self_loop = True
        self.act_final = True

        self.learning_rate = 0.001
        self.max_episodes = 25
        self.update_timestep = 20
        self.K_epochs = 5
        self.eps_clip = 0.2
        self.gamma = 0.99
        self.discount_rate = 0.95

        self.device = 'cpu'


    def __getitem__(self, key):
        return getattr(self, key)