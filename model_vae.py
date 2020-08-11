import torch
import torch.nn.functional as F

from layers import ChebConv_Coma, Pool

class ComaVAE(torch.nn.Module):

    def __init__(self, dataset, config, downsample_matrices, upsample_matrices, adjacency_matrices, num_nodes):
        super(ComaVAE, self).__init__()
        self.n_layers = config['n_layers']
        self.filters = config['num_conv_filters']
        self.filters.insert(0, dataset.num_features)  # To get initial features per node
        self.K = config['polygon_order']
        self.z = config['z']
        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices
        self.A_edge_index, self.A_norm = zip(*[ChebConv_Coma.norm(self.adjacency_matrices[i]._indices(),
                                                                  num_nodes[i]) for i in range(len(num_nodes))])
        self.cheb = torch.nn.ModuleList([ChebConv_Coma(self.filters[i], self.filters[i+1], self.K[i])
                                         for i in range(len(self.filters)-2)])
        self.cheb_dec = torch.nn.ModuleList([ChebConv_Coma(self.filters[-i-1], self.filters[-i-2], self.K[i])
                                             for i in range(len(self.filters)-1)])
        self.cheb_dec[-1].bias = None  # No bias for last convolution layer
        self.pool = Pool()
        self.enc_lin1 = torch.nn.Linear(self.downsample_matrices[-1].shape[0]*self.filters[-1], self.z)
        self.enc_lin2 = torch.nn.Linear(self.downsample_matrices[-1].shape[0]*self.filters[-1], self.z)
        self.dec_lin = torch.nn.Linear(self.z, self.filters[-1]*self.upsample_matrices[-1].shape[1])
        self.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch_size = data.num_graphs
        x = x.reshape(batch_size, -1, self.filters[0])
        mu,logvar = self.encoder(x)
        #print([mu,logvar])
        z = self.reparameterize(mu, logvar)
        #print(torch.mean(z))
        x = self.decoder(z)
        x = x.reshape(-1, self.filters[0])
        return x,mu,logvar

    def encoder(self, x):
        for i in range(self.n_layers):
            # x = F.relu(self.cheb[i](x, self.A_edge_index[i], self.A_norm[i]))
            x = F.sigmoid(self.cheb[i](x, self.A_edge_index[i], self.A_norm[i]))
            x = self.pool(x, self.downsample_matrices[i])
        x = x.reshape(x.shape[0], self.enc_lin1.in_features)
        # x1 = F.relu(self.enc_lin1(x))
        # x2 = F.relu(self.enc_lin2(x))
        x1 = F.sigmoid(self.enc_lin1(x))
        x2 = F.sigmoid(self.enc_lin2(x))
        return x1,x2

    def decoder(self, x):
        #x = F.relu(self.dec_lin(x))
        x = F.sigmoid(self.dec_lin(x))
        x = x.reshape(x.shape[0], -1, self.filters[-1])
        for i in range(self.n_layers):
            x = self.pool(x, self.upsample_matrices[-i-1])
            # x = F.sigmoid(self.cheb_dec[i](x, self.A_edge_index[self.n_layers-i-1], self.A_norm[self.n_layers-i-1]))
            x = F.sigmoid(self.cheb_dec[i](x, self.A_edge_index[self.n_layers-i-1], self.A_norm[self.n_layers-i-1]))
        x = self.cheb_dec[-1](x, self.A_edge_index[-1], self.A_norm[-1])
        return x

    def reparameterize(self, mu, logvar):
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
    
    def reset_parameters(self):
        torch.nn.init.normal_(self.enc_lin1.weight, 0, 0.1)
        torch.nn.init.normal_(self.enc_lin2.weight, 0, 0.1)
        torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)

