from torch import nn
import torch
from DGCN import DyGCN, GCN



class STIAM_Net(nn.Module):
    def __init__(self, obs_len:tuple, M_adj:tuple, dyn_kernel_config:dict, sta_kernel_config:dict,
                 n_nodes:int, input_dim:int, hidden_dim:int, meta_dim:int):
        super().__init__()

        self.N = n_nodes                    # N
        self.hidden_dim = hidden_dim        # H
        self.total_len = sum(obs_len)       # T
        self.serial_len, self.daily_len, self.weekly_len = obs_len
        self.M_dyn, self.M_sta = M_adj
        self.M = self.M_dyn + self.M_sta    # M
        self.dyn_K = self.get_support_K(dyn_kernel_config)
        self.sta_K = self.get_support_K(sta_kernel_config)

        # initialize layers
        # MGCN
        self.gcn_dyn, self.gcn_sta = nn.ModuleList(), nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        for t in range(self.total_len):
            d_list, s_list = nn.ModuleList(), nn.ModuleList()
            bn = nn.ModuleList()
            for d in range(self.M_dyn):
                d_list.append(DyGCN(K=self.dyn_K, input_dim=input_dim, hidden_dim=hidden_dim, activation=nn.LeakyReLU))
                bn.append(nn.BatchNorm1d(num_features=n_nodes))
            for s in range(self.M_sta):
                s_list.append(GCN(K=self.sta_K, input_dim=input_dim, hidden_dim=hidden_dim, activation=nn.ReLU))
            self.gcn_dyn.append(d_list), self.gcn_sta.append(s_list)
            self.layer_norm.append(bn)

        # embed
        if self.M > 1:
            self.dyn_embed, self.sta_embed = nn.ModuleList(), nn.ModuleList()
            for d in range(self.M_dyn):
                self.dyn_embed.append(nn.Linear(in_features=self.total_len * n_nodes ** 2, out_features=hidden_dim))
            for s in range(self.M_sta):
                self.sta_embed.append(nn.Linear(in_features=n_nodes ** 2, out_features=hidden_dim))
        # STIA-FC-alpha
        self.stia = nn.ModuleDict()
        self.stia['ST'] = nn.ModuleList([nn.Linear(in_features=self.total_len * n_nodes * hidden_dim,
                                                   out_features=hidden_dim, bias=True),  # spatial dim: 5->4
                                         nn.Linear(in_features=hidden_dim * 2,
                                                   out_features=1, bias=True),
                                         nn.Linear(in_features=n_nodes * hidden_dim,
                                                   out_features=meta_dim, bias=True),
                                         nn.Linear(in_features=meta_dim * 2,
                                                   out_features=1, bias=True)])  # temporal dim: 4->3
        self.stia['TS'] = nn.ModuleList([nn.Linear(in_features=self.M * n_nodes * hidden_dim,
                                                   out_features=meta_dim, bias=True),  # temporal dim: 5->4
                                         nn.Linear(in_features=meta_dim * 2,
                                                   out_features=1, bias=True),
                                         nn.Linear(in_features=n_nodes * hidden_dim,
                                                   out_features=hidden_dim, bias=True),
                                         nn.Linear(in_features=hidden_dim * 2,
                                                   out_features=1, bias=True)])  # spatial dim: 4->3
        # output
        self.fc_out = nn.Linear(in_features=hidden_dim, out_features=input_dim, bias=True)

    @staticmethod
    def get_support_K(config:dict):
        if config['kernel_type'] == 'localpool':
            assert config['K'] == 1
            K = 1
        elif config['kernel_type'] == 'chebyshev':
            K = config['K'] + 1
        elif config['kernel_type'] == 'random_walk_diffusion':
            K = config['K'] + 1     #* 2 + 1    # bidirectional diffusion
        else:
            raise ValueError('Invalid kernel_type. Must be one of [chebyshev, localpool, random_walk_diffusion].')
        return K

    def forward(self, x_seq:torch.Tensor, meta:torch.Tensor, dyn_adj_list:list, sta_adj_list:list, hidden=None):
        '''
        MGCN -> ST Interlacing Attention -> output
        :param x_seq: observation sequence - torch.Tensor (batch_size, total_len, N, n_feats)
        :param meta: metadata sequence - torch.Tensor (batch_size, total_len, meta_feats)
        :param dyn_adj_list: [(batch_size, total_len, K_supports, N, N)] * M_dyn
        :param sta_adj_list: [(K_supports, N, N)] * M_sta
        :return: y_pred (t+1) - torch.Tensor (batch_size, n_nodes, n_feats)
        '''
        assert self.M_dyn == len(dyn_adj_list) and self.M_sta == len(sta_adj_list)
        batch_size = x_seq.shape[0]

        step_list = list()
        for t in range(self.total_len):
            t_list = list()
            for d in range(self.M_dyn):
                t_list.append(self.layer_norm[t][d](self.gcn_dyn[t][d](dyn_adj_list[d][:,t,...], x_seq[:,t,...])))
            for s in range(self.M_sta):
                t_list.append(self.gcn_sta[t][s](sta_adj_list[s], x_seq[:, t, ...]))
            t_set = torch.stack(t_list, dim=1)
            step_list.append(t_set)
        h_set = torch.stack(step_list, dim=1)       # (batch, seq, M, N, hidden)

        # embed
        if self.M > 1:
            g_embeds = list()
            g_embeds.extend([self.dyn_embed[d](dyn_adj_list[d][:, :, 1, :, :].reshape(batch_size, -1)) for d in
                             range(self.M_dyn)])
            g_embeds.extend([self.sta_embed[s](sta_adj_list[s][0, :, :].reshape(-1)).repeat(batch_size, 1) for s in
                             range(self.M_sta)])
            g_embeds = torch.stack(g_embeds, dim=1)
        else:
            g_embeds = None

        # STIA
        st_att = self.ST_att(h_set, g_embeds, meta)
        ts_att = self.TS_att(h_set, meta, g_embeds)
        output = st_att + ts_att

        output = output.reshape(batch_size, self.N, self.hidden_dim)
        output = torch.tanh(self.fc_out(output))
        return output


    def ST_att(self, x:torch.Tensor, g_embeds:torch.Tensor, meta:torch.Tensor):
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1, 3, 4)    # (B, M, T, N, H)

        if self.M > 1:
            x_spatial = x.reshape(batch_size, self.M, -1)
            x_spatial = torch.tanh(self.stia['ST'][0](x_spatial))
            a_spatial = torch.tanh(self.stia['ST'][1](torch.cat([x_spatial, g_embeds], dim=-1)))   # (B, M, 1)
            a_spatial = torch.softmax(a_spatial.squeeze(), dim=1)
            x = torch.einsum('bmtnf,bm->bmtnf', [x, a_spatial])
            # x+= self-assignment not allowed for BPTT
            x = x.sum(dim=1)    # (B, T, N, H)
        else:
            x = x.squeeze(dim=1)

        x_temporal = x.reshape(batch_size, self.total_len, -1)
        x_temporal = torch.tanh(self.stia['ST'][2](x_temporal))
        a_temporal = torch.tanh(self.stia['ST'][3](torch.cat([x_temporal, meta], dim=-1)))  # (B, T, 1)
        a_temporal = torch.softmax(a_temporal.squeeze(), dim=1)
        x = torch.einsum('btnf,bt->btnf', [x, a_temporal])
        x = x.sum(dim=1)    # (B, N, H)

        return x

    def TS_att(self, x:torch.Tensor, meta:torch.Tensor, g_embeds:torch.Tensor):
        batch_size = x.shape[0]

        x_temporal = x.reshape(batch_size, self.total_len, -1)
        x_temporal = torch.tanh(self.stia['TS'][0](x_temporal))
        a_temporal = torch.tanh(self.stia['TS'][1](torch.cat([x_temporal, meta], dim=-1)))   # (B, T, 1)
        a_temporal = torch.softmax(a_temporal.squeeze(), dim=1)
        x = torch.einsum('btmnf, bt->btmnf', [x, a_temporal])
        x = x.sum(dim=1)    # (B, M, N, H)

        if self.M > 1:
            x_spatial = x.reshape(batch_size, self.M, -1)
            x_spatial = torch.tanh(self.stia['TS'][2](x_spatial))
            a_spatial = torch.tanh(self.stia['TS'][3](torch.cat([x_spatial, g_embeds], dim=-1)))   # (B, M, 1)
            a_spatial = torch.softmax(a_spatial.squeeze(), dim=1)
            x = torch.einsum('bmnf,bm->bmnf', [x, a_spatial])
            x = x.sum(dim=1)    # (B, N, H)
        else:
            x = x.squeeze(dim=1)

        return x


