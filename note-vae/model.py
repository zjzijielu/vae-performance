import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from params import parameters

'''
based on Ruihan Yang's code
'''

p = parameters()
# dimensions
score_1hot_dim = p.score_1hot_dim
perf_1hot_dim = p.perf_1hot_dim
durratio_dim = p.durratio_dim
dy_dim = p.dy_dim
ioi_time_dim = p.ioi_time_dim

# score idx
pitch_idx = p.pitch_idx
ioi_beat_idx = p.ioi_beat_idx
dur_idx = p.dur_idx

# perf idx
durratio_idx = p.durratio_idx - score_1hot_dim
dy_idx = p.dy_idx - score_1hot_dim
ioi_time_idx = p.ioi_time_idx - score_1hot_dim


class VAE(nn.Module):
    def __init__(self, roll_dims, hidden_dims,
                 z_dims, batch_size, k=1500):
        super(VAE, self).__init__()
        self.gru_0 = nn.GRU(roll_dims, hidden_dims, batch_first=False, bidirectional=True)
        self.linear_mu = nn.Linear(hidden_dims * 2, z_dims)
        self.linear_var = nn.Linear(hidden_dims * 2, z_dims)
        self.grucell_0 = nn.GRUCell(z_dims + roll_dims, hidden_dims)
        self.grucell_1 = nn.GRUCell(hidden_dims, hidden_dims)
        self.grucell_2 = nn.GRUCell(hidden_dims, hidden_dims)
        self.linear_out = nn.Linear(hidden_dims, perf_1hot_dim)
        self.linear_init = nn.Linear(z_dims, hidden_dims)
        self.bn = nn.BatchNorm2d(1) # as channel is 1
        self.roll_dims = roll_dims
        self.hidden_dims = hidden_dims
        self.eps = 1
        self.batch_size = batch_size
        self.z_dims = z_dims
        self.sample = None
        self.iteration = 0
        self.k = torch.FloatTensor([k])

    def _sampling(self, x):
        idx = x.max(1)[1]
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            arange = arange.cuda()  
        x[arange, idx] = 1
        return x

    def encode(self, x, x_lengths):
        '''
        in our case, x denotes all notes in 2 bars (length is not fixed)
        shape of x: n * onehot_dim, n being the number of notes

        result:
            hidden: 1 * hidden_dim
            x: 1 * 

        if gru.batch_first == false, data dim: 
            [n_seq, batch, feature_len]
        else
            [batch, n_seq, feature_len]
        '''
        # self.gru_0.flatten_parameters()
        print(len(x_lengths))
        n_seq = x.shape[0]
        hidden = torch.zeros((2, len(x_lengths), self.hidden_dims))

        x = x.float()
        X = nn.utils.rnn.pack_padded_sequence(x, x_lengths)

        X, hidden = self.gru_0(X, hidden)
        # x, hidden = self.gru_0(x.view(n_seq, 1, -1), hidden)
        # x = self.gru_0(x)[-1]
        # x = x.transpose_(0, 1).contiguous()
        # x = x.view(x.size(0), -1)
        hidden = hidden.view(len(x_lengths), 1, -1, 1)
        print(hidden.size())
        hidden = self.bn(hidden).view(len(x_lengths), -1)
        # mean = self.linear_mu(x)
        mean = self.linear_mu(hidden)
        stddev = (self.linear_var(hidden) * 0.5).exp_()
        return mean, stddev

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, Z, score, x_lengths):
        '''
        in vanilla VAE, we don't have hierarchical structure
        z will have shape 1 * z_dims

        input:
            1. representation z (batch_size, z_dims) 
            2. score pitch (n_seq, score_dim)
            
        output: 
            predicted perf (n_seq, perf_dim) for a batch
        '''
        # out = torch.zeros((z.size(0), self.roll_dims))
        outs = []

        for i in range(len(x_lengths)):
            steps = x_lengths[i]
            out = torch.zeros((1, perf_1hot_dim))
            score_out = torch.zeros((1, score_1hot_dim))
            z = Z[i:i+1, :]
            # start with rest 
            out[:, -1] = 1. # TODO: rest is not defined
            score_out[:, -1] = 1 # TODO: rest is not defined
            hx = [None, None, None]
            x = []
            t = F.tanh(self.linear_init(z))
            hx[0] = t # the initial hidden state
            if torch.cuda.is_available():
                out = out.cuda()
            for j in range(steps):
                # at each step we concatenate score, z, and 
                # the predicted result from the last step
                assert(out.size(-1) == perf_1hot_dim)
                out = out.view(1, -1)
                assert(score_out.size(-1) != 0)
                out = torch.cat([out.float(), score_out.float(), z], 1)
                hx[0] = self.grucell_0(
                    out, hx[0])
                if j == 0:
                    hx[1] = hx[0]
                hx[1] = self.grucell_1(
                    hx[0], hx[1])
                if j == 0:
                    hx[2] = hx[1]
                hx[2] = self.grucell_2(
                    hx[1], hx[2])
                hx_out = self.linear_out(hx[2])
                out = torch.zeros((1, perf_1hot_dim))
                out[:, durratio_idx:dy_idx] = F.log_softmax(hx_out[:, durratio_idx:dy_idx], 1)
                out[:, dy_idx:ioi_time_idx] = F.log_softmax(hx_out[:, dy_idx:ioi_time_idx], 1)
                out[:, ioi_time_idx:] = F.log_softmax(hx_out[:, ioi_time_idx:], 1)
                x.append(out)
                score_out = score[j, i:i+1, :]
                if self.training:
                    p = torch.rand(1).item()
                    if p < self.eps:
                        out = self.sample[j, i, durratio_idx+score_1hot_dim:]
                    else:
                        out = self._sampling(out)
                    # print(out.shape)
                    self.eps = self.k / \
                        (self.k + torch.exp(self.iteration / self.k))
                    self.iteration += 1
                else:
                    out = self._sampling(out)
            outs.append(torch.stack(x, 0))
        return outs


    def forward(self, x, x_lengths):
        score = x[:, :, :score_1hot_dim]
        if self.training:
            self.sample = x.clone()
        means, stddevs = self.encode(x, x_lengths)
        assert(means.size(0) == len(x_lengths))
        assert(stddevs.size(0) == len(x_lengths))
        z = torch.zeros((len(x_lengths), self.z_dims))
        dis_mean = torch.zeros((len(x_lengths), self.z_dims))
        dis_stddev = torch.zeros((len(x_lengths), self.z_dims))
        if self.training:
            for i in range(len(x_lengths)):
                dis = Normal(means[i], stddevs[i])
                z[i] = dis.rsample()
                dis_mean[i] = dis.mean
                dis_stddev[i] = dis.stddev
        else:
            for i in range(len(x_lengths)):
                z[i] = Normal(means[i], stddevs[i]).mean
        assert(z.size(0) == len(x_lengths))
        return self.decode(z, score, x_lengths), dis_mean, dis_stddev


class VAE_disentangle(nn.Module):
    def __init__(self, roll_dims, hidden_dims,
                 z_dims, dy_dims, k=1500):
        super(VAE, self).__init__()
        self.gru_0 = nn.GRU(roll_dims, hidden_dims, batch_first=False, bidirectional=True)
        self.linear_mu = nn.Linear(hidden_dims * 2, z_dims)
        self.linear_var = nn.Linear(hidden_dims * 2, z_dims)
        self.grucell_0 = nn.GRUCell(z_dims + dy_dims, hidden_dims)
        self.grucell_1 = nn.GRUCell(hidden_dims, hidden_dims)
        self.grucell_2 = nn.GRUCell(hidden_dims, hidden_dims)
        self.linear_out_dy = nn.Linear(hidden_dims, dy_dims)
        self.linear_init = nn.Linear(z_dims, hidden_dims)
        self.roll_dims = roll_dims
        self.dy_dims = dy_dims
        self.hidden_dims = hidden_dims
        self.eps = 1
        self.sample = None
        self.iteration = 0
        self.k = torch.FloatTensor([k])

    def _sampling(self, x):
        idx = x.max(1)[1]
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            arange = arange.cuda()
        x[arange, idx] = 1
        return x

    def encode(self, x):
        '''
        in our case, x denotes all notes in 2 bars (length is not fixed)
        shape of x: n * onehot_dim, n being the number of notes

        result:
            hidden: 1 * hidden_dim
            x: 1 * 

        if gru.batch_first == false, data dim: 
            [n_seq, batch, feature_len]
        else
            [batch, n_seq, feature_len]
        '''
        # self.gru_0.flatten_parameters()
        n_seq = x.shape[0]
        hidden = torch.zeros((2, 1, self.hidden_dims))
        
        x, hidden = self.gru_0(x.view(n_seq, 1, -1), hidden)
        # x = self.gru_0(x)[-1]
        # x = x.transpose_(0, 1).contiguous()
        # x = x.view(x.size(0), -1)
        hidden = hidden.view(1, -1)
        # mean = self.linear_mu(x)
        mean = self.linear_mu(hidden)
        stddev = (self.linear_var(hidden) * 0.5).exp_()
        return Normal(mean, stddev)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_dy(self, z, score_pitch):
        '''
        for now decode is not conditioned on score, and we only decode dynamic
        z will have shape 1 * z_dims

        input:
            1. representation z
            2. score pitch (n_seq, 128)
        return 
        '''
        # out = torch.zeros((z.size(0), self.roll_dims))
        out = torch.zeros((1, self.dy_dims))
        steps = score_pitch.shape[0]
        out[:, -1] = 1.
        x, hx = [], [None, None, None]
        t = F.tanh(self.linear_init(z))
        hx[0] = t
        if torch.cuda.is_available():
            out = out.cuda()
        for i in range(steps):
            print("#" * 5, 'step', i)
            out = out.view(1, -1)
            print(out.shape, z.shape)
            out = torch.cat([out, z], 1)
            hx[0] = self.grucell_0(
                out, hx[0])
            if i == 0:
                hx[1] = hx[0]
            hx[1] = self.grucell_1(
                hx[0], hx[1])
            if i == 0:
                hx[2] = hx[1]
            hx[2] = self.grucell_2(
                hx[1], hx[2])
            out = F.log_softmax(self.linear_out_dy(hx[2]), 1)
            print("out shape")
            print(out.shape)
            x.append(out)
            if self.training:
                p = torch.rand(1).item()
                if p < self.eps:
                    out = self.sample[i, dy_idx:tempo_idx]
                else:
                    out = self._sampling(out)[dy_idx:tempo_idx]
                print(out.shape)
                self.eps = self.k / \
                    (self.k + torch.exp(self.iteration / self.k))
                self.iteration += 1
            else:
                out = self._sampling(out)
        print("#" * 10, "x")
        print(x)
        return torch.stack(x, 0)

    def decode_global(self, z):
        return


    def forward(self, x):
        score_pitch = x[:, 0:128]
        if self.training:
            self.sample = x.clone()
        dis = self.encode(x)
        if self.training:
            z = dis.rsample()
        else:
            z = dis.mean
        return self.decode_dy(z, score), dis.mean, dis.stddev