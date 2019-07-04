import argparse
import torch
import numpy as np
from model import VAE
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from utils import *
from torch.distributions import kl_divergence, Normal
from torch.nn import functional as F
from params import parameters

p = parameters()

# dimensions
onehot_dim = p.onehot_dim
dy_dim = p.dy_dim
score_1hot_dim = p.score_1hot_dim

# score idx
pitch_idx = p.pitch_idx
ioi_beat_idx = p.ioi_beat_idx
dur_idx = p.dur_idx

# perf idx
durratio_idx = p.durratio_idx - score_1hot_dim
dy_idx = p.dy_idx - score_1hot_dim
ioi_time_idx = p.ioi_time_idx - score_1hot_dim

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
data_dir = parent_dir + '/data/2bars_data/'

def loss_function(recon_x, x, mean, stddev, beta=1):
    recon_x = recon_x.view(recon_x.shape[0], -1)
    x = x[:, score_1hot_dim:]
    recon_x_durratio = recon_x[:, durratio_idx:dy_idx]   
    recon_x_dy = recon_x[:, dy_idx:ioi_time_idx]
    recon_x_ioi = recon_x[:, ioi_time_idx:]
    
    mean_durratio = mean[:, durratio_idx:dy_idx]
    mean_dy = mean[:, dy_idx:ioi_time_idx]
    mean_ioi = mean[:, ioi_time_idx:]

    stddev_durratio = stddev[:, durratio_idx:dy_idx]
    stddev_dy = stddev[:, dy_idx:ioi_time_idx]
    stddev_ioi = stddev[:, ioi_time_idx:]

    x_durratio = x[:, durratio_idx:dy_idx]
    x_dy = x[:, dy_idx:ioi_time_idx]
    x_ioi = x[:, ioi_time_idx:]

    CE_durratio = 0
    KLD_durratio = 0
    CE_dy = 0
    KLD_dy = 0
    CE_ioi = 0
    KLD_ioi = 0

    for i in range(len(x)):
        # print(np.where(x_durratio[i] == 1)[0].shape)
        # print(x_durratio[i].view(-1, x_durratio[i].shape[0]).max(-1)[1])
        target_durratio = x_durratio[i].view(-1, x_durratio[i].shape[0]).max(-1)[1]
        target_dy = x_dy[i].view(-1, x_dy[i].shape[0]).max(-1)[1]
        target_ioi = x_ioi[i].view(-1, x_ioi[i].shape[0]).max(-1)[1]
        CE_durratio += F.nll_loss(
            recon_x_durratio[i].view(-1, recon_x_durratio[i].size(-1)), target_durratio, reduction='elementwise_mean')
        CE_dy += F.nll_loss(
            recon_x_dy[i].view(-1, recon_x_dy[i].size(-1)), target_dy, reduction='elementwise_mean')
        CE_ioi += F.nll_loss(
            recon_x_ioi[i].view(-1, recon_x_ioi[i].size(-1)), target_ioi, reduction='elementwise_mean')
    
    KLD_durratio = beta * kl_divergence(
            Normal(mean_durratio, stddev_durratio),
            Normal(torch.zeros_like(mean_durratio), torch.ones_like(stddev_durratio))).mean()
    KLD_dy = beta * kl_divergence(
            Normal(mean_dy, stddev_dy),
            Normal(torch.zeros_like(mean_dy), torch.ones_like(stddev_dy))).mean()
    KLD_ioi = beta * kl_divergence(
            Normal(mean_ioi, stddev_ioi),
            Normal(torch.zeros_like(mean_ioi), torch.ones_like(stddev_ioi))).mean()
    return CE_durratio + KLD_durratio + CE_dy + KLD_dy + CE_ioi + KLD_ioi

def evaluate(batch):
    model.eval()
    n_batch, n_seq, n_features = batch.shape
    encode_tensor = torch.from_numpy(batch).float()
    target_tensor = torch.from_numpy(batch).float()
    target_tensor = target_tensor.view(-1, target_tensor.size(-1)).max(-1)[1]
    target_tensor = target_tensor
    if torch.cuda.is_available():
        encode_tensor = encode_tensor.cuda()
        target_tensor = target_tensor.cuda()
    recon_batch, mu, logvar = model(encode_tensor)
    loss = loss_function(recon_batch, target_tensor, mu, logvar, step)
    return loss.item()

def train(model, data, step, optimizer, beta, writer):
    model.train()
    n_seq, n_features = data.shape
    assert(n_features == onehot_dim)
    encode_tensor = torch.from_numpy(data).float()
    target_tensor = torch.from_numpy(data[:, durratio_idx:]).float()
    if torch.cuda.is_available():
        encode_tensor = encode_tensor.cuda()
        target_tensor = target_tensor.cuda()
    optimizer.zero_grad()
    recon_x, mean, stddev = model(encode_tensor)
    loss = loss_function(recon_x, target_tensor, mean, stddev, beta)
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), 1)
    optimizer.step()
    step += 1
    print('loss: %.5f' % loss.item())
    writer.add_scalar('loss', loss.item(), step)
    return step


class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hidden", '-hid', type=int, default=512, 
                        help="hidden state dimension")
    parser.add_argument('--epochs', '-e', type=int, default=5, 
                        help="number of epochs")
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, 
                        help="learning rate")
    parser.add_argument('--grudim', '-gd', type=int, default=1024, 
                        help='dimension for gru layer')
    parser.add_argument('--batch_size', '-b', type=int, default=512, 
                        help='input batch size for training')
    parser.add_argument('--name', '-n', type=str, default='embedded', 
                        help='tensorboard visual name')
    parser.add_argument('--decay', '-d', type=float, default=-1, 
                        help='learning rate decay: Gamma')
    parser.add_argument('--beta', type=float, default=0.1, 
                        help='beta for kld')

    args = parser.parse_args()

    hidden_dim = args.hidden
    epochs = args.epochs
    gru_dim = args.grudim
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    decay = args.decay
    beta = args.beta

    folder_name = "hid%d_e%d_gru%d_lr%.4f_batch%d_decay%.4f_beta%.2f" % (  
                    hidden_dim, epochs, gru_dim, learning_rate, batch_size, decay, beta)
    
    writer = SummaryWriter('../logs/{}'.format(folder_name))
    
    # load data
    file_list = find('*.npy', data_dir)
    f = np.load(data_dir + file_list[0])
    note_dim = f.shape[1]

    model = VAE(note_dim, gru_dim, hidden_dim, dy_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if decay > 0:
        scheduler = MinExponentialLR(optimizer, gamma=decay, minimum=1e-5)
    step = 0

    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        model.cuda()
    else:
        print('CPU mode')

    for epoch in range(1, epochs):
        print("#" * 5, epoch, "#" * 5)
        for f in file_list:
            # data are of different length, can we do 
            data = np.load(data_dir + f)
            step = train(model, data, step, optimizer, beta, writer)
            if decay > 0:
                scheduler.step()
        
        save_path = '../params/{}.pt'.format()
    
        

if __name__ == '__main__':
    main()
