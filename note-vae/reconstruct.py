from model import VAE
import torch
import numpy as np
import argparse
from params import parameters
from utils import *
from torch import LongTensor
from matrix2midi import synthesize

p = parameters()
score_1hot_dim = p.score_1hot_dim

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
data_dir = parent_dir + '/data/2bars_data/'
model_params_dir = parent_dir + '/params/'

def evaluate(model, batch, seq_len, output_dir, name):
    model.eval()
    n_seq, _, n_features = batch.shape
    score = batch[:, :, :score_1hot_dim].view(n_seq, -1)
    encode_tensor = batch
    target_tensor = batch
    if torch.cuda.is_available():
        encode_tensor = encode_tensor.cuda()
        target_tensor = target_tensor.cuda()
    recon_x, mean, stddev = model(encode_tensor, seq_len)
    recon_x = recon_x[0].view(recon_x[0].shape[0], -1)
    target = target_tensor[:, :, score_1hot_dim:]
    target = target.view(recon_x.shape[0], -1)
    if not os.path.exists(output_dir) or not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    synthesize(recon_x, score, target, output_dir, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", '-hid', type=int, default=768, 
                        help="hidden state dimension")
    parser.add_argument('--epochs', '-e', type=int, default=5, 
                        help="number of epochs")
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, 
                        help="learning rate")
    parser.add_argument('--grudim', '-gd', type=int, default=1024, 
                        help='dimension for gru layer')
    parser.add_argument('--batch_size', '-b', type=int, default=64, 
                        help='input batch size for training')
    parser.add_argument('--name', '-n', type=str, default='embedded', 
                        help='tensorboard visual name')
    parser.add_argument('--decay', '-d', type=float, default=-1, 
                        help='learning rate decay: Gamma')
    parser.add_argument('--beta', type=float, default=0.1, 
                        help='beta for kld')
    parser.add_argument('--data', type=int, default=1000,
                        help='how many pieces of music to use')

    args = parser.parse_args()

    hidden_dim = args.hidden
    epochs = args.epochs
    gru_dim = args.grudim
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    decay = args.decay
    beta = args.beta
    data_num = args.data

    index = 0
    # load data 
    file_list = find('*.npy', data_dir)
    f = np.load(data_dir + file_list[index])
    note_dim = f.shape[1]

    param_name = "hid%d_e%d_gru%d_lr%.4f_batch%d_decay%.4f_beta%.2f_data%d_epoch4" % (  
                    hidden_dim, epochs, gru_dim, learning_rate, batch_size, decay, beta, data_num)

    model = VAE(note_dim, gru_dim, hidden_dim, batch_size)
    model.load_state_dict(torch.load(model_params_dir + param_name + '.pt'))

    # load song
    data = np.load(data_dir + file_list[index])
    seq_len = LongTensor([len(data)])
    batch = torch.from_numpy(data)
    batch = batch.view(-1, 1, note_dim)
    print(batch.size())
    
    output_dir = parent_dir + '/results/' + param_name + '/'
    evaluate(model, batch, seq_len, output_dir, file_list[index][:-4])
    

    

if __name__ == '__main__':
    main()
