import argparse
import os

import numpy as np

import mesh_operations
import torch
import torch.nn.functional as F
from config_parser import read_config
from data import ComaDataset
from model import Coma
from psbody.mesh import Mesh, MeshViewer
from torch_geometric.data import DataLoader
from transform import Normalize
from sklearn.preprocessing import MinMaxScaler
import torch_geometric.transforms as T
# from mydata import FcadDataset


def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor

def adjust_learning_rate(optimizer, lr_decay):

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_decay

def save_model(coma, optimizer, epoch, train_loss, val_loss, checkpoint_dir):
    checkpoint = {}
    checkpoint['state_dict'] = coma.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['epoch_num'] = epoch
    checkpoint['train_loss'] = train_loss
    checkpoint['val_loss'] = val_loss
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_'+ str(epoch)+'.pt'))

def main(args):
    if not os.path.exists(args.conf):
        print('Config not found' + args.conf)

    config = read_config(args.conf)

    print('Initializing parameters')
    template_file_path = config['template_fname']
    template_mesh = Mesh(filename=template_file_path)
    print(template_file_path)
        
    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
        print(os.path.exists(checkpoint_dir))

    else:
        checkpoint_dir = config['checkpoint_dir']
        print(os.path.exists(checkpoint_dir))
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)

    visualize = config['visualize']
    output_dir = config['visual_output_dir']
    if visualize is True and not output_dir:
        print('No visual output directory is provided. Checkpoint directory will be used to store the visual results')
        output_dir = checkpoint_dir

    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    eval_flag = config['eval']
    lr = config['learning_rate']
    lr_decay = config['learning_rate_decay']
    weight_decay = config['weight_decay']
    total_epochs = config['epoch']
    workers_thread = config['workers_thread']
    opt = config['optimizer']
    batch_size = config['batch_size']
    val_losses, accs, durations = [], [], []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('\ncuda is available...\n')
    else:
        print('\ncuda is NOT available...\n')
    # device = 'cpu'

    # print('Generating transforms')
    # M, A, D, U = mesh_operations.generate_transform_matrices(template_mesh, config['downsampling_factors'])

    # D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
    # U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
    # A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
    # num_nodes = [len(M[i].v) for i in range(len(M))]

    print('\n*** Loading Dataset ***\n')
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = config['data_dir']


    print(data_dir)
    normalize_transform = Normalize()
    # normalize_transform = MinMaxScaler()
    dataset = ComaDataset(data_dir, dtype='train', split=args.split, split_term=args.split_term)
    dataset_test = ComaDataset(data_dir, dtype='test', split=args.split, split_term=args.split_term, pre_transform=normalize_transform)

    # dataset = FcadDataset(data_dir, dtype='train', transform=T.NormalizeScale())

    print('Done ......... \n')

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers_thread)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=workers_thread)

    # print('after normalization')
    # print("mean :{}, \nstd : {} for dataset".format(dataset.mean.numpy() ,dataset.std.numpy()))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch Trainer for Convolutional Mesh Autoencoders')
    parser.add_argument('-c', '--conf', help='path of config file')
    parser.add_argument('-s', '--split', default='sliced', help='split can be sliced, expression or identity ')
    parser.add_argument('-st', '--split_term', default='sliced', help='split term can be sliced, expression name '
                                                               'or identity name')
    parser.add_argument('-d', '--data_dir', help='path where the downloaded data is stored')
    parser.add_argument('-cp', '--checkpoint_dir', help='path where checkpoints file need to be stored')

    args = parser.parse_args()

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), 'default.cfg')
        print('configuration file not specified, trying to load '
              'it from current directory', args.conf)

    main(args)
