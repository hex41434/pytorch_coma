import argparse
import os
import numpy as np
import mesh_operations
import torch
import torch.nn.functional as F
from config_parser import read_config
from data import ComaDataset
from model_vae import ComaVAE
from psbody.mesh import Mesh, MeshViewer
from torch_geometric.data import DataLoader
from transform import Normalize
import torch_geometric.transforms as T
import pytorch_model_summary as pms
from torch.utils.tensorboard import SummaryWriter
from termcolor import colored
import datetime
import shutil

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

def save_model(coma, optimizer, epoch, train_loss, val_loss, save_checkpoint_dir):
    checkpoint = {}
    checkpoint['state_dict'] = coma.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['epoch_num'] = epoch
    checkpoint['train_loss'] = train_loss
    checkpoint['val_loss'] = val_loss
    torch.save(checkpoint, os.path.join(save_checkpoint_dir,'checkpoint_'+ str(epoch)+'.pt'))

def log_main_model_params(folder):
    target = os.path.join(folder,'main_vae.py')
    original = '/work/aifa/MeshAutoencoder/MySource/pytorch_coma/main_vae.py'
    shutil.copyfile(original, target)
    target = os.path.join(folder,'model_vae.py')
    original = '/work/aifa/MeshAutoencoder/MySource/pytorch_coma/model_vae.py'
    shutil.copyfile(original, target)
    print(colored('log of main_vae.py and model_vae.py done!','green'))

def main(args):
    if not os.path.exists(args.conf):
        print('Config not found' + args.conf)

    config = read_config(args.conf)
    print(colored(str(config),'cyan'))

    eval_flag = config['eval']
    
    if not eval_flag: #train mode : fresh or reload
        current_log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        current_log_dir = os.path.join('../Experiments/',current_log_dir)    
    else: #eval mode : save result plys     
        if not args.load_checkpoint_dir:
            print(colored('*****please provide checkpoint file path to reload!*****','red'))
            return #exit if not provided
        else:
            #this folder (_Eval) contains unnecessary informatio and could be removed... 
            current_log_dir = os.path.join('../Experiments/',args.load_checkpoint_dir,'_Eval')
    
    print(colored('logs will be saved in:{}'.format(current_log_dir),'yellow'))

    if args.load_checkpoint_dir:
        load_checkpoint_dir = os.path.join('../Experiments/',args.load_checkpoint_dir,'chkpt')#load last checkpoint 
        print(colored('load_checkpoint_dir: {}'.format(load_checkpoint_dir), 'red'))

    save_checkpoint_dir = os.path.join(current_log_dir , 'chkpt')
    print(colored('save_checkpoint_dir: {}\n'.format(save_checkpoint_dir), 'yellow'))
    if not os.path.exists(save_checkpoint_dir):
        os.makedirs(save_checkpoint_dir)

    with open(os.path.join(current_log_dir, 'config.txt'),'a') as f: 
        for cf in config:
            f.write(str(cf) + ' : ')
            f.write(str(config[cf]) + '\n')

    log_main_model_params(current_log_dir)
           
    print('Initializing parameters')
    template_file_path = config['template_fname']
    template_mesh = Mesh(filename=template_file_path)
    print(template_file_path)

    visualize = config['visualize']
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
        print(colored('\n...cuda is available...\n', 'green'))
    else:
        print(colored('\n...cuda is NOT available...\n', 'red'))

    ds_factors = config['downsampling_factors']
    print('Generating transforms')
    M, A, D, U = mesh_operations.generate_transform_matrices(template_mesh, ds_factors)

    D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
    U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
    A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
    num_nodes = [len(M[i].v) for i in range(len(M))]
    print(colored('number of nodes in encoder : {}'.format(num_nodes),'blue'))

    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = config['data_dir']

    print('*** data loaded from {} ***'.format(data_dir))

    dataset = ComaDataset(data_dir, dtype='train', split=args.split, split_term=args.split_term)
    dataset_test = ComaDataset(data_dir, dtype='test', split=args.split, split_term=args.split_term)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers_thread)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=workers_thread)

    print("x :\n{} for dataset[0] element".format(dataset[0]))
    print(colored(train_loader,'red'))
    print('Loading Model : \n')
    start_epoch = 1
    coma = ComaVAE(dataset, config, D_t, U_t, A_t, num_nodes)

    tbSummWriter = SummaryWriter(current_log_dir)

    print_model_summary = False
    if print_model_summary:
        print(coma)

    mrkdwn = str('<pre><code>'+str(coma)+'</code></pre>')
    tbSummWriter.add_text('tag2', mrkdwn , global_step=None, walltime=None)
    
    #write network architecture into text file 
    logfile = os.path.join(current_log_dir, 'coma.txt')
    my_data_file = open(logfile, 'w')
    my_data_file.write(str(coma))
    my_data_file.close()

    if opt == 'adam':
        optimizer = torch.optim.Adam(coma.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(coma.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise Exception('No optimizer provided')

    if args.load_checkpoint_dir:
        #to load the newest saved checkpoint
        to_back = os.getcwd()
        os.chdir(load_checkpoint_dir)
        chkpt_list = sorted(os.listdir(os.getcwd()), key=os.path.getctime)
        os.chdir(to_back)
        checkpoint_file = chkpt_list[-1]

        logfile = os.path.join(current_log_dir, 'loadedfrom.txt')
        my_data_file = open(logfile, 'w')
        my_data_file.write(str(load_checkpoint_dir))
        my_data_file.close()

        print(colored('\n\nloading Newest checkpoint : {}\n'.format(checkpoint_file),'red'))
        if checkpoint_file:
            checkpoint = torch.load(os.path.join(load_checkpoint_dir,checkpoint_file))
            start_epoch = checkpoint['epoch_num']
            coma.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            #To find if this is fixed in pytorch
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
    coma.to(device)

    for i, dt in enumerate(train_loader):
        dt = dt.to(device)
        graphstr = pms.summary(coma, dt,batch_size=-1, show_input=True , show_hierarchical=False)
        if print_model_summary:
            print(graphstr)

        print(colored('dt in enumerate(train_loader):{} '.format(dt),'green'))
        #write network architecture into text file 
        logfile = os.path.join(current_log_dir, 'pms.txt')
        my_data_file = open(logfile, 'w')
        my_data_file.write(graphstr)
        my_data_file.close()
        
        mrkdwn = str('<pre><code>'+graphstr+'</code></pre>')
        tbSummWriter.add_text('tag', mrkdwn, global_step=None, walltime=None)
        break#for one sample only

    if eval_flag and args.load_checkpoint_dir:
        evaluatedFrom = 'predictedPlys_' + checkpoint_file 
        output_dir = os.path.join('../Experiments/',args.load_checkpoint_dir,evaluatedFrom)#load last checkpoint 
        val_loss = evaluate(coma, test_loader, dataset_test, template_mesh, device, visualize=False, output_dir=output_dir, eval_flag=eval_flag)
        print('val loss', val_loss)
        return

    best_val_loss = float('inf')
    val_loss_history = []

    for epoch in range(start_epoch, total_epochs + 1):
        print("Training for epoch ", epoch)
        print('dataset.len : {}'.format(len(dataset)))
        
        train_loss = train(coma, train_loader, len(dataset), optimizer, device)
        val_loss = evaluate(coma, test_loader, dataset_test, template_mesh, device, visualize=False, output_dir='',eval_flag=eval_flag)#train without visualization
        sample_latent_space(coma,epoch,device,template_mesh,current_log_dir)

        tbSummWriter.add_scalar('Loss/train',train_loss,epoch)
        tbSummWriter.add_scalar('Val Loss/train',val_loss,epoch)
        tbSummWriter.add_scalar('learning_rate',lr,epoch)


        print('epoch ', epoch,' Train loss ', train_loss, ' Val loss ', val_loss)
        if val_loss < best_val_loss:
            save_model(coma, optimizer, epoch, train_loss, val_loss, save_checkpoint_dir)
            best_val_loss = val_loss

        val_loss_history.append(val_loss)
        val_losses.append(best_val_loss)

        if opt=='sgd':
            adjust_learning_rate(optimizer, lr_decay)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    tbSummWriter.flush()
    tbSummWriter.close()    


def loss_function(out,batch_y, mu,logvar):
    l1 = F.l1_loss(out,batch_y)
    # kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
    kl = (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
    # l = 0.9*l1+100*kl
    l = l1+kl
    return l 

def sample_latent_space(coma,epoch, device,template_mesh,current_log_dir):
    nz = 16
    with torch.no_grad():
        sample = torch.randn(1, nz).to(device)
        meshsample = coma.decoder(sample).cpu()
        # print(f'meshsample: {meshsample}')
        pth = os.path.join(current_log_dir,'./sample_plys')
        
        # if epoch % 10 == 0:
        v = meshsample.view(381,3)
        result_mesh = Mesh(v = v, f=template_mesh.f)            
        # result_mesh.write_ply(os.path.join(pth,f'meshsample_{epoch}.ply'))
        result_mesh.write_obj(os.path.join(pth,f'meshsample_{epoch}.obj'))
        
def train(coma, train_loader, len_dataset, optimizer, device):
    coma.train()
    total_loss = 0
    for btch in train_loader:
        btch = btch.to(device)
        optimizer.zero_grad()
        out,mu,logvar = coma(btch)
        loss = loss_function(out,btch.y,mu,logvar)
        #print(f'loss: {loss}')
        # print(colored("\n\nnum_graphs is {} \n\n".format(btch.num_graphs),'blue'))
        total_loss += btch.num_graphs * loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len_dataset


def evaluate(coma , test_loader, dataset, template_mesh, device, visualize, output_dir,eval_flag):
    coma.eval()
    total_loss = 0
    
    for i, data in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            out,mu,logvar = coma(data)
        loss = loss_function(out,data.y,mu, logvar)
        total_loss += data.num_graphs * loss.item()

        if eval_flag and i % 100 == 0:
            # meshviewer = MeshViewer(shape=(1, 2))

            save_out = out.detach().cpu().numpy()
            save_out = save_out*dataset.std.numpy()+dataset.mean.numpy()
            expected_out = (data.y.detach().cpu().numpy())*dataset.std.numpy()+dataset.mean.numpy()
            result_mesh = Mesh(v=save_out, f=template_mesh.f)
            expected_mesh = Mesh(v=expected_out, f=template_mesh.f)

            # meshviewer[0][0].set_dynamic_meshes([result_mesh])
            # meshviewer[0][1].set_dynamic_meshes([expected_mesh])
            # meshviewer[0][0].save_snapshot(os.path.join(output_dir, 'file'+str(i)+'.png'), blocking=True)

            save_mesh = 'obj'
            if save_mesh == 'ply':
                result_mesh.write_ply('{}/result_{}.ply'.format(output_dir,i))
                # expected_mesh.write_ply('{}/expected_{}.ply'.format(output_dir,i))
                print('result meshes are saved as .ply')
            else:
                result_mesh.write_obj('{}/result_{}.obj'.format(output_dir,i))
                # expected_mesh.write_obj('{}/expected_{}.obj'.format(output_dir,i))
                print('result meshes are saved as .obj')
            

    return total_loss/len(dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch Trainer for Convolutional Mesh Autoencoders')
    parser.add_argument('-c', '--conf', help='path of config file')
    parser.add_argument('-s', '--split', default='sliced', help='split can be sliced, expression or identity ')
    parser.add_argument('-st', '--split_term', default='sliced', help='split term can be sliced, expression name '
                                                               'or identity name')
    parser.add_argument('-d', '--data_dir', help='path where the downloaded data is stored')
    parser.add_argument('-cp', '--load_checkpoint_dir', help='path where checkpoints file need to be stored')

    args = parser.parse_args()

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), 'default.cfg')
        print('configuration file not specified, trying to load '
              'it from current directory', args.conf)

    main(args)
