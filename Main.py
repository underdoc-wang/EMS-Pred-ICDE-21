import os
import argparse
import torch
from torch import nn, optim
import Data_Container, Model_Trainer, DGCN, STIAM



data_dir = './data/EMSTokyo-dict.npz'
dt = 1      # time_slice
epoch = 100
batch_size = 32
learn_rate, weight_decay = 2e-3, 1e-4       # L2 regularization
M_adj = (2, 3)      # num adjs (dynamic, static)
dyn_kernel_config = {'kernel_type':'random_walk_diffusion', 'K':3}
sta_kernel_config = {'kernel_type':'localpool', 'K':1}
loss_opt = 'Huber'



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model with specified model_name; sliding time window')
    parser.add_argument('-device', '--device', type=str, help='Specify device usage',
                        choices=['cpu']+[f'cuda:{gpu}' for gpu in range(4)], default='cuda:2')
    parser.add_argument('-model', '--model_name', type=str, help='Specify model_name',
                        choices=['STIAM_Net'], default='STIAM_Net')
    parser.add_argument('-date', '--dates', type=str, nargs='+',
                        help='Start/end dates of train/test sets. Test follows train.'
                             ' Example: -date 0101 0630 0701 0731',
                        default=['0101', '0630', '0701', '0731'])
    parser.add_argument('-sdw', '--obs_len', type=int, nargs='+',
                        help='Parameters for short-term/daily/weekly observations.'
                             ' Example: -sdw 3 1 1',
                        default=[3, 1, 1])
    args = parser.parse_args()

    # parameters
    model_name = args.model_name
    dates = args.dates
    obs_len = tuple(args.obs_len)

    norm_opt = False if model_name=='SARIMA' or model_name=='VAR' else True

    data_in = Data_Container.DataInput(M_adj=M_adj, data_dir=data_dir, norm_opt=norm_opt)
    data = data_in.load_data()

    print('Seq:', obs_len, 'Keys:', list(data.keys()))
    # prepare static adjs
    sta_adj_list = list()
    for key in list(data.keys()):
        if key.endswith('_adj'):
            adj_preprocessor = DGCN.DyAdj_Preprocessor(**sta_kernel_config)
            b_adj = torch.from_numpy(data[key]).float().unsqueeze(dim=0)    # batch_size=1
            adj = adj_preprocessor.process(b_adj)
            sta_adj_list.append(adj.squeeze(dim=0).to(args.device))     # [(K, N, N)}*M_sta
    assert len(sta_adj_list) == M_adj[1]     # ensure sta adj dim correct

    data_generator = Data_Container.DataGenerator(dt=dt, obs_len=obs_len, val_ratio=0.2, train_test_dates=dates)
    data_loader = data_generator.get_data_loader(data=data, batch_size=batch_size, device=args.device)

    # model
    if model_name == 'STIAM_Net':
        model = STIAM.STIAM_Net(obs_len=obs_len, M_adj=M_adj,
                                dyn_kernel_config=dyn_kernel_config,
                                sta_kernel_config=sta_kernel_config,
                                n_nodes=58, input_dim=1, hidden_dim=16,
                                meta_dim=(24//dt)+7+1, lstm_alter=False)
        model = model.to(args.device)
    else:
        raise ValueError('Unknown model name.')

    if loss_opt == 'MSE':
        loss = nn.MSELoss(reduction='mean')
    elif loss_opt == 'MAE':
        loss = nn.L1Loss(reduction='mean')
    elif loss_opt == 'Huber':
        loss = nn.SmoothL1Loss(reduction='mean')
    else:
        raise Exception('Unknown loss function.')
    optimizer = optim.Adam

    trainer = Model_Trainer.ModelTrainer(model=model, loss=loss, optimizer=optimizer, lr=learn_rate, wd=weight_decay,
                                         n_epochs=epoch, dyn_kernel_config=dyn_kernel_config, device=args.device)

    model_dir = './output'
    os.makedirs(model_dir, exist_ok=True)

    trainer.train(data_loader=data_loader, sta_adj_list=sta_adj_list,
                  modes=['train', 'validate'], model_dir=model_dir)

    print('Test: on Month', dates[2][:2], 'Model', model_name)
    trainer.test(data_loader=data_loader, sta_adj_list=sta_adj_list,
                 modes=['train', 'test'], model_dir=model_dir, data_class=data_in)

