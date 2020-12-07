import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms

import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DKT(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.4):
        super().__init__()
        self.seq_model = nn.RNN(input_size, hidden_size, dropout=dropout)
        self.decoder = nn.Sequential(
                        nn.Linear(hidden_size, input_size),
                        nn.Sigmoid(),
        )


    def forward(self, seq_in, hidden):
        seq_out, hidden = self.seq_model(seq_in, hidden)
        return self.decoder(seq_out)

def model_test():
    batch_size, input_size, hidden_size, seq_len = 2, 18, 200, 3
    model = DKT(2 * input_size,hidden_size).to(device)
    hidden = torch.randn(1, batch_size, hidden_size).to(device)
    data = torch.randint(0, seq_len, (1, batch_size, 2 * input_size), dtype=torch.float32).to(device)
    res = model(data, hidden)
    print(res.shape)

class Dataset(data.Dataset):
    def __init__(self):
        self.data = []
        max_prob = 0
        with open('data/assistments/builder_train.csv', 'r') as f:
            while f.readline():
                prob = [int(i) for i in f.readline().split(',')[:-1]]
                ans = [int(i) for i in f.readline().split(',')[:-1]]

                self.data.append((prob, ans))
                for i in prob:
                    if max_prob < i + 1:
                        max_prob = i + 1
        
        self.train_data = []
        for prob, ans in self.data:
            prob = torch.tensor(prob).unsqueeze(1)
            prob_onehot = torch.zeros(len(prob), max_prob)
            prob_onehot.scatter_(1, prob, 1)
            for i in ans:
                if i == 0:
                    correct = torch.zeros(len(prob), max_prob)
                else:
                    correct = prob_onehot
            data = torch.cat([prob_onehot, correct], axis=1)
            self.train_data.append(data)
        
    def __getitem__(self, i):
        return self.train_data[i]

    def __len__(self):
        return len(self.train_data)
        
def data_test():
    dataset = Dataset()

if __name__ == '__main__':
    # model_test()
    data_test()

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import os
# import os.path as osp
# import argparse
# import numpy as np

# from torch.utils.tensorboard import SummaryWriter
# from torchvision.utils import make_grid

# from model import *
# from loss import *
# from dataloader import *

# def get_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-n', '--num-workers', type=int, default = 4)
#     parser.add_argument('-e', '--epoch', type=int, default=20)
#     parser.add_argument('-b', '--batch-size', type=int, default = 256)
#     parser.add_argument('-d', '--display-step', type=int, default = 500)
#     parser.add_argument('--dataset', type=str, default = 'mnist', help='mnist or celeba')
#     parser.add_argument('-m', '--mode', type=str, default = 'local', help='local or colab')
#     opt = parser.parse_args()
#     return opt

# def train(opt):
#     # Init Model
#     generator = Generator(opt.dataset).cuda()
#     discriminator = Discriminator(opt.dataset).cuda()
#     discriminator.train()

#     # Load Dataset
#     dataset = Dataset(opt.dataset)
#     data_loader = Dataloader(opt, dataset)

#     # Set Optimizer
#     optim_gen = torch.optim.Adam(generator.parameters(), lr=0.0002)
#     optim_dis = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

#     # Set Loss
#     loss = Loss()

#     writer = SummaryWriter()

#     if opt.mode == 'local':
#         save_dir = 'checkpoints'
#     elif opt.mode == 'colab':
#         save_dir = '/content/gdrive/My Drive/checkpoints/'
#     if not osp.isdir(save_dir):
#         os.makedirs(save_dir)

#     for epoch in range(opt.epoch):
#         for i in range(len(data_loader.data_loader)):
#             step = epoch * len(data_loader.data_loader) + i + 1
#             # load dataset only batch_size
#             image, label = data_loader.next_batch()
#             image = image.cuda()
#             batch_size = image.shape[0]

#             # train discriminator
#             optim_dis.zero_grad()

#             noise = Variable(torch.randn(batch_size, 100)).cuda()
#             gen = generator(noise)

#             validity_real = discriminator(image)
#             loss_dis_real = loss(validity_real, Variable(torch.ones(batch_size,1)).cuda())

#             validity_fake = discriminator(gen.detach())
#             loss_dis_fake = loss(validity_fake, Variable(torch.zeros(batch_size,1)).cuda())

#             loss_dis = (loss_dis_real + loss_dis_fake) / 2
#             loss_dis.backward()
#             optim_dis.step()

#             # train generator
#             generator.train()
#             optim_gen.zero_grad()

#             noise = Variable(torch.randn(batch_size, 100)).cuda()
            
#             gen = generator(noise)
#             validity = discriminator(gen)
            
#             loss_gen = loss(validity, Variable(torch.ones(batch_size,1)).cuda())
#             loss_gen.backward()
#             optim_gen.step()

#             writer.add_scalar('loss/gen', loss_gen, step)
#             writer.add_scalar('loss/dis', loss_dis, step)
#             writer.add_scalar('loss/dis_real', loss_dis_real, step)
#             writer.add_scalar('loss/dis_fake', loss_dis_fake, step)
            
#             if step % opt.display_step == 0:
#                 writer.add_images('image', image[0][0], step, dataformats="HW")
#                 writer.add_images('result', gen[0][0], step, dataformats="HW")

#                 print('[Epoch {}] G_loss : {:.2} | D_loss : {:.2}'.format(epoch + 1, loss_gen, loss_dis))
                
#                 generator.eval()
#                 z = Variable(torch.randn(9, 100)).cuda()
#                 sample_images = generator(z)
#                 grid = make_grid(sample_images, nrow=3, normalize=True)
#                 writer.add_image('sample_image', grid, step)

#                 torch.save(generator.state_dict(), osp.join(save_dir, 'checkpoint_{}.pt'.format(step)))

# if __name__ == '__main__':
#     # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
#     opt = get_opt()
#     train(opt)