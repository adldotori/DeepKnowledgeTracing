import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms

import pandas as pd
import argparse
from tqdm import tqdm
from sklearn import metrics
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DKT(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.4):
        super().__init__()
        self.seq_model = nn.RNN(2 * input_size, hidden_size, dropout=dropout)
        self.decoder = nn.Sequential(
                        nn.Linear(hidden_size, input_size),
                        nn.Sigmoid(),
        )


    def forward(self, seq_in, hidden):
        seq_out, hidden = self.seq_model(seq_in, hidden)
        return self.decoder(seq_out)

def model_test():
    batch_size, input_size, hidden_size, seq_len = 2, 18, 200, 3
    model = DKT(input_size,hidden_size).to(device)
    hidden = torch.randn(1, batch_size, hidden_size).to(device)
    data = torch.randint(0, seq_len, (1, batch_size, 2 * input_size), dtype=torch.float32).to(device)
    res = model(data, hidden)
    print(res.shape)

class Dataset(data.Dataset):
    def __init__(self, mode):
        self.data = self.read_csv(f'data/assistments/builder_{mode}.csv')

    def read_csv(self, file):
        data = []
        self.max_prob = 0
        self.max_sqlen = 0
        with open(file, 'r') as f:
            while f.readline():
                prob = [int(i) for i in f.readline().split(',')[:-1]]
                ans = [int(i) for i in f.readline().split(',')[:-1]]

                data.append((prob, ans))
                for i in prob:
                    if self.max_prob < i + 1:
                        self.max_prob = i + 1
                if self.max_sqlen < len(prob):
                    self.max_sqlen = len(prob)

        final_data = []
        for prob, ans in data:
            prob = torch.tensor(prob).unsqueeze(1)
            prob_onehot = torch.zeros(self.max_sqlen, self.max_prob)
            prob_onehot.scatter_(1, prob, 1)
            correct = prob_onehot.clone()
            for i in range(len(ans)):
                if ans[i] == 0:
                    correct[i] = torch.zeros(self.max_prob)
            emb = torch.cat([prob_onehot, correct], axis=1)
            final_data.append(emb)

        return final_data
    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

def data_test():
    dataset = Dataset('train')
    print(dataset.max_prob)
    print(dataset.max_sqlen)

class Trainer():
    def __init__(self, args, hidden_size):
        self.args = args

        self.train_dataset = Dataset('train')
        self.data_loader = data.DataLoader(\
            self.train_dataset, batch_size=self.args.batch_size
        )

        self.max_sqlen = self.train_dataset.max_sqlen
        self.input_size = self.train_dataset.max_prob
        self.hidden_size = hidden_size

        self.model = DKT(self.input_size, self.hidden_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss = nn.BCELoss()

    def train(self):
        # self.model.load_state_dict(torch.load(f'{self.args.name}.pt'))
        for epoch in range(self.args.epoch):
            pbar = tqdm(self.data_loader)
            loss_sum = 0
            for batch in pbar:
                data = batch.to(device).permute(1,0,2)
                hidden = torch.randn(1,  data.shape[1], self.hidden_size).to(device)
                output = self.model(data[:-1], hidden)

                label = (data[:,:,:data.shape[2]//2]==1)[1:].to(device)
                output = torch.where(label, output, torch.tensor(0.))
                ans = torch.where(label, data[1:,:,data.shape[2]//2:], torch.tensor(-1.))
                loss = self.loss(output[output>0], ans[ans!=-1])

                self.optimizer.zero_grad()
                loss.backward()     
                self.optimizer.step()

                loss_sum += loss
                pbar.set_description(f'Loss : {loss:.2f}')
            
            print(f'Loss : {loss_sum/len(self.data_loader):.2f}')
            torch.save(self.model.state_dict(), f'{self.args.name}.pt')

    def infer(self):
        self.model.load_state_dict(torch.load(f'{self.args.name}.pt'))
        self.model.eval()

        self.test_dataset = Dataset('test')
        self.data_loader = torch.utils.data.DataLoader(\
            self.test_dataset, batch_size=1
        )

        y_true = []
        y_pred = []
        for batch in self.data_loader:
            data = batch.to(device).permute(1,0,2)
            hidden = torch.randn(1,  data.shape[1], self.hidden_size).to(device)
            output = self.model(data[:-1], hidden)
            label = (data[:,:,:data.shape[2]//2]==1)[1:].to(device)
            output = torch.where(label, output, torch.tensor(0.))
            ans = torch.where(label, data[1:,:,data.shape[2]//2:], torch.tensor(-1.))
            y_pred += output[output>0].data.numpy().tolist()
            y_true += ans[ans!=-1].data.numpy().tolist()
        print(metrics.roc_auc_score(np.array(y_true), np.array(y_pred)))

    def seq_op(self, seq, length):
        self.model.load_state_dict(torch.load(f'{self.args.name}.pt'))
        self.model.eval()

        hidden = torch.randn(1,  1, self.hidden_size).to(device)
        
        prob, ans = seq

        for i in range(length):
            emb = self._emb(prob, ans)
            output = self.model(emb, hidden)
            pred_cor = output[-1] 
            max_reward = -1
            max_idx = -1
            for j in range(self.input_size):
                emb = self._emb(prob + [j], ans + [1])
                output = self.model(emb, hidden)
                reward_1 = output[-1].mean()

                emb = self._emb(prob + [j], ans + [0])
                output = self.model(emb, hidden)
                reward_0 = output[-1].mean()
                reward = reward_1 * pred_cor[0,j] + reward_0 * (1 - pred_cor[0,j])
                if reward > max_reward:
                    max_reward = reward
                    max_idx = j
            # print(pred_cor[0,max_idx])
            print(max_reward.data, max_idx)
            prob = prob + [max_idx]
            ans = ans + [1 if pred_cor[0,max_idx]>0.5 else 0]

    def _emb(self, prob, ans):
        prob = torch.tensor(prob).unsqueeze(1)
        prob_onehot = torch.zeros(len(prob), self.input_size)
        prob_onehot.scatter_(1, prob, 1)
        correct = prob_onehot.clone()
        for i in range(len(ans)):
            if ans[i] == 0:
                correct[i] = torch.zeros(self.input_size)
        emb = torch.cat([prob_onehot, correct], axis=1)
        emb.unsqueeze_(1)
        return emb


def get_args():
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--name', type=str, default='base')
    parser.add_argument('-e', '--epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=100)
    parser.add_argument('-m', '--mode', type=str, default='infer')
    args = parser.parse_args()

    return args

def train_test():
    args = get_args()
    trainer = Trainer(args, 200)
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'infer':
        trainer.infer()
    elif args.mode == 'seq_op':
        trainer.seq_op(([90,90,90,90,90,90],[1,1,0,0,1,1]), 10)

if __name__ == '__main__':
    # model_test()
    # data_test()
    train_test()