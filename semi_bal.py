from __future__ import print_function, division
import argparse
from http.client import ImproperConnectionState
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import scipy.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import LoadDataset, cluster_acc, SDCS, NE
import time
import warnings
warnings.filterwarnings("ignore")
import os

class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()

        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)

        self.z_layer = Linear(n_enc_3, n_z)

        # decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)

        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):

        # encoder
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))

        z = self.z_layer(enc_h3)

        # decoder
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z


class IDEC(nn.Module):

    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_enc_3,
                 n_dec_1,
                 n_dec_2,
                 n_dec_3,
                 n_input,
                 n_z,
                 n_clusters,
                 delta=1,
                 pretrain_path=''):
        super(IDEC, self).__init__()
        self.delta = delta
        self.pretrain_path = pretrain_path

        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self):
        if not os.path.exists(self.pretrain_path):
            pretrain_ae(self.ae)
        # load pretrain weights
        self.ae.load_state_dict(torch.load(self.pretrain_path))
        print('load pretrained ae from', self.pretrain_path)

    def forward(self, x):

        x_bar, z = self.ae(x)
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.delta)
        q = q.pow((self.delta + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_bar, q


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def pretrain_ae(model):
    '''
    pretrain autoencoder
    '''
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(50):
        total_loss = 0.
        for batch_idx, (x, _, _) in enumerate(train_loader):
            x = x.to(device)

            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))
        # import pdb; pdb.set_trace()
        torch.save(model.state_dict(), args.pretrain_path)
    print("model saved to {}.".format(args.pretrain_path))


def train(args):
    model = IDEC(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=2000,
        n_dec_1=2000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        delta=1.0,
        pretrain_path=args.pretrain_path).to(device)
    start = time.time()  
    model.pretrain()

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # cluster parameter initiate
    data = dataset.x
    y = dataset.y
    data = torch.Tensor(data).to(device)
    x_bar, hidden = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
    nmi_k = nmi_score(y_pred, y)
    print("nmi score={:.4f}".format(nmi_k))

    hidden = None
    x_bar = None

    # semi-supervised matrix initiate
    with torch.no_grad():
        ml_pairs = []
        cl_pairs = []

        q_ml_list = []
        q_cl_list = []
        
        for _, (_, batch_labels, idx) in enumerate(train_loader):
            mini_batch_size = len(batch_labels)
            n_ml = np.int32(np.floor(mini_batch_size * 0.1));
            n_cl = n_ml;
            ml_pair, cl_pair = generate_pairs(batch_labels, n_ml, n_cl);

            q_ml = np.zeros((mini_batch_size, mini_batch_size))
            q_ml[ml_pair[:,0], ml_pair[:,1]] = 1
            q_ml += q_ml.T

            q_cl = np.zeros((mini_batch_size, mini_batch_size))
            q_cl[cl_pair[:,0], cl_pair[:,1]] = 1
            q_cl += q_cl.T

            ml_pairs.append(ml_pair)
            cl_pairs.append(cl_pair)
            q_ml_list.append(torch.from_numpy(q_ml).float().to(device))
            q_cl_list.append(torch.from_numpy(q_cl).float().to(device))
    
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    max_acc = 0
    model.train()
    gamma = args.gamma

    acc_record = np.zeros((50, 1))
    loss_record = np.zeros((50, 1))
    for epoch in range(50):
        total_loss = 0.


        for batch_idx, (x, l, idx) in enumerate(train_loader):
            x = x.to(device)
            idx = idx.to(device)

            x_bar, y_pred = model(x)           

            
            reconstr_loss = F.mse_loss(x_bar, x)

            # balanced loss (harmonic mean)
            value = torch.sum(y_pred**2, axis=0)
            balanced_loss = torch.sum(1.0/value)
            
            # unormalized semi-supervised loss
            semi_loss = torch.sum(0.5*(1-y_pred)*(torch.mm(q_ml_list[batch_idx].data, y_pred)) + 0.5*y_pred*(torch.mm(q_cl_list[batch_idx].data, y_pred)))


            # import pdb; pdb.set_trace()

            # total loss
            loss = args.alpha * balanced_loss + reconstr_loss + args.beta*semi_loss
            # print( ', Loss {:.4f}'.format(loss.item()))
            total_loss += loss.item()
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update Q_ML
            for i in range(len(ml_pairs[batch_idx])):
                tmp_pair = ml_pairs[batch_idx][i]
                row = tmp_pair[0]; col = tmp_pair[1]
                
                # 限制乘子上限
                tmp = q_ml_list[batch_idx][row, col] + gamma*(1-torch.dot(y_pred[row], y_pred[col]))
                if tmp > 2:
                    q_ml_list[batch_idx][row, col] = q_ml_list[batch_idx][row, col]
                    q_ml_list[batch_idx][col, row] = q_ml_list[batch_idx][col, row]

                else:
                    q_ml_list[batch_idx][row, col] = q_ml_list[batch_idx][row, col] + gamma*(1-torch.dot(y_pred[row], y_pred[col]))
                    q_ml_list[batch_idx][col, row] = q_ml_list[batch_idx][col, row] + gamma*(1-torch.dot(y_pred[row], y_pred[col]))

            q_ml_list[batch_idx] = q_ml_list[batch_idx]



            # update Q_CL
            tmp_cl = torch.zeros_like(q_cl_list[batch_idx]).data
            last_qcl = q_cl_list[batch_idx].data
            for i in range(len(cl_pairs[batch_idx])):
                tmp_pair = cl_pairs[batch_idx][i]
                row = tmp_pair[0]; col = tmp_pair[1]
                tmp = last_qcl[row, col] + gamma*torch.dot(y_pred[row], y_pred[col])
                if tmp > 2:
                    tmp_cl[row, col] = last_qcl[row, col]
                    tmp_cl[col, row] = last_qcl[col, row]
                else:
                    tmp_cl[row, col] = last_qcl[row, col] + gamma*torch.dot(y_pred[row], y_pred[col])
                    tmp_cl[col, row] = last_qcl[col, row] + gamma*torch.dot(y_pred[row], y_pred[col])

                # import pdb; pdb.set_trace()
            # tmp_cl += tmp_cl.T
            q_cl_list[batch_idx] = tmp_cl

        if epoch % args.update_interval == 0:

            _, tmp_q = model(data)

            # update target distribution p
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            # evaluate clustering performance
            y_pred = tmp_q.cpu().numpy().argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            acc = cluster_acc(y, y_pred)
            nmi = nmi_score(y, y_pred)
            ari = ari_score(y, y_pred)
            ne = NE(y, y_pred)
            # import pdb; pdb.set_trace()
            print('Iter {}'.format(epoch), ':acc {:.4f}'.format(acc),
                  ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
                  ', ne {:.4f}'.format(ne), ', Loss {:.4f}'.format(total_loss/(batch_idx + 1)))
            # import pdb; pdb.set_trace()
            if max_acc < acc: 
                max_acc = acc
                
            if epoch > 0 and delta_label < args.tol:
                print('delta_label {:.4f}'.format(delta_label), '< tol',
                      args.tol)
                print('Reached tolerance threshold. Stopping training.')
                break


            acc_record[epoch] = acc
            loss_record[epoch] = total_loss/(batch_idx + 1)

    end = time.time()
    print('Running time: ', end-start)


def generate_pairs(labels, n_ml, n_cl):
    # generate must-link pairs
    ml_pairs = np.zeros((n_ml, 2), dtype=np.int64)

    for row in range(n_ml):
        left = np.random.randint(0,len(labels),1)
        ind = np.where(labels == labels[left])[0]
        ind = np.delete(ind, np.where(ind==left)[0])
        right = np.random.permutation(ind)[0]
        ml_pairs[row] = np.array([left, right], dtype=object)

    # generate cannot-link pairs
    cl_pairs = np.zeros((n_cl, 2), dtype=np.int64)

    for row in range(n_cl):
        left = np.random.randint(0,len(labels),1)
        ind = np.where(labels != labels[left])[0]
        ind = np.delete(ind, np.where(ind==left)[0])
        right = np.random.permutation(ind)[0]
        
        cl_pairs[row] = np.array([left, right], dtype=object)

    return ml_pairs, cl_pairs

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--pretrain_path', type=str, default='data/ae_mnist.pkl')
    parser.add_argument('--alpha', default=0.1, type=float, help='coefficient of clustering loss')
    parser.add_argument('--beta', default=0.1, type=float, help='coefficient of balanced loss')
    parser.add_argument('--gamma', default=1.0, type=float, help='coefficient of learning rate')
    parser.add_argument('--ratio', default=0.1, type=float, help='ratio of #ml')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.dataset == 'mnist':
        args.pretrain_path = 'data/ae_mnist.pkl'
        args.n_clusters = 10
        args.n_input = 784
        args.beta = 0.01
    elif args.dataset == 'usps':
        args.pretrain_path = 'data/ae_usps.pkl'
        args.n_clusters = 10
        args.n_input = 256
    elif args.dataset == 'Reuters':
        args.pretrain_path = 'data/ae_reuters10k.pkl'
        args.n_clusters = 4
        args.n_input = 2000
    elif args.dataset == 'stl-10':
        args.pretrain_path = 'data/ae_stl-10.pkl'
        args.n_clusters = 10
        args.n_input = 2048

    dataset = LoadDataset(args.dataset)

    n_ml = np.int32(np.floor(len(dataset) * args.ratio));
    n_cl = n_ml;
    ml_pairs, cl_pairs = generate_pairs(dataset.y, n_ml, n_cl);

    print(args)
    train(args)
