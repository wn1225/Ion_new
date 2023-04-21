import argparse
from itertools import product
import os

# 指定使用0,1,2三块卡
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

torch.set_printoptions(profile="full")
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
import numpy as np
from torch.nn import Sequential as Seq, Dropout, Linear as Lin, ReLU, BatchNorm1d as BN, LayerNorm as LN, Sigmoid
import torch_geometric.transforms as T
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from torch_geometric.nn import global_max_pool, radius, global_mean_pool, knn
from torch_geometric.nn.pool import radius
from sklearn import metrics
from pytorchtools import EarlyStopping
from divide_tools import divide_cdhit
from pos_weight_fold import pos
from PointTransformerConv import PointTransformerConv
import sys
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
import scipy.spatial
from torch_scatter import scatter_add

Center = T.Center()
Normalscale = T.NormalizeScale()
Delaunay = T.Delaunay()
Normal = T.GenerateMeshNormals()


def normalize_point_pos(pos):
    # pos_AB=torch.cat([pos_A, pos_B])
    pos = pos - pos.mean(dim=-2, keepdim=True)
    # pos_B=pos_B-pos_AB.mean(dim=-2, keepdim=True)
    scale = (1 / pos.abs().max()) * 0.999999
    pos = pos * scale
    # scale_B = (1 / pos_B.abs().max()) * 0.999999
    # pos_B = pos_B * scale_B
    return pos


def load_data(data_path):
    print('loading data')
    data_list = []

    with open(data_path, 'r') as f:
        n_g = int(f.readline().strip())
        num = 0
        for i in range(n_g):  # for each protein
            n = int(f.readline().strip())  # atom number
            point_tag = []
            point_fea_pssm = []
            point_pos = []
            point_aa = []
            aa_y = []
            mask = []
            mask_t = []

            for j in range(n):
                row = f.readline().strip().split()
                point_tag.append(int(row[2]))  # label , atom level
                mask.append(int(row[3]))  # surface
                mask_t.append(int(row[0]))  # residue types
                pos, fea_pssm = np.array([float(w) for w in row[4:7]]), np.array([float(w) for w in row[7:]])
                point_pos.append(pos)
                point_fea_pssm.append(fea_pssm)
                point_aa.append(int(row[1]))  # atom

            flag = -1
            for i in range(len(point_aa)):
                if (flag != point_aa[i]):
                    flag = point_aa[i]
                    aa_y.append(point_tag[i])  # label , residue level
            # print(aa_y)
            try:
                x = torch.tensor(point_fea_pssm, dtype=torch.float)  # 59
            except:
                print(x.shape)

            y = torch.tensor(point_tag)
            pos = torch.tensor(point_pos, dtype=torch.float)  # 3
            mask = torch.tensor(mask)
            mask_t = torch.tensor(mask_t)

            # pos=normalize_point_pos(pos)
            data = Data(x=x, y=y, pos=pos)
            # print(data.norm)

            for i in range(len(point_aa)):
                point_aa[i] = point_aa[i] + num
            num = num + len(aa_y)

            aa = torch.tensor(point_aa)
            # print(aa)
            number = len(aa_y)  # 氨基酸数量
            aa_y = torch.tensor(aa_y)

            data.aa = aa
            data.aa_y = aa_y
            data.num = number
            data.mask = mask
            data.mask_t = mask_t

            data = Center(data)
            # data = Normalscale(data)
            data = Delaunay(data)
            data = Normal(data)

            data = data.to(device)
            data_list.append(data)
    # print(data_list)
    return data_list


def MLP(channels):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), BN(channels[i]), ReLU(), Dropout(0.3))
        for i in range(1, len(channels))
    ])


def generate_normal(pos, batch):
    data_norm = []
    batch_list = torch.unique(batch)
    for b in batch_list:
        pos_temp = pos[batch == b]
        pos_temp = pos_temp - pos_temp.mean(dim=-2, keepdim=True)
        pos_temp = pos_temp.cpu().numpy()
        tri = scipy.spatial.Delaunay(pos_temp, qhull_options='QJ')
        face = torch.from_numpy(tri.simplices)

        data_face = face.t().contiguous().to(device, torch.long)
        pos_temp = torch.tensor(pos_temp).to(device)

        vec1 = pos_temp[data_face[1]] - pos_temp[data_face[0]]
        vec2 = pos_temp[data_face[2]] - pos_temp[data_face[0]]
        face_norm = F.normalize(vec1.cross(vec2), p=2, dim=-1)  # [F, 3]

        idx = torch.cat([data_face[0], data_face[1], data_face[2]], dim=0)
        face_norm = face_norm.repeat(3, 1)

        norm = scatter_add(face_norm, idx, dim=0, dim_size=pos_temp.size(0))
        norm = F.normalize(norm, p=2, dim=-1)  # [N, 3]

        data_norm.append(norm)

    return torch.cat(data_norm, dim=0)


class PointTransformerConv1(torch.nn.Module):
    def __init__(self, r, in_channels, out_channels):
        super(PointTransformerConv1, self).__init__()
        self.k = None
        self.r = r
        self.pos_nn = MLP([6, out_channels])

        self.attn_nn = MLP([out_channels, out_channels])

        self.conv = PointTransformerConv(in_channels, out_channels,
                                         pos_nn=self.pos_nn,
                                         attn_nn=self.attn_nn)

    def forward(self, x, pos, normal, batch):
        # row, col = knn(pos, pos, self.k, batch, batch)
        row, col = radius(pos, pos, self.r, batch, batch, max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, pos, edge_index, normal, self.r)
        return x


class Net(torch.nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        self.conv1 = PointTransformerConv1(5, in_channels=39 + 20, out_channels=128)
        self.conv2 = PointTransformerConv1(8.5, in_channels=39 + 20, out_channels=128)
        # self.conv3 = PointTransformerConv1(10, in_channels=39 + 20, out_channels=128)
        self.neck = Seq(Lin(128 + 128, 512), BN(512), ReLU(), Dropout(0.3))
        # self.conv4 = PointTransformerConv1(15, in_channels=512, out_channels=512)
        self.mlp = Seq(Lin(512, 256), BN(256), ReLU(), Dropout(0.3), Lin(256, out_channels))

    def forward(self, data):
        x0, pos, batch, normal, pool_batch, aa_num, mask, mask_t = data.x, data.pos, data.batch, data.norm, data.aa, data.num, data.mask, data.mask_t

        # atom to residue
        flag = torch.Tensor([-1]).to(device)
        num = -1
        for i in range(len(pool_batch)):
            if not torch.eq(pool_batch[i], flag):
                flag = pool_batch[i].clone()
                num = num + 1
                pool_batch[i] = torch.Tensor([num]).to(device)
            else:
                pool_batch[i] = torch.Tensor([num]).to(device)

        x1 = self.conv1(x0, pos, normal, batch)
        x2 = self.conv2(x0, pos, normal, batch)
        # x3 = self.conv3(x0, pos, normal, batch)
        # print(batch)
        out = self.neck(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, pool_batch)  # out-512
        # print(out)

        # residual batch
        # print(num)
        num_total = 0
        for i in range(len(aa_num)):
            num_total += aa_num[i]
        # print(num_total)
        aa_batch = torch.zeros(num_total).to(device)
        number = 0
        for m in range(len(aa_num)):
            # print(m)
            for n in range(aa_num[m].item()):
                aa_batch[n + number] = m
            number += aa_num[m].item()
        aa_batch = aa_batch.long()

        aa_pos = global_mean_pool(pos, pool_batch)

        aa_norm = generate_normal(aa_pos, aa_batch).to(device)

        # out = self.conv4(out, aa_pos, aa_norm, aa_batch)
        out = self.mlp(out)

        # mask = global_max_pool(mask, pool_batch)
        # mask = global_max_pool(mask.unsqueeze(dim=1), pool_batch).squeeze()
        # mask = mask == 1
        mask_t = global_max_pool(mask_t.unsqueeze(dim=1), pool_batch).squeeze()
        mask_t = mask_t == 1
        # data.label = mask & data.aa_y
        data.label = data.aa_y[mask_t]
        out = out[mask_t]
        # sample = data.label.shape[0]
        # one_sample = data.label.nonzero().shape[0]
        # if one_sample == 0:
        #     m = sample
        # else:
        #     m = (sample-one_sample)/one_sample

        # data.label = data.aa_y
        return out


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=.25, gamma=2):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#
#     def forward(self, inputs, targets, mask_t):
#
#         pos_weight = torch.FloatTensor([1.0]).to(device)
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight, reduction='mean')
#         # pt = torch.exp(-BCE_loss)
#         # F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
#         loss = self.alpha * BCE_loss
#         return loss.mean()

def BCE_loss(inputs, targets, m):
    # if fold == 0 or fold == '0':
    #     m = 4.748087431693989
    # else:
    #     if fold == 1 or fold == '1':
    #         m = 4.933297180043384
    #     else:
    #         if fold == 2 or fold == '2':
    #             m = 4.668763102725367
    #         else:
    #             if fold == 3 or fold == '3':
    #                 m = 4.704557935200439
    #             else:
    #                 if fold == 4 or fold == '4':
    #                     m = 4.851167843563281
    pos_weight = torch.tensor([m]).to(device)
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = loss(inputs, targets)
    return loss


def train_model(model, patience, n_epochs, checkpoint, m):
    train_losses = []
    valid_losses = []
    label_total = []
    score_total = []
    avg_train_losses = []
    avg_valid_losses = []
    early_stopping = EarlyStopping(patience=patience, path=checkpoint, verbose=True)

    for epoch in range(1, n_epochs + 1):

        model.train()
        for data in trainloader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            label = data.label.float()
            label = label.unsqueeze(1)
            # loss = focalloss(out, label, mask_t)
            loss = BCE_loss(out, label, m)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            for data in valloader:
                data = data.to(device)
                out = model(data)
                score = torch.sigmoid(out)
                score_total.extend(score.detach().cpu().numpy())
                label = data.label.float()
                label = label.unsqueeze(1)
                label_total.extend(label.detach().cpu().numpy())
                # loss = focalloss(out, label,mask_t)
                loss = BCE_loss(out, label, m)
                valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        auc = metrics.roc_auc_score(label_total, score_total)
        ap = metrics.average_precision_score(label_total, score_total)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f} ' +
                     f'AUC: {auc:.5f}' +
                     f'AP: {ap:.5f}')

        print(print_msg)

        train_losses = []
        valid_losses = []
        label_total = []
        score_total = []
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        scheduler.step(valid_loss)
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # model = torch.load(checkpoint)

    return avg_train_losses, avg_valid_losses


parser = argparse.ArgumentParser(
    description='supply the old and new directory to update the downloaded pdbs for a specific ion i.e. ZN, CA, CO3')
parser.add_argument('-input', dest='input', type=str, help='Specify the input feature file', required=True)
parser.add_argument('-tnum', dest='tnum', type=str,
                    help='The last 20 percent of the data is used as the test set, specifying the starting position of the test set',
                    required=True)
parser.add_argument('-cpath', dest='cpath', type=str, help='Storage location of model parameter files', required=True)
parser.add_argument('-spath', dest='spath', type=str, help='Storage location for predicted scores', required=True)
parser.add_argument('-mpath', dest='mpath', type=str, help='Storage location for metrics file', required=True)
# parser.add_argument('-ion', dest='ion', type=str, help='Specify the ion', required=True)

args = parser.parse_args()

input = args.input
tnum = args.tnum
cpath = args.cpath
spath = args.spath
mpath = args.mpath

os.makedirs(cpath, exist_ok=True)
os.makedirs(spath, exist_ok=True)
os.makedirs(mpath, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = load_data(input)
# train, val, test = dataset[: 576], dataset[576: 721], dataset[721: 843]
test = dataset[int(tnum):]
# dataset = torch.load('zy_checkpoint_ppf.pt')
print(len(dataset))
testloader = DataLoader(test, batch_size=4)
directory_name = mpath
file_name = "metrics"
file_path = os.path.join(directory_name, file_name)
ff = open(file_path, 'w')
for fold in [0, 1,2,3,4]:

    ff.write(str(fold) + '_model' + '\n')
    train, val = divide_cdhit(dataset, fold, int(tnum))
    trainloader = DataLoader(train, batch_size=4, shuffle=True, drop_last=True)
    valloader = DataLoader(val, batch_size=4)

    m, neg_num, pos_num = pos(trainloader)
    ff.write('neg num:' + str(neg_num) + r'\t' + 'pos num' + str(pos_num) + r'\t' + 'pos weight:' + str(m) + r'\t')

    model = Net()
    # focalloss = FocalLoss()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)  #
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, verbose=True)

    n_epochs = 15000
    patience = 20
    checkpoint = cpath + str(fold) + '.pt'
    train_loss, valid_loss = train_model(model, patience, n_epochs, checkpoint, m)

    pred_total = []
    aa_total = []
    out_total = []

    model = Net().to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    with torch.no_grad():
        for data in testloader:
            data = data.to(device)
            out = model(data)
            out = F.sigmoid(out)
            out_total.extend(out.cpu().tolist())
            pred = out.ge(0.844).float()
            pred_total.extend(pred.detach().cpu().numpy())
            aa_total.extend(data.label.detach().cpu().numpy())

    pred_total = torch.tensor(pred_total)
    out_total = torch.tensor(out_total)
    pred_total = pred_total.squeeze()
    out_total = out_total.squeeze()

    aa_total = torch.tensor(aa_total)

    correct = int(pred_total.eq(aa_total).sum().item())
    tn, fp, fn, tp = confusion_matrix(aa_total, pred_total).ravel()
    print('tn:' + str(tn) + ' tp:' + str(tp) + ' fn:' + str(fn) + ' fp:' + str(fp))
    ff.write('tn:' + str(tn) + ' tp:' + str(tp) + ' fn:' + str(fn) + ' fp:' + str(fp) + '\n')
    # r = recall_score(aa_total, pred_total)
    recall = tp / (tp + fn)
    print('recall:' + str(recall))
    ff.write('recall:' + str(recall) + '\n')
    sp = tn / (fp + tn)
    print('sp:' + str(sp))
    ff.write('sp:' + str(sp) + '\n')
    precision = tp / (tp + fp)
    print('precision:' + str(precision))
    ff.write('precision:' + str(precision) + '\n')
    mcc = float(tp * tn - fp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + sys.float_info.epsilon)
    print('mcc:' + str(mcc))
    ff.write('mcc:' + str(mcc) + '\n')
    auc = metrics.roc_auc_score(aa_total, out_total)
    print('AUC:' + str(auc))
    ff.write('AUC:' + str(auc) + '\n')
    ap = metrics.average_precision_score(aa_total, out_total)
    print('AP:' + str(ap))
    ff.write('AP:' + str(ap) + '\n')
    f1 = metrics.f1_score(aa_total, pred_total)
    print('f1:' + str(f1))
    ff.write('f1:' + str(f1) + '\n')
    ff.write('\n')

    out_total = out_total.tolist()
    aa_total = aa_total.tolist()
    with open(spath + str(fold) + '_result.txt', 'w') as f:
        for i in range(len(out_total)):
            f.write(str(aa_total[i]) + '\t' + str(out_total[i]) + '\n')

# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
