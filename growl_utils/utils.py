import os
import csv
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv


person_log = 'geometryGT/'
fformation_log = 'fformationGT.csv'


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        out_score = self.W2(F.relu(self.W1(h))).squeeze(1)
        out_label = torch.round(torch.sigmoid(out_score))
        # print(out_score, out_label)
        out_dict = {'score': out_score, 'label': out_label}
        return out_dict

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            # return g.edata['score']
            # print('executes', g.edata)
            out_dict = dict(g.edata)
            return out_dict, g


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_loss_posonly(pos_score):
    scores = torch.cat([pos_score])
    labels = torch.cat([torch.ones(pos_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)


def fetch_person_data(person_id, frame_ts, base_path):
    f_name = str(person_id).rjust(2, '0') + '.csv'
    f_path = os.path.join(base_path, person_log, f_name)
    # print(f_path)
    with open(f_path, 'r') as csvf:
        csvrdr = csv.reader(csvf, delimiter=',')
        for row in csvrdr:
            frame = float(row[0])
            data = row[1:]
            # print(person_id, frame_ts, base_path, data)
            if frame == round(frame_ts, 5):
                return data


def read_frame_data(base_p, extra_t=0):
    ff_path = os.path.join(base_p, fformation_log)
    frame_data = {}
    with open(ff_path, 'r') as csvf:
        csvrdr = csv.reader(csvf, delimiter=',')
        for row in csvrdr:
            frame = str(float(row[0]) + extra_t)
            if frame not in frame_data.keys():
                frame_data[frame] = []
            group = []
            for idx in row[1:]:
                try:
                    group.append(int(idx))
                except ValueError:
                    print('BAD INPUT: ', idx)
            frame_data[frame].append(group)
    return frame_data


def read_rica_frdata(bpath, rel_side_dist=1, extra_t=0,):
    frame_data = {}
    with open(bpath, 'r') as csvf:
        csvrdr = csv.reader(csvf, delimiter=';')
        row_c = 0
        for row in csvrdr:
            row_c += 1
            if row_c == 1:
                continue
            # print(row)
            frame = int(row[0]) + extra_t
            # print(frame, int(row[5])*rel_side_dist, float(row[7]), row[9], row[10])  # or 8 , frno, x, y, rot, label
            if frame not in frame_data.keys():
                frame_data[frame] = []
            frame_data[frame].append([frame, int(row[5])*rel_side_dist, float(row[7]), float(row[9]), int(row[10])])
            # if row_c == 6:
            #     break
    return frame_data


def get_clusters(nodec, srcn, dstn):
    clusters = {}
    cluster_idx = 0
    for node in range(nodec):
        if node not in clusters.keys():
            clusters[node] = -1
            if node in srcn:
                clusters[node] = cluster_idx
                cluster_idx += 1
                for idx, u in enumerate(list(srcn)):
                    if u == node:
                        clusters[int(dstn[idx])] = clusters[node]
    return clusters


def swap_clusters(clusters):
    swapped_clusters = {}
    for key, val in clusters.items():
        if val not in swapped_clusters.keys():
            swapped_clusters[val] = []
        swapped_clusters[val].append(key)
    return swapped_clusters
