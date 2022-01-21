# ---------------------------------------------------------------------------
# Created By  : Viktor Schmuck - https://github.com/d4rkspir1t
# Created Date: 07/01/2021
# Last revised: 21/01/2022
# ---------------------------------------------------------------------------
__author__ = 'Viktor Schmuck'
__credits__ = ['Viktor Schmuck', 'Oya Celiktutan']
__license__ = 'MIT'
__version__ = '1.0.1'
__maintainer__ = 'Viktor Schmuck'
__email__ = 'viktor.schmuck@kcl.ac.uk'
__status__ = 'Development in Progress'

import argparse
import copy
import csv
import datetime
import itertools
import json
import logging
import math
import os
from pprint import pprint
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, accuracy_score

import dgl
import dgl.dataloading as dgl_dl
from dgl.nn import SAGEConv
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

import growl_utils.utils as ut

logger = logging.getLogger()
logger.setLevel('INFO')
parser = argparse.ArgumentParser(description='To perform tests with GROWL, decide which data to test on '
                                             'and whether this is an ablation test.')
parser.add_argument('-f', '--feats', default=20, help='Sets the number of features to create an embedding for.'
                    , type=int)
parser.add_argument('-e', '--epochs', default=100, help='Sets the numebr of epochs to train GROWL for.', type=int)
parser.add_argument('-s', '--string_end', default='20220117', help='The postfix of the results filename.', type=str)
parser.add_argument('-t', '--test', help='Which dataset to test on.', type=str, default='ps',
                    choices=['ps', 'cpp', 'all_salsa', 'rica', 'rica_yolo'])
parser.add_argument('-a', '--ablation', help='Is this a test involving ablation?', type=str,
                    default='no', choices=['no', 'ori', 'edg'])
parser.add_argument('-b', '--balance_samples', action='store_true', help='Enable sample balancing to discard runs '
                                                                         'if there is too big of an imbalance between '
                                                                         'positive and negative edge counts.')
parser.add_argument('-p', '--plot', action='store_true', help='Enable plotting the graphs produced during the tests.')
parser.add_argument('--yolo_acc', action='store_true', help='Print YOLOv4 detection accuracy compared '
                                                            'to RICA\'s ground truth. Only has an effect '
                                                            'if test=rica_yolo.')
parser.add_argument('--yolo_to_gt', action='store_true', help='When testing on YOLOv4 detections, '
                                                              'calculate F1-scores of detected groups compared '
                                                              'to the GT detections.')

args = parser.parse_args()
logger.info('Running test with: -t="%s", -a="%s"' % (args.test, args.ablation))

# Point to where the SALSA cpp and ps folders are. Keep the structure SALSA provides.
base_path_cpp = 'salsa/Annotation/salsa_cpp/'
base_path_ps = 'salsa/Annotation/salsa_ps/'

# Paths for RICA's ground truth and YOLOv4 detected data.
base_path_rica_gt = './gt_db_orientation_20210412_cd_1.csv'
base_path_rica = './yolo_db_orientation_20210810_cd_1.csv'

# Extra time needs to be higher than the first read part's frame count, so the dict merging
# doesn't overwrite values for matching keys.
extra_time = 10000
frame_data_ps = ut.read_frame_data(base_path_ps, 0)
if args.test == 'cpp' or args.test == 'all_salsa':
    frame_data_cpp = ut.read_frame_data(base_path_cpp, extra_time)

salsa_ps_keys = list(frame_data_ps.keys())
random.shuffle(salsa_ps_keys)
split_idx = math.ceil(len(salsa_ps_keys)*0.6)
train_set_keys = salsa_ps_keys[:split_idx]
test_set_keys = salsa_ps_keys[split_idx:]

train_dict = dict((k, frame_data_ps[k]) for k in train_set_keys)
if args.test == 'ps':
    test_dict = dict((k, frame_data_ps[k]) for k in test_set_keys)
elif args.test == 'cpp' or args.test == 'all_salsa':
    remaining_dict = dict((k, frame_data_ps[k]) for k in test_set_keys)
    test_dict = {**remaining_dict, **frame_data_cpp}
    test_dict = frame_data_cpp
if args.test in ['ps', 'cpp', 'all_salsa']:
    logger.info('Train len: %d, Test len: %d' % (len(train_dict.keys()), len(test_dict.keys())))

train_node_data = {}
train_edge_data = {}
max_side_dist = 0

for frame_id, frame_info in train_dict.items():
    node_data = []
    group_id_tracker = 0
    for group in frame_info:
        if len(group) == 1:
            group_id = -1
        else:
            group_id = group_id_tracker
            group_id_tracker += 1
        for person in group:
            data = ut.fetch_person_data(person, float(frame_id), base_path_ps)
            pos_x = float(data[0])
            if pos_x > max_side_dist:
                max_side_dist = pos_x
            pos_y = float(data[1])
            if args.ablation == 'no' or args.ablation == 'edg':
                body_pose = float(data[3])
                rel_head_pose = float(data[4])
                head_pose = body_pose + rel_head_pose
                node_data.append([person, group_id, pos_x, pos_y, round(head_pose, 4)])
            else:
                node_data.append([person, group_id, pos_x, pos_y])

    train_node_data[frame_id] = node_data
    edge_data = []
    for person_data in node_data:
        person = person_data[0]
        group = person_data[1]
        for idx in range(len(node_data)):
            if node_data[idx][0] != person and node_data[idx][1] != -1:
                if group == node_data[idx][1]:
                    # src dst distance effort
                    distance = math.dist([person_data[2], person_data[3]], [node_data[idx][2], node_data[idx][3]])

                    if args.ablation == 'no' or args.ablation == 'edg':
                        angle_diff = person_data[-1] - (node_data[idx][-1] - math.pi)
                        if angle_diff > math.pi * 2:
                            angle_diff = angle_diff % (math.pi * 2)
                            # print('\tcorrected: ', angle_diff)
                        elif angle_diff < math.pi * -2:
                            angle_diff = angle_diff % (math.pi * 2)
                            # print('\tcorrected: ', angle_diff)
                        if angle_diff < 0:
                            effort = math.pi * 2 + angle_diff
                        else:
                            effort = angle_diff
                        # src dst dist eff
                        edge_data.append([person, node_data[idx][0], distance, effort])
                    else:
                        edge_data.append([person, node_data[idx][0], distance])

    train_edge_data[frame_id] = edge_data

if 'rica' not in args.test:
    rica_test = False
else:
    rica_test = True
    if max_side_dist == 0:
        rel_dist = 1
    else:
        rel_dist = max_side_dist / 640
    if args.test == 'rica':
        frame_data_rica = ut.read_rica_frdata(base_path_rica_gt, rel_dist, 0)
    else:
        frame_data_rica = ut.read_rica_frdata(base_path_rica, rel_dist, 0)
    frame_data_rica_gt = ut.read_rica_frdata(base_path_rica_gt, rel_dist, 0)
    test_dict = frame_data_rica
    if args.test == 'rica_yolo' and args.yolo_acc:
        det_accs = []
        for frame, gtbb in frame_data_rica_gt.items():
            pred_bbc = 0
            if frame in frame_data_rica.keys():
                pred_bbc = len(frame_data_rica[frame])
            det_accs.append(pred_bbc/len(gtbb))
            # print(frame, len(gtbb), len(frame_data_rica[frame]))
        logger.info(str(np.average(det_accs)))

test_node_data = {}
test_edge_data = {}
for frame_id, frame_info in test_dict.items():
    if rica_test:
        node_data = []
        person_count = 1
        for person in frame_info:
            # print(frame, int(row[5])*rel_side_dist, float(row[7]), row[9], row[10])  # or 8 , frno, x, y, rot, label
            pos_x = person[1]
            pos_y = person[2]
            head_pose = person[3]
            group_id = int(person[4])
            if group_id == 0:
                group_id = -1

            if args.ablation == 'no' or args.ablation == 'edg':
                node_data.append([person_count, group_id, round(pos_x, 2), round(pos_y, 2), round(head_pose, 4)])
            elif args.ablation == 'ori':
                node_data.append([person_count, group_id, round(pos_x, 2), round(pos_y, 2)])

            person_count += 1
        test_node_data[frame_id] = node_data
        edge_data = []
        for person_data in node_data:
            person = person_data[0]
            group = person_data[1]
            for idx in range(len(node_data)):
                if node_data[idx][0] != person and node_data[idx][1] != -1:
                    if group == node_data[idx][1]:
                        # src dst distance effort
                        distance = math.dist([person_data[2], person_data[3]],
                                             [node_data[idx][2], node_data[idx][3]])

                        if args.ablation == 'no' or args.ablation == 'edg':
                            angle_diff = person_data[-1] - (node_data[idx][-1] - math.pi)
                            if angle_diff > math.pi * 2:
                                angle_diff = angle_diff % (math.pi * 2)
                                # print('\tcorrected: ', angle_diff)
                            elif angle_diff < math.pi * -2:
                                angle_diff = angle_diff % (math.pi * 2)
                                # print('\tcorrected: ', angle_diff)
                            if angle_diff < 0:
                                effort = math.pi * 2 + angle_diff
                            else:
                                effort = angle_diff
                            # src dst dist eff
                            edge_data.append([person, node_data[idx][0], distance, effort])
                        else:
                            edge_data.append([person, node_data[idx][0], distance])
        test_edge_data[frame_id] = edge_data
    else:
        node_data = []
        group_id_tracker = 0
        for group in frame_info:
            if len(group) == 1:
                group_id = -1
            else:
                group_id = group_id_tracker
                group_id_tracker += 1
            for person in group:
                if float(frame_id) < extra_time:
                    data = ut.fetch_person_data(person, float(frame_id), base_path_ps)
                else:
                    sub_id = float(frame_id)-extra_time
                    # print(sub_id)
                    data = ut.fetch_person_data(person, float(sub_id), base_path_cpp)
                # print(data)
                pos_x = float(data[0])
                if pos_x > max_side_dist:
                    max_side_dist = pos_x
                pos_y = float(data[1])
                if args.ablation == 'no' or args.ablation == 'edg':
                    body_pose = float(data[3])
                    rel_head_pose = float(data[4])
                    head_pose = body_pose + rel_head_pose
                    node_data.append([person, group_id, pos_x, pos_y, round(head_pose, 4)])
                else:
                    node_data.append([person, group_id, pos_x, pos_y])

        test_node_data[frame_id] = node_data
        edge_data = []
        for person_data in node_data:
            person = person_data[0]
            group = person_data[1]
            for idx in range(len(node_data)):
                if node_data[idx][0] != person and node_data[idx][1] != -1:
                    if group == node_data[idx][1]:
                        # src dst distance effort
                        distance = math.dist([person_data[2], person_data[3]], [node_data[idx][2], node_data[idx][3]])
                        if args.ablation == 'no' or args.ablation == 'edg':
                            angle_diff = person_data[-1] - (node_data[idx][-1] - math.pi)
                            if angle_diff > math.pi * 2:
                                angle_diff = angle_diff % (math.pi * 2)
                                # print('\tcorrected: ', angle_diff)
                            elif angle_diff < math.pi * -2:
                                angle_diff = angle_diff % (math.pi * 2)
                                # print('\tcorrected: ', angle_diff)
                            if angle_diff < 0:
                                effort = math.pi * 2 + angle_diff
                            else:
                                effort = angle_diff
                            # src dst dist eff
                            edge_data.append([person, node_data[idx][0], distance, effort])
                        else:
                            edge_data.append([person, node_data[idx][0], distance])

        test_edge_data[frame_id] = edge_data

train_graphs = []
test_graphs = []
test_graph_frame_ids = []
skipped = 0
iters_ps = 0
for frame_id, val in train_edge_data.items():
    # print('FR ID: ', frame_id)
    if float(frame_id) >= extra_time:
        custom_node_count = 21
    else:
        # continue
        custom_node_count = 18
    srcs = []
    dsts = []
    pos = {}
    for entry in val:
        srcs.append(entry[0]-1)
        dsts.append(entry[1]-1)
    feats = []
    for person in train_node_data[frame_id]:
        pos[person[0]-1] = [person[2], person[3]]
        feat = person[2:5]
        # print(person[0])
        feats.append(feat)

    feats = torch.from_numpy(np.array(feats))
    try:
        graph = dgl.graph((srcs, dsts), num_nodes=len(train_node_data[frame_id]))
    except dgl._ffi.base.DGLError:
        skipped += 1
        continue

    # print(graph.number_of_nodes(), len(feats), len(train_node_data[frame_id]))
    draw_graph = False
    graph.ndata['feat'] = feats.float()
    # print(graph.ndata['feat'][:10])
    # print('# nodes: %d, # edges: %d' % (graph.number_of_nodes(), graph.number_of_edges()))
    if draw_graph:
        nx_g = graph.to_networkx().to_undirected()
        # pos = nx.kamada_kawai_layout(nx_g)
        logger.debug(pos)
        # should assign pos on -1:1 scale based on coordinates
        try:
            nx.draw(nx_g, pos, with_labels=True, node_color="#A0CBE2")
        except nx.exception.NetworkXError:
            node_cs = []
            for i in range(graph.number_of_nodes()):
                if i not in pos.keys():
                    pos[i] = [0, 0]
                    node_cs.append('#541E1B')
                else:
                    node_cs.append("#A0CBE2")
            nx.draw(nx_g, pos, with_labels=True, node_color=node_cs)
        base_path = 'salsa/ps_graphs'
        iters_ps += 1
        name = '%d.png' % iters_ps
        graph_path = os.path.join(base_path, name.rjust(9, '0'))
        plt.savefig(graph_path)
        plt.close()
    train_graphs.append(graph)
logger.info('Skipped: %d' % skipped)

skipped = 0
iters_ps = 0
for frame_id, val in test_edge_data.items():
    # print('FR ID: ', frame_id)
    if float(frame_id) >= extra_time:
        custom_node_count = 21
    else:
        # continue
        custom_node_count = 18
    srcs = []
    dsts = []
    pos = {}
    for entry in val:
        srcs.append(entry[0]-1)
        dsts.append(entry[1]-1)
    feats = []
    for person in test_node_data[frame_id]:
        pos[person[0]-1] = [person[2], person[3]]
        feat = person[2:5]
        # print(person[0])
        feats.append(feat)

    feats = torch.from_numpy(np.array(feats))
    try:
        graph = dgl.graph((srcs, dsts), num_nodes=len(test_node_data[frame_id]))
    except dgl._ffi.base.DGLError:
        skipped += 1
        continue

    # print(graph.number_of_nodes(), len(feats), len(train_node_data[frame_id]))
    draw_graph = False
    graph.ndata['feat'] = feats.float()
    # print(graph.ndata['feat'][:10])
    # print('# nodes: %d, # edges: %d' % (graph.number_of_nodes(), graph.number_of_edges()))
    if draw_graph:
        nx_g = graph.to_networkx().to_undirected()
        # pos = nx.kamada_kawai_layout(nx_g)
        logger.debug(pos)
        # should assign pos on -1:1 scale based on coordinates
        try:
            nx.draw(nx_g, pos, with_labels=True, node_color="#A0CBE2")
        except nx.exception.NetworkXError:
            node_cs = []
            for i in range(graph.number_of_nodes()):
                if i not in pos.keys():
                    pos[i] = [0, 0]
                    node_cs.append('#541E1B')
                else:
                    node_cs.append("#A0CBE2")
            nx.draw(nx_g, pos, with_labels=True, node_color=node_cs)
        base_path = 'salsa/ps_graphs'
        iters_ps += 1
        name = '%d.png' % iters_ps
        graph_path = os.path.join(base_path, name.rjust(9, '0'))
        plt.savefig(graph_path)
        plt.close()
    test_graphs.append(graph)
    test_graph_frame_ids.append(frame_id)
logger.info('Skipped: %d' % skipped)

count = 0

train_set = train_graphs
test_set = test_graphs

h_feats = 20
epochs = 100

random_graph = random.sample(train_set, 1)[0]
# print(random_graph)
model = ut.GraphSAGE(random_graph.ndata['feat'].shape[1], h_feats)
pred = ut.MLPPredictor(h_feats)
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
pos_edge_count = 0
neg_edge_count = 0
for single_train_graph in train_set:
    u, v = single_train_graph.edges()
    if args.ablation == 'edg':
        train_pos_u, train_pos_v = u, v
        train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=single_train_graph.num_nodes())
    else:
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
        try:
            adj_neg = 1 - adj.todense() - np.eye(single_train_graph.num_nodes())
        except ValueError:
            continue
        neg_u, neg_v = np.where(adj_neg != 0)
        train_pos_u, train_pos_v = u, v
        train_neg_u, train_neg_v = neg_u, neg_v
        train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=single_train_graph.num_nodes())
        train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=single_train_graph.num_nodes())
        pos_edge_count += len(u)
        neg_edge_count += len(neg_u)

    all_logits = []

    for e in range(epochs):
        # forward
        # print('FEAT COUNT', len(batched_graph.ndata['feat']))
        h = model(single_train_graph, single_train_graph.ndata['feat'])
        pos_score = pred(train_pos_g, h)[0]['score']
        if args.ablation == 'edg':
            loss = ut.compute_loss_posonly(pos_score)
        else:
            neg_score = pred(train_neg_g, h)[0]['score']
            loss = ut.compute_loss(pos_score, neg_score)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
logger.info('+ edge c %d' % pos_edge_count)
logger.info('- edge c %d' % neg_edge_count)

# Until a better method is found for balancing positive and negative feature data, if the sets are imbalanced,
# discard the run.
if args.balance_samples:
    if pos_edge_count < 23400 and (neg_edge_count-pos_edge_count) > 65500:  # can be separate
        exit()

# TEST
auc_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
logger.info('Starting tests %d' % len(test_set))
test_c = 0
for single_val_idx, single_val_graph in enumerate(test_set):
    test_c += 1
    val_graph = copy.copy(single_val_graph)
    # print('Test graph', test_graph.ndata['feat'])
    test_eids = val_graph.edges(form='eid')
    val_graph.remove_edges(test_eids)
    u, v = single_val_graph.edges()
    u_t, v_t = val_graph.edges()
    # ABLATION 2
    # ===============================================================
    try:
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    except ValueError:
        continue
    try:
        adj_neg = 1 - adj.todense() - np.eye(single_val_graph.num_nodes())
        adj_t_neg = 1 - np.eye(val_graph.num_nodes())
    except ValueError:
        continue
    neg_u, neg_v = np.where(adj_neg != 0)
    neg_t_u, neg_t_v = np.where(adj_t_neg != 0)
    test_pos_u, test_pos_v = u, v
    test_neg_u, test_neg_v = neg_u, neg_v

    test_full_graph = dgl.graph((neg_t_u, neg_t_v), num_nodes=val_graph.num_nodes())

    with torch.no_grad():
        h = model(single_val_graph, single_val_graph.ndata['feat'])
        test_out, test_graph_out = pred(test_full_graph, h)
        test_labels = test_out['label']

        to_remove = []
        for i in range(len(test_labels)):
            if test_labels[i] == 0:
                to_remove.append(i)

        test_graph_out.remove_edges(to_remove)

        original_nodec = single_val_graph.num_nodes()
        original_u, original_v = single_val_graph.edges()
        pred_nodec = test_graph_out.num_nodes()
        pred_u, pred_v = test_graph_out.edges()
        original_clusters = ut.get_clusters(original_nodec, original_u, original_v)
        pred_clusters = ut.get_clusters(pred_nodec, pred_u, pred_v)

        swap_original_clusters = ut.swap_clusters(original_clusters)
        swap_pred_clusters = ut.swap_clusters(pred_clusters)

        tp = 0
        fp = 0
        fn = 0
        t = 2 / 3
        t_ = 1 - t
        used_pred_clusters = [-1]
        for key, cluster in swap_original_clusters.items():
            if key == -1:
                continue
            else:
                matched_clusters = {}
                fullsize = len(cluster)
                for pred_key, pred_cluster in swap_pred_clusters.items():
                    if pred_key == -1:
                        continue
                    match = 0
                    miss = 0
                    for node in cluster:
                        if node in pred_cluster:
                            match += 1
                        else:
                            miss += 1
                    if args.yolo_to_gt:
                    # ==================================================================================================
                    # MISSES DUE TO YOLO NOT RECOGNISING PEOPLE
                    # ==================================================================================================
                        for rgt_key, rgt_val in frame_data_rica_gt.items():
                            logger.debug('%s - %d' % (str(rgt_key), len(rgt_val)))
                        rgt_node_count = len(frame_data_rica_gt[test_graph_frame_ids[single_val_idx]])
                        ryo_node_count = len(frame_data_rica[test_graph_frame_ids[single_val_idx]])
                        miss += rgt_node_count - ryo_node_count
                    # ==================================================================================================
                    # ==================================================================================================

                    if match > 0:
                        matched_clusters[pred_key] = [match, miss]
                max_match = 0
                best_match = {}
                for match_key, match_val in matched_clusters.items():
                    if match_val[0] > max_match:
                        max_match = match_val[0]
                        best_match = {match_key: match_val}
                if len(list(best_match.keys())) == 0:
                    continue
                used_pred_clusters.append(list(best_match.keys())[0])
                best_match_val = list(best_match.values())[0]
                match = best_match_val[0]
                miss = best_match_val[1]
                if match / fullsize >= t and miss / fullsize <= t_:
                    tp += 1
                    verdict = 'tp'
                else:
                    fn += 1
                    verdict = 'fn'
                # print(key, match, miss, fullsize, verdict)
        for key in swap_pred_clusters.keys():
            if key not in used_pred_clusters:
                fp += 1
        # print('TP: %d, FN: %d, FP: %d' % (tp, fn, fp))
        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        if precision + recall == 0:
            f1 = 0.01
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        if args.plot:
            nx_g = test_graph_out.to_networkx().to_undirected()
            try:
                nx.draw(nx_g, pos, with_labels=True, node_color="#A0CBE2")
            except nx.exception.NetworkXError:
                pass
            plt.savefig(os.path.join('./quickplots', '%d.png' % test_c))
            plt.close()

if len(f1_scores) > 0:
    tracker_file_path = 'growl_param_analysis/%s_ablation_%s_test_%d_feats_%d_epochs_%d_balanced_model_f1output_%s.csv' % \
                        (args.ablation, args.test, args.feats, args.epochs, int(args.balance_samples), args.string_end)
    model_output_tracker = pd.DataFrame(
        list(zip([datetime.datetime.now()], [h_feats], [epochs], [len(f1_scores)],
                 [np.mean(precision_scores)], [np.mean(recall_scores)], [np.mean(f1_scores)],
                 [pos_edge_count], [neg_edge_count], [neg_edge_count-pos_edge_count])),
        columns=['time', 'feature_count', 'epoch_count', 'test_length', 'mean_precision', 'mean_recall',
                 'mean_f1', '+', '-', '- minus +'])
    if os.path.exists(tracker_file_path):
        model_output_tracker.to_csv(tracker_file_path, mode='a',
                                    index=False, header=False)
    else:
        model_output_tracker.to_csv(tracker_file_path, mode='w',
                                    index=False, header=True)