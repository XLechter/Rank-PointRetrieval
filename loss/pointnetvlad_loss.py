import numpy as np
import math
import torch
from torch.nn import functional as F

def best_pos_distance(query, pos_vecs):
    num_pos = pos_vecs.shape[1]
    query_copies = query.repeat(1, int(num_pos), 1)
    diff = ((pos_vecs - query_copies) ** 2).sum(2)
    min_pos, _ = diff.min(1)
    max_pos, _ = diff.max(1)
    return min_pos, max_pos


def triplet_loss(q_vec, pos_vecs, neg_vecs, margin, use_min=False, lazy=False, ignore_zero_loss=False):
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    # PointNetVLAD official code use min_pos, but i think max_pos should be used
    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    num_neg = neg_vecs.shape[1]
    batch = q_vec.shape[0]
    query_copies = q_vec.repeat(1, int(num_neg), 1)
    positive = positive.view(-1, 1)
    positive = positive.repeat(1, int(num_neg))
    
    print(positive.size(), query_copies.size(), neg_vecs.size())

    loss = margin + positive - ((neg_vecs - query_copies) ** 2).sum(2)
    loss = loss.clamp(min=0.0)
    if lazy:
        triplet_loss = loss.max(1)[0]
    else:
        triplet_loss = loss.sum(1)
    if ignore_zero_loss:
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
    else:
        triplet_loss = triplet_loss.mean()
    return triplet_loss
    
def triplet_ranking_loss(q_vec, pos_vecs, neg_vecs, margin, use_min=False, lazy=False, ignore_zero_loss=False):
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    # PointNetVLAD official code use min_pos, but i think max_pos should be used
    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    B = q_vec.shape[0]
    L = q_vec.shape[2]

    print('q_vec.size(), pos_vecs.size()', q_vec.size(), pos_vecs.size())

    dist_pos = ((q_vec - pos_vecs) ** 2).sum(2)
    dist_pos = dist_pos.view(B, 1)
    print('dist_pos.size()', dist_pos.size())

    output_anchors = q_vec.expand_as(neg_vecs).contiguous().view(-1, L)
    print(output_anchors.shape)
    output_negatives = neg_vecs.contiguous().view(-1, L)
    print(output_negatives.shape)
    dist_neg = ((output_anchors - output_negatives) ** 2).sum(1)
    print(dist_neg.shape)
    dist_neg = dist_neg.view(B, -1)

    print('dist_neg.size()', dist_neg.size())

    dist = - torch.cat((dist_pos, dist_neg), 1)
    print(dist.shape)
    dist = F.log_softmax(dist, 1)
    print(dist.shape)
    loss = (- dist[:, 0]).mean()
    print('loss', loss.shape, loss)

    return loss

def triplet_t_loss(q_vec, pos_vecs, neg_vecs, margin, use_min=False, lazy=False, ignore_zero_loss=False):
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    # PointNetVLAD official code use min_pos, but i think max_pos should be used
    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    B = q_vec.shape[0]
    L = q_vec.shape[2]

    print(q_vec.size(), pos_vecs.size())

    dist_pos = ((q_vec - pos_vecs) ** 2).sum(2)
    dist_pos = dist_pos.view(B, 1)
    print(dist_pos.size())

    output_anchors = q_vec.expand_as(neg_vecs).contiguous().view(-1, L)
    output_negatives = neg_vecs.contiguous().view(-1, L)
    dist_neg = ((output_anchors - output_negatives) ** 2).sum(1)
    dist_neg = dist_neg.view(B, -1)

    #print(dist_neg.size())

    dist = torch.cat((dist_pos, dist_neg), 1)
    #print('dist before', dist)
    dist = 1 / (1+ dist)
    #print('dist after', dist)
    dist_sum = torch.sum(dist, 1)
    #print(dist_sum)
    dist_sum = dist_sum.reshape(dist_sum.shape[0],-1).expand_as(dist)
    #print(dist_sum.size())
    #print(dist_sum)
    dist = torch.log(dist / dist_sum)
    #print('dist final', dist)
    loss = (- dist[:, 0]).mean()

    return loss

def triplet_loss_wrapper(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=False, ignore_zero_loss=False):
    return triplet_loss(q_vec, pos_vecs, neg_vecs, m1, use_min, lazy, ignore_zero_loss)

def triplet_ranking_loss_wrapper(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=False, ignore_zero_loss=False):
    return triplet_ranking_loss(q_vec, pos_vecs, neg_vecs, m1, use_min, lazy, ignore_zero_loss)

def triplet_t_loss_wrapper(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=False, ignore_zero_loss=False):
    return triplet_t_loss(q_vec, pos_vecs, neg_vecs, m1, use_min, lazy, ignore_zero_loss)

def quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=False, ignore_zero_loss=False):
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    # PointNetVLAD official code use min_pos, but i think max_pos should be used
    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    num_neg = neg_vecs.shape[1]
    batch = q_vec.shape[0]
    query_copies = q_vec.repeat(1, int(num_neg), 1)
    positive = positive.view(-1, 1)
    positive = positive.repeat(1, int(num_neg))

    loss = m1 + positive - ((neg_vecs - query_copies) ** 2).sum(2)
    loss = loss.clamp(min=0.0)
    if lazy:
        triplet_loss = loss.max(1)[0]
    else:
        triplet_loss = loss.sum(1)
    if ignore_zero_loss:
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
    else:
        triplet_loss = triplet_loss.mean()

    other_neg_copies = other_neg.repeat(1, int(num_neg), 1)
    second_loss = m2 + positive - ((neg_vecs - other_neg_copies) ** 2).sum(2)
    second_loss = second_loss.clamp(min=0.0)
    if lazy:
        second_loss = second_loss.max(1)[0]
    else:
        second_loss = second_loss.sum(1)

    if ignore_zero_loss:
        hard_second = torch.gt(second_loss, 1e-16).float()
        num_hard_second = torch.sum(hard_second)
        second_loss = second_loss.sum() / (num_hard_second + 1e-16)
    else:
        second_loss = second_loss.mean()

    total_loss = triplet_loss + second_loss
    return total_loss
