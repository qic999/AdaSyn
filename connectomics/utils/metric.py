import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment

# calculate cost: l2-norm
def cost_matrix(test, gt):
    matrix = np.zeros((test.shape[0], gt.shape[0]))
    for i in range(gt.shape[0]):
        diff = test - gt[i, :]
        square = np.power(diff, 2)
        sum_of_square = np.sum(square, axis=1)
        cost = np.sqrt(sum_of_square)
        matrix[:, i] = cost
    return matrix

# assign and calculate f-score
def assign_cal_f1(test, gt, radius, use_radius):
    cost_m = cost_matrix(test, gt)

    row_ind, col_ind = linear_sum_assignment(cost_m)
    final_row_ind = list(row_ind)
    final_col_ind = list(col_ind)
    if use_radius:
        r = radius
        for i in list(zip(row_ind, col_ind)):
            if cost_m[i[0], i[1]] > r:
                final_row_ind.remove(i[0])
                final_col_ind.remove(i[1])
    assignments = dict(zip(final_row_ind, final_col_ind))
    associated_cost = cost_m[final_row_ind, final_col_ind].sum()

    fp_fn_1 = np.abs(test.shape[0] - gt.shape[0])
    fp_fn_2 = (len(list(zip(row_ind, col_ind))) - len(list(zip(final_row_ind, final_col_ind)))) * 2
    tp = len(list(zip(final_row_ind, final_col_ind)))
    fscore = 2 * tp / (2 * tp + fp_fn_1 + fp_fn_2)
    return assignments, associated_cost, fscore, final_row_ind, final_col_ind

def pre_score(pre_test, pre_gt, offset, radius=11.0, use_radius=True):
    # offset gt
    pre_gt = pre_gt - offset

    # evaluation pre
    pre_assignments, pre_associated_cost, pre_fscore, pre_test_node, pre_gt_node = assign_cal_f1(pre_test, pre_gt, radius, use_radius)
    # print('pre_assignments:', pre_assignments)
    # print('pre_cost:', pre_associated_cost)
    return pre_fscore

def synapse_score(pre_test, post_test, pre_gt, post_gt, offset, radius=2500.0, use_radius=True):
    # offset gt
    pre_gt = pre_gt - offset
    post_gt[:, 1:] = post_gt[:, 1:] - offset

    # evaluation pre
    pre_assignments, pre_associated_cost, pre_fscore, pre_test_node, pre_gt_node = assign_cal_f1(pre_test, pre_gt, 11, use_radius)
    # print('pre_assignments:', pre_assignments)
    # print('pre_cost:', pre_associated_cost)

    # evaluation post
    post_fscore_all = []
    for i in list(zip(pre_test_node, pre_gt_node)):
        post_test_each = post_test[post_test[:, 0] == i[0]]
        post_gt_each = post_gt[post_gt[:, 0] == i[1]]
        if len(post_test_each[:, 1:]) == 0 and len(post_gt_each[:, 1:]) == 0:
            continue
        post_assignments_each, post_associated_cost_each, post_fscore_each, _, _ = assign_cal_f1(post_test_each[:, 1:], post_gt_each[:, 1:], 6.5, True)
        post_fscore_all.append(post_fscore_each)
    post_fscore = np.mean(post_fscore_all)

    # evaluation all
    final_fscore = 0.5 * pre_fscore + 0.5 * post_fscore
    # print('final_fscore:', final_fscore)

    return final_fscore, pre_fscore, post_fscore