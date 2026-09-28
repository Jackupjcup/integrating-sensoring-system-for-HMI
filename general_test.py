import  pandas as pd
import numpy as np

def diff_idx_to_data_idx(pairs):
    if pairs[0] < pairs[1]:
        pairs[0] += 1
    return pairs

def func1(data):
    lift_count = 0
    drop_count = 0
    for i in range(data.shape[0]):
        data_bool = np.zeros(data.shape[1], dtype=bool)
        pos_neg_pairs = []
        pair_0 = None
        pair_1 = None
        lift_num = (lift_idx[0] == i).sum()
        drop_num = (drop_idx[0] == i).sum()
        if pos_max_change_rate[i] > change_rate_threshold and -neg_max_change_rate[i] > change_rate_threshold:

            cols_cat = np.hstack((lift_idx[1][lift_count: lift_count + lift_num],
                                  drop_idx[1][drop_count: drop_count + drop_num]))
            # print(cols_cat)
            left_edge_flag = False
            right_edge_flag = False

            for j in np.sort(cols_cat):
                index = np.where(cols_cat == j)[0][0]
                if index < lift_num:
                    if pair_0 is None and j != np.sort((cols_cat))[-1]:
                        pair_0 = cols_cat[index]
                    elif pair_0 is None and j == np.sort((cols_cat))[-1]:
                        pair_0 = cols_cat[index]
                        pairs = diff_idx_to_data_idx([pair_0, pair_1])
                        data_bool[pairs[0]] = True
                        data_bool[pairs[1]] = True
                        pair_0 = None
                    elif pair_0 is not None and left_edge_flag:
                        pair_1 = pair_0
                        # pos_neg_pairs.append(diff_idx_to_data_idx([pair_0, pair_1]))
                        pairs = diff_idx_to_data_idx([pair_0, pair_1])
                        data_bool[pairs[0]] = True
                        data_bool[pairs[1]] = True
                        pair_0 = cols_cat[index]
                        pair_1 = None
                if index >= lift_num:
                    if pair_0 is not None:
                        pair_1 = cols_cat[index]
                        # pos_neg_pairs.append(diff_idx_to_data_idx([pair_0, pair_1]))
                        pairs = diff_idx_to_data_idx([pair_0, pair_1])
                        data_bool[pairs[0]] = True
                        data_bool[pairs[1]] = True
                        pair_0 = None
                        pair_1 = None
                    elif pair_0 is None and cols_cat[index] == 0:
                        pair_0 = cols_cat[index]
                        left_edge_flag = True

        lift_count += lift_num
        drop_count += drop_num
        print(data_bool)

data = np.array([[1, 0, 0, 1, 0, 0, 2, 2, 0, 0], [1, 0, 0, 1, 0, 0, 2, 2, 0, 0], [1, 0, 0, 1, 0, 0, 2, 2, 0, 0], [1, 0, 0, 1, 0, 0, 2, 2, 0, 0]])

change_rates_np = np.diff(data, axis=1)
pos_max_change_rate = np.max(change_rates_np, axis=1)
neg_max_change_rate = np.min(change_rates_np, axis=1)

target_rows = np.where((pos_max_change_rate >= 1) & (neg_max_change_rate <= -1))[0]
lift_idx = np.where(change_rates_np >= 1)
drop_idx = np.where(change_rates_np <= -1)

print(data)
print(change_rates_np)
# print(change_rates_np.shape)
# print(pos_max_change_rate)
# print(neg_max_change_rate)
#
print(f"target_rows: {target_rows}")
# print(lift_idx)
# print(drop_idx)



change_rate_threshold = 1
print('##############################################')

















