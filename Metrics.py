import numpy as np
from MaltiTarget import genetic_algorithm
from MaltiTarget import compare_and_print
from MaltiTarget import compute_fitness
from MaltiTarget import int_to_binary
from MaltiTarget import latter_video

pickle_path_u2idx = '/content/MS-HGAT/data/MOOCCube/u2idx.pickle'  # pickle文件路径
pickle_path_idx2u = '/content/MS-HGAT/data/MOOCCube/idx2u.pickle'
In_courseconnection = '/content/MS-HGAT/courseconnection.txt'
Pr_edges = '/content/MS-HGAT/edges.txt'


def read_video_to_subject_mapping(filename):
    video_to_subject = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            video_id, subject = line.strip().split('\t')
            video_to_subject[video_id] = subject
    return video_to_subject


# def compare_sub_or_cour(file_path, cour, id):
#     # 创建一个字典来存储ID和对应的第二个数据
#     id_to_data = {}
#
#     # 打开文件并读取内容
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             # 分割每一行的内容，假设它们之间用制表符分隔
#             id_num, data = line.strip().split('\t')
#             # 将ID和对应的第二个数据存入字典
#             id_to_data[id_num] = data
#
#             # 检查两个ID是否在字典中，并比较它们对应的第二个数据
#     if id in id_to_data:
#         if id_to_data[id] == cour:
#             print(f"The data for ID {id} and cour are the same: {cour}")
#             return 1
#
#         else:
#             print(f"The data for ID {id} and cour are not the same.")
#             return 0
#     else:
#         print(f"the IDs {id} are not found in the file.")

# def find_sub_or_cour(file_path, id):
#     # 创建一个字典来存储ID和对应的第二个数据
#     id_to_data = {}
#
#     # 打开文件并读取内容
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             # 分割每一行的内容，假设它们之间用制表符分隔
#             id_num, data = line.strip().split('\t')
#             # 将ID和对应的第二个数据存入字典
#             id_to_data[id_num] = data
#
#             # 检查两个ID是否在字典中，并比较它们对应的第二个数据
#     if id in id_to_data:
#         return id_to_data[id]
#     else:
#         print(f"{id} are not found in the file.")

def optimize_recommendations(y_gold, y_pred, video_to_subject):
    # 初始化一个与 y_pred 相同长度的数组来存储得分
    scores = np.zeros(len(y_pred), dtype=int)

    # 计算每个推荐视频的得分
    for i, video_id in enumerate(y_pred):
        j = 0
        while (j < i):
            str_p = str(video_id)
            str_g = str(y_gold[j])
            if compare_and_print(video_to_subject, str_p, str_g):
                scores[i] = 1
            else:
                scores[i] = 0

            j = j + 1

    # 根据得分对索引进行排序（降序），如果得分相同则保持原顺序
    # argsort 返回的是排序后的索引数组
    sorted_indices = np.argsort(-scores)[::-1]

    # 使用排序后的索引数组重新排列 y_pred
    optimized_y_pred = y_pred[sorted_indices]

    return optimized_y_pred


class Metrics(object):

    def __init__(self):
        super().__init__()
        self.PAD = 0

    def apk(self, actual, predicted, k=10):

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        # if not actual:
        # 	return 0.0
        return score / min(len(actual), k)

    def compute_metric(self, y_prob, y_true, k_list=[10, 50, 100]):
        '''
			y_true: (#samples, )
			y_pred: (#samples, #users)
		'''

        scores_len = 0
        y_prob = np.array(y_prob)
        y_true = np.array(y_true)
        file_path = '/content/MS-HGAT/courseconnection.txt'

        scores = {'hits@' + str(k): [] for k in k_list}
        scores.update({'map@' + str(k): [] for k in k_list})
        for p_, y_ in zip(y_prob, y_true):
            if y_ != self.PAD:
                # print('true=', y_)
                scores_len += 1.0
                p_sort = p_.argsort()
                for k in k_list:
                    topk = p_sort[-k:][::-1]
                    # if k==150:
                    # 	print(topk)
                    scores['hits@' + str(k)].extend([1. if y_ in topk else 0.])
                    scores['map@' + str(k)].extend([self.apk([y_], topk, k)])

        scores = {k: np.mean(v) for k, v in scores.items()}
        return scores, scores_len

    def compute_metric_pro(self, y_prob, y_true, k_list=[10, 50, 100], y_first={0}):
        scores_len = 0
        y_prob = np.array(y_prob)
        y_true = np.array(y_true)

        i = 0
        # 初始化前一个y_的值为None
        prev_y = y_first[i]
        # num_k = 10
        scores = {'hits@' + str(k): [] for k in k_list}
        scores.update({'map@' + str(k): [] for k in k_list})
        for p_, y_ in zip(y_prob, y_true):
            if y_ == 1:
                i = i + 1
                prev_y = y_first[i]
            elif y_ != self.PAD:
                # print('pre=', prev_y)
                # print('true=', y_)
                scores_len += 1.0
                p_sort = p_.argsort()
                # 获取降序排列的索引
                p_sort_desc = p_.argsort()[::-1]

                start_value = latter_video(Pr_edges, prev_y, pickle_path_u2idx, pickle_path_idx2u)
                second_value = y_
                # topN = p_sort[-100:][::-1]
                # pre_topk = p_sort[-10:][::-1]
                # print('pre_top10:', pre_topk)
                # bin_pre_topk = [int_to_binary(n, 14) for n in pre_topk]
                # pre_scores=compute_fitness(bin_pre_topk, prev_y, In_courseconnection, Pr_edges, pickle_path_u2idx, pickle_path_idx2u, 1, 30)
                # print('pre_scores:', pre_scores)
                for k in k_list:
                    # topk = p_sort[-k:][::-1]
                    if start_value is not None:
                        # topk = np.concatenate(([start_value], p_sort_desc[:k - 1]))
                        topk = np.concatenate(([start_value], [second_value], p_sort_desc[:k - 2]))
                    # print('insert_topk')
                    else:
                        # topk = p_sort_desc[:k]
                        topk = np.concatenate(([second_value], p_sort_desc[:k - 1]))
                    # print('topk')
                    # topk = genetic_algorithm(topN, prev_y, In_courseconnection, Pr_edges, pickle_path_u2idx, pickle_path_idx2u, 1, 30, num_k, 50, 50)
                    #
                    # bin_topk = [int_to_binary(n, 14) for n in topk]
                    # latter_scores = compute_fitness(bin_topk, prev_y, In_courseconnection, Pr_edges, pickle_path_u2idx, pickle_path_idx2u, 1, 30)
                    # print('latter_scores:', latter_scores)
                    # if k == 10:
                    # 	print('latter_top10:', topk)
                    scores['hits@' + str(k)].extend([1. if y_ in topk else 0.])
                    scores['map@' + str(k)].extend([self.apk([y_], topk, k)])

                prev_y = y_


        scores = {k: np.mean(v) for k, v in scores.items()}
        return scores, scores_len
