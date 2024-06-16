import pickle

# 加载idx2u映射
with open('F:\\HGNN\\MSHGAT\\MS-HGAT-main\\MS-HGAT-main\\data\\twitter\\idx2u.pickle', 'rb') as f:
    idx2u = pickle.load(f)

# 加载u2idx映射
with open('F:\\HGNN\\MSHGAT\\MS-HGAT-main\\MS-HGAT-main\\data\\twitter\\u2idx.pickle', 'rb') as f:
    u2idx = pickle.load(f)

# 使用映射
user_id = '30863036'
index = u2idx[user_id]  # 获取用户ID对应的索引
user_id_from_index = idx2u[1]  # 获取索引对应的用户ID

# 打印结果
print(f"原始用户ID: {user_id}")
print(f"用户ID对应的索引: {index}")
print(f"索引对应的用户ID: {user_id_from_index}")