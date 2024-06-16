import numpy as np
# from random import randint, uniform, choice, random, sample
import random
from operator import itemgetter
import pickle


def compare_and_print(file_path, id1, id2):
    # 创建一个字典来存储ID和对应的第二个数据
    id_to_data = {}

    # 打开文件并读取内容
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 分割每一行的内容，假设它们之间用制表符分隔
            id_num, data = line.strip().split('\t')
            # 将ID和对应的第二个数据存入字典
            id_to_data[id_num] = data

            # 检查两个ID是否在字典中，并比较它们对应的第二个数据
    if id1 in id_to_data and id2 in id_to_data:
        if id_to_data[id1] == id_to_data[id2]:
            # print(f"The data for ID {id1} and ID {id2} are the same: {id_to_data[id1]}")
            return 1

        else:
            # print(f"The data for ID {id1} and ID {id2} are not the same.")
            return 0
    # else:
    #     print(f"One or both of the IDs {id1} and {id2} are not found in the file.")


def find_videoid(pickle_path, target_index):
    # 使用pickle模块的load函数加载.pickle文件
    with open(pickle_path, 'rb') as f:
        idx2u_list = pickle.load(f, encoding='bytes')

    if target_index < len(idx2u_list):
        # 获取目标索引处的video_id（假设它是一个字符串）
        video_id = idx2u_list[target_index]
        if isinstance(video_id, bytes):
            video_id_str = video_id.decode('utf-8')
        else:
            video_id_str = video_id
            # 打印video_id
        return video_id_str
    else:
        print(f"Index {target_index} is out of range.")


def find_index(pickle_path, target_name):
    # 使用pickle模块的load函数加载.pickle文件
    with open(pickle_path, 'rb') as f:
        u2idx = pickle.load(f)  # 不需要指定encoding，因为pickle是二进制格式

    # 检查u2idx中的键是否是字节串
    if isinstance(list(u2idx.keys())[0], bytes):
        # 如果是，将目标video_id也转换为字节串
        target_video_id_bytes = target_name.encode('utf-8')
        # 在字典中查找
        index = u2idx.get(target_video_id_bytes)
    else:
        # 如果不是，直接使用字符串查找
        index = u2idx.get(target_name)
    return index


def latter_video(file_path, prev_y, pickle_path_u2idx, pickle_path_idx2u):
    prev_y = find_videoid(pickle_path_idx2u, prev_y)

    # 创建一个字典来存储ID和对应的第二个数据
    id_to_data = {}

    # 打开文件并读取内容
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 分割每一行的内容，这里假设它们之间用逗号分隔，而不是制表符
            id_num, data = line.strip().split(',')

            # 将ID和对应的第二个数据存入字典
            id_to_data[id_num] = data

    if prev_y in id_to_data:
        # 假设你想查找的video_id是存储在id_to_data[prev_y]中的
        target_video_id = id_to_data[prev_y]
        latter_index = find_index(pickle_path_u2idx, target_video_id)
        return latter_index
    else:
        # print(f"{prev_y} is not found in the file.")
        return None  # 返回None或其他合适的值来表示未找到


# 转换整数为二进制字符串的函数
def int_to_binary(n, num_bits):
    return format(n, '0' + str(num_bits) + 'b')


# 转换二进制字符串为整数的函数
def binary_to_int(binary_str):
    return int(binary_str, 2)


def binary_to_int_list(binary_strings):
    return [binary_to_int(binary_str) for binary_str in binary_strings]


def compute_In(file_path, prev_y, individual):
    In_score = 0  # 初始化分数
    for binary_value in individual:
        # 将二进制字符串转换为整数
        int_value = binary_to_int(binary_value)
        if compare_and_print(file_path, prev_y, int_value) == 1:
            In_score += 1
    return In_score


def compute_Pr(file_path, prev_y, individual, pickle_path_u2idx, pickle_path_idx2u):
    Pr_score = 0  # 初始化分数
    next_y = latter_video(file_path, prev_y, pickle_path_u2idx, pickle_path_idx2u)  # 假设pickle_path不是必要的，传入None
    if next_y is None:
        binary_next_y = 0
        # print('next_y is None')
    else:
        binary_next_y = int_to_binary(next_y, 14)

    top1_values = individual[:1]
    topk_values = individual[:2]  # 取前两个值
    # 将整数转换为二进制字符串
    binary_prev_y = int_to_binary(prev_y, 14)

    # 检查topk的第一个或第二个值是否等于prev_y或next_y
    if binary_prev_y in topk_values:
        Pr_score = Pr_score + 1  # 如果任一条件满足，则Pr_score为1
    if binary_next_y in top1_values:
        Pr_score = Pr_score + 1  # 如果任一条件满足，则Pr_score为1
    if binary_next_y in topk_values:
        Pr_score = Pr_score + 1  # 如果任一条件满足，则Pr_score为1
    return Pr_score


# 初始化种群，包括topN的前num_genes个值作为初始个体的基因片段（二进制表示）
def initialize_population(topN, population_size, num_genes, num_bits):
    # 确保topN有足够的元素来初始化第一个个体
    num_initial_genes = min(len(topN), num_genes)

    # 初始化第一个个体，使用topN的前num_initial_genes个值转换为二进制字符串
    first_individual = [int_to_binary(n, num_bits) for n in topN[:num_initial_genes]]
    population = [first_individual]  # 第一个个体基于topN的前num_initial_genes个值

    # 生成剩余的个体
    while len(population) < population_size:
        # 从topN中随机选择num_genes个不重复的值（可能需要重复选取如果topN较小）
        selected_genes = np.random.choice(topN, size=num_genes, replace=False)
        # selected_genes = random.sample(topN, num_genes)
        # 将这些值转换为二进制字符串，并创建新的个体
        new_individual = [int_to_binary(n, num_bits) for n in selected_genes]
        population.append(new_individual)

    return population


# 计算适应度
def compute_fitness(individual, prev_y, In_file_path, Pr_file_path, pickle_path_u2idx, pickle_path_idx2u, KI, KP):

    In_score = compute_In(In_file_path, prev_y, individual)
    Pr_score = compute_Pr(Pr_file_path, prev_y, individual, pickle_path_u2idx, pickle_path_idx2u)
    return KI * In_score + KP * Pr_score


def select(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [f / total_fitness for f in fitness_scores]
    return random.choice(population, p=probabilities)


def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return sorted(child1), sorted(child2)


def mutate_1(individual, mutation_rate, chromosome_length):
    if mutation_rate <= 0 or mutation_rate >= 1:
        raise ValueError("Mutation rate must be between 0 (exclusive) and 1 (inclusive).")
        # 根据变异率决定是否进行变异
    if random.random() < mutation_rate:
        # 随机选择两个不同的索引进行交换
        # 注意：range(chromosome_length) 生成的索引是从 0 到 chromosome_length-1
        # 使用 random.sample 确保选择的索引是不同的
        indices_to_swap = random.sample(range(chromosome_length), 2)
        # 交换这两个索引处的基因片段
        individual[indices_to_swap[0]], individual[indices_to_swap[1]] = individual[indices_to_swap[1]], individual[
            indices_to_swap[0]]
    return individual


def mutate_2(individual, topN, mutation_rate, num_bits=14):
    # 复制原始个体，以避免直接修改原始数据
    mutated_individual = individual[:]
    # 计算需要变异的基因片段数量
    num_to_mutate = int(len(individual) * mutation_rate)
    # 确保至少变异一个基因片段（如果mutation_rate非常低且个体很大）
    num_to_mutate = max(1, num_to_mutate)
    # 随机选择num_to_mutate个索引进行变异
    indices_to_mutate = random.sample(range(len(individual)), num_to_mutate)
    # 对选中的基因片段进行变异
    for index in indices_to_mutate:
        # 生成一个新的随机基因片段（确保位数相同）
        new_gene = int_to_binary(random.choice(topN), num_bits)
        # 替换原始基因片段
        mutated_individual[index] = new_gene

    return mutated_individual


def genetic_algorithm(topN, prev_y, In_file_path, Pr_file_path, pickle_path_u2idx, pickle_path_idx2u, KI, KP, num_genes, population_size, num_generations):
    population = initialize_population(topN, population_size, num_genes, 14)

    for generation in range(num_generations):
        fitness_scores = [compute_fitness(ind, prev_y, In_file_path, Pr_file_path, pickle_path_u2idx, pickle_path_idx2u, KI, KP) for ind in
                          population]

        # 根据适应度复制优秀个体
        sorted_population = sorted(zip(population, fitness_scores), key=itemgetter(1), reverse=True)
        top_individuals = [ind for ind, _ in sorted_population[:30]]  # 复制适应度最高的20个个体

        new_population = []
        # new_population.extend(top_individuals)
        for i in range(0, len(top_individuals)):
            # 对个体执行变异操作
            mutated_individual = mutate_1(top_individuals[i], mutation_rate=0.01, chromosome_length=len(top_individuals[i]))
            # 将变异后的个体添加到新种群中
            new_population.append(mutated_individual)

        # 随机生成剩余个体
        random_population = initialize_population(topN, 20, num_genes, 14)
        new_population.extend(random_population)

        # 执行变异
        for i in range(len(new_population)):
            new_population[i] = mutate_2(new_population[i], topN, 0.1, 14)  # 直接使用索引进行替换

        # 更新种群
        population = new_population[:population_size]

        # 找到当前种群的最佳解决方案
        best_solution = max(zip(population, fitness_scores), key=itemgetter(1))[0]
        # 这里可以添加逻辑来处理 best_solution，例如输出或保存

    decimal_best_solution = binary_to_int_list(best_solution)
    # 遗传算法结束后，返回最终的最佳解决方案
    return decimal_best_solution
