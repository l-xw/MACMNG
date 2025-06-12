import pandas as pd
import numpy as np


def addition(s1, s2):
    a = s1[0] + s2[0]
    b = s1[1] + s2[1]
    c = s1[2] + s2[2]
    return [a, b, c]


def subtract(s1, s2):
    a = s1[0] - s2[0]
    b = s1[1] - s2[1]
    c = s1[2] - s2[2]
    return [a, b, c]


def old_max(s1, s2):
    a = max(s1[0], s2[0])
    b = max(s1[1], s2[1])
    c = max(s1[2], s2[2])
    return [a, b, c]


def value(s):
    return (s[0] + 2 * s[1] + s[2]) / 4


def rank(s1, s2):
    if value(s1) > value(s2):
        return s1
    elif value(s1) < value(s2):
        return s2
    elif s1[1] > s2[1]:
        return s1
    elif s1[1] < s2[1]:
        return s2
    elif (s1[2] - s1[0]) > (s2[2] - s2[0]):
        return s1
    else:
        return s2


def equality(s1, s2):
    if s1[0] != s2[0]: return False
    if s1[1] != s2[1]: return False
    if s1[2] != s2[2]: return False
    return True


def get_max(s):
    res = s[0]
    for i in range(1, len(s)):
        res = rank(res, s[i])
    return res


def get_min(s):
    res = s[0]
    for i in range(1, len(s)):
        max_ = rank(res, s[i])
        if equality(max_, res) and sum(s[i]) >= 0 or sum(res) < 0:
            res = s[i]
    return res


def get_avg(s):
    res = [0, 0, 0]
    count = 0
    for i in s:
        if sum(i) > 0:
            res = addition(res, i)
            count += 1
    if count != 0:
        for index in range(len(res)):
            res[index] /= count
    return res


def get_sum(S):
    res = [0, 0, 0]
    for s in S:
        res = addition(res, s)
    return res


def multiplication(s, x):
    return [s[0] * x, s[1] * x, s[2] * x]


def insertable(s1, s2):
    return equality(rank(s1, s2), s1)


def read_FMK(path):
    with open(path) as file:
        lines = file.read().split('\n')
    part = lines[0].split()
    job_nb, op_nb, machine_nb, factor_nb = int(part[0]), int(part[1]), int(part[2]), int(part[3])

    total_fuzzy_pt = []

    for k in range(1, job_nb + 1):
        job_fuzzy_pt = []
        line = lines[k]
        part = line.split()
        index = 1
        for i in range(int(part[0])):
            pt = [[-1, -1, -1] for _ in range(machine_nb)]
            op_nb_new = int(part[index])
            index += 1
            for j in range(op_nb_new):
                pt[int(part[index]) - 1] = [int(part[index + 1]), int(part[index + 2]), int(part[index + 3])]
                index += 4
            job_fuzzy_pt.append(pt)
        while len(job_fuzzy_pt) < op_nb:
            job_fuzzy_pt.append([[0, 0, 0] for _ in range(machine_nb)])
        total_fuzzy_pt.append(job_fuzzy_pt)
    return factor_nb, job_nb, op_nb, machine_nb, total_fuzzy_pt


def read_PMK(path):
    with open(path) as file:
        lines = file.read().split('\n')
    part = lines[0].split()
    job_nb, op_nb, machine_nb, factor_nb = int(part[0]), int(part[1]), int(part[2]), int(part[3])

    total_fuzzy_pt = []

    for k in range(1, job_nb + 1):
        job_fuzzy_pt = []
        line = lines[k]
        part = line.split()
        index = 1
        for i in range(int(part[0])):
            pt = [[-1, -1, -1] for _ in range(machine_nb)]
            op_nb_new = int(part[index])
            index += 1
            for j in range(op_nb_new):
                pt[int(part[index]) - 1] = [int(part[index + 1]), int(part[index + 1]), int(part[index + 1])]
                index += 2
            job_fuzzy_pt.append(pt)
        while len(job_fuzzy_pt) < op_nb:
            job_fuzzy_pt.append([[0, 0, 0] for _ in range(machine_nb)])
        total_fuzzy_pt.append(job_fuzzy_pt)
    return factor_nb, job_nb, op_nb, machine_nb, total_fuzzy_pt


def read_data(excel_file):
    df = pd.read_excel(excel_file, sheet_name='Sheet1')
    all_data = []
    df = df.fillna(0)
    for index, row in df.iterrows():
        # 将每行的数据转换为整数并存储在一个列表中
        int_row = [int(val) for val in row]
        # 将每行的整数列表添加到总列表中
        all_data.append(int_row)
    factor_nb, job_nb, machine_nb, op_nb = all_data[0][3], all_data[0][0], all_data[0][2], all_data[0][1]
    op_list = []
    for i in range(job_nb):
        op_list.append(all_data[1][i])
    total_fuzzy_pt = []
    index = 2
    for i in range(job_nb):
        fz = []
        for j in range(op_list[i]):
            fz_op = []
            k = 0
            for m in range(machine_nb):
                temp = [int(all_data[index][k]), int(all_data[index][k + 1]), int(all_data[index][k + 2])]
                fz_op.append(temp)
                k += 3
            fz.append(fz_op)
            index += 1
        while len(fz) < op_nb:
            fz.append([[0, 0, 0] for _ in range(machine_nb)])
        total_fuzzy_pt.append(fz)
    return factor_nb, job_nb, op_nb, machine_nb, total_fuzzy_pt


def read_reman(path):
    with open(path, 'r') as fin:
        A = np.fromfile(fin, dtype=int, sep=' ')

    N = A[0]  # 总工件数
    H = np.zeros(N, dtype=int)  # 各工件工序数
    NM = {}  # 各工序可选机器数
    M = {}  # 各工序可选机器号
    time = {}  # 工序时间
    p = 3  # 当前位置
    TM = A[1]
    f = A[2]
    for i in range(N):
        H[i] = A[p]
        for j in range(H[i]):
            p += 1
            NM[i, j] = A[p]
            for k in range(NM[i, j]):
                p += 1
                M[i, j, k] = A[p]
                time[i, j, M[i, j, k]] = A[p + 1:p + 4].tolist()
                p += 3
        p += 1

    total_fuzzy_pt = [[[[-1, -1, -1] for _ in range(TM)] for _ in range(H[j])] for j in range(N)]
    for k, v in time.items():
        total_fuzzy_pt[k[0]][k[1]][k[2] - 1] = v
    for i in range(len(total_fuzzy_pt)):
        while len(total_fuzzy_pt[i]) < max(H):
            total_fuzzy_pt[i].append([[0, 0, 0] for _ in range(TM)])
    return f, N, max(H), TM, total_fuzzy_pt


def read_example(path, dtype):
    if dtype == 'dat': return read_data(path)
    if dtype == 'FMK': return read_FMK(path)
    if dtype == 'PMK': return read_PMK(path)
    return read_reman(path)


def check_valid(pop, op_nb, job_nb):
    pop = pop.tolist()
    for p in pop:
        for i in range(job_nb):
            if p.count(i) != op_nb:
                print('error')
    print('right')


def find_pareto_front(res):
    pareto_indices = []

    for i in range(len(res)):
        dominated = False
        for j in range(len(res)):
            if i != j and is_dominated(res[i], res[j]):
                dominated = True
                break
        if not dominated:
            pareto_indices.append(i)

    return pareto_indices


def is_dominated(a, b):
    return all(b_i <= a_i for a_i, b_i in zip(a, b)) and any(b_i < a_i for a_i, b_i in zip(a, b))
