"""
NMRC4.0 修改dist的计算策略 使用相同邻域点进行计算 而非使用所有的领域点
参照LPM对NMRC进行加速
"""
import numpy as np
import cv2 as cv
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import time
import F_Score
from sklearn.neighbors import KDTree


def SIFT(img1, img2):
    """
    使用SIFT计算粗匹配点
    :param img1: queryImage
    :param img2: trainImage
    :return: src_pts, dst_pts: 粗匹配点对
    """
    sift = cv.SIFT_create(400)
    kp1 = sift.detect(img1)
    kp2 = sift.detect(img2)
    kp1, des1 = sift.compute(img1, kp1)
    kp2, des2 = sift.compute(img2, kp2)
    # 创建设置FLANN匹配
    ratio = 0.85
    matcher = cv.BFMatcher()
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m1, m2 in raw_matches:
        #  如果最接近和次接近的比值大于一个既定的值，那么我们保留这个最接近的值，认为它和其匹配的点为good_match
        if m1.distance < ratio * m2.distance:
            good_matches.append([m1])

    # 获取关键点的坐标
    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])

    return src_pts, dst_pts


def solver_lsp(A, b):
    """
    使用cvxopt求解带约束的最小二乘问题
    :param A: Ax-b
    :param b: Ax-b
    :param K: A=n x K
    :return: x: 权重最优解
    """
    K = A.shape[1]
    P = matrix(2 * A.T * A + np.eye(K, K) * 0.1)  # 加单位阵，避免元素过小导致不满秩
    q = matrix(-2 * b * A, (K, 1))
    a = matrix(np.ones(K), (1, K))
    b = matrix([1.0])
    sol = solvers.qp(P=P, q=q, A=a, b=b)
    return sol['x']


def construct_neighbor(m1, K, index_N):
    """
    使用m1[index_N, :]的元素构建m1[I0, :]的领域
    :param m1:
    :param K:
    :param index_N:
    :return: index_N[neighborm1]: 表示邻域元素在m1中的索引
    """
    treem1 = KDTree(m1[index_N, :])
    _, neighborm1 = treem1.query(m1, K)
    return index_N[neighborm1]


def interation_neighbor(m1, m2, M, K, k, nm):
    """
    通过多次迭代选择出可用于构造邻域的鲁棒点集
    :param m1: 原始图像特征点
    :param m2: 新图像特征点
    :param M: 迭代次数
    :param K: K近邻
    :param k: 计算公共特征点的比率 分母
    :param nm: 判断邻域是否相同的阈值
    :return: index: 迭代后的邻域点集对在初始匹配点对中的索引 m1[index]
    """
    N = m1.shape[0]
    # 邻域点集索引初始化
    index_N = np.arange(0, N, 1, int)
    index_Um = np.array(np.zeros(N) - 1, int)  # 迭代后用于构造邻域的点的索引
    for m in range(M):
        index_X = construct_neighbor(m1, K, index_N)  # x的邻域
        index_Y = construct_neighbor(m2, K, index_N)  # y的邻域
        ti = -1  # 计数器，表示筛选出来的点对个数

        for i in range(N):
            ni = len(set(index_X[i, :]) & set(index_Y[i, :]))
            ratio = ni / k
            if ratio > nm[m]:
                ti = ti + 1
                index_Um[ti] = i  # 存放筛选出来的匹配点对的索引
        if ti > K:
            index_N = index_Um[0:ti]
        else:
            print("迭代滤波第", m+1, "波中途退出")
            break
    return index_N


def same_neighbor(index_X, index_Y, k, yita):
    """

    :param index_X:
    :param index_Y:
    :param k:
    :param I0:
    :param yita:
    :return: index_I0: 返回邻域相同点大于yita的索引
    """
    N = index_X.shape[0]
    index_std = np.arange(0, N, 1, int)  # 标准点集索引
    index_Um = np.array(np.zeros(N) - 1, int)  # 迭代后用于构造邻域的点的索引
    ti = -1
    for i in range(N):
        ni = len(set(index_X[i, :]) & set(index_Y[i, :]))
        ratio = ni/k
        if ratio >= yita:
            ti = ti + 1
            index_Um[ti] = index_std[i]
            index_I0 = index_Um[0:ti]

    return index_I0


def calculate_dist(m1, m2, K, index, option, yita):
    """
    计算m1与邻域m1[index]的距离
    :param m1:
    :param m2:
    :param K:
    :param index: 用m1[index]构造点集m1的邻域
    :param I0: m1[I0]
    :return: index_X: m1[index_X[i, :], :] are the neighbors of m1[index_I0[i], :]
    """
    index_I0 = None
    index_X = construct_neighbor(m1, K, index)  # x的邻域
    index_Y = construct_neighbor(m2, K, index)  # y的邻域
    if option == 1:
        index_I0 = same_neighbor(index_X, index_Y, K, yita)  # K=k
        N = index_I0.shape[0]
        index_X = index_X[index_I0, :]
        index_Y = index_Y[index_I0, :]
        Um_N = m1[index_I0, :]
        Um_C = m2[index_I0, :]
    else:
        N = m1.shape[0]
        Um_N = m1
        Um_C = m2

    # 计算权重W
    dist = np.array(np.zeros((N, 2)), float)  # 距离度量初始化
    x_x = np.matrix([m1[index_X[i, :], 0] for i in range(N)])  # 原图像m1的邻域点的横坐标
    x_y = np.matrix([m1[index_X[i, :], 1] for i in range(N)])
    y_x = np.matrix([m2[index_Y[i, :], 0] for i in range(N)])
    y_y = np.matrix([m2[index_Y[i, :], 0] for i in range(N)])
    for i in range(N):
        list_x = index_X[i, :].tolist()
        list_y = index_Y[i, :].tolist()
        same_index = set(list_x) & set(list_y)
        same_num = len(same_index)
        same_index_x = np.zeros(same_num, int)
        same_index_y = np.zeros(same_num, int)
        ti = -1  # 循环计数
        for j in same_index:
            ti = ti + 1
            same_index_x[ti] = list_x.index(j)
            same_index_y[ti] = list_y.index(j)

        A_xx = x_x[i, same_index_x].astype(np.double)  # 原图像x坐标，需转成double
        b_xx = np.mat(Um_N[i, 0]).astype(np.double)  #
        A_xy = x_y[i, same_index_x].astype(np.double)
        b_xy = np.mat(Um_N[i, 1]).astype(np.double)
        A_yx = y_x[i, same_index_y].astype(np.double)
        b_yx = np.mat(Um_C[i, 0]).astype(np.double)
        A_yy = y_y[i, same_index_y].astype(np.double)
        b_yy = np.mat(Um_C[i, 1]).astype(np.double)

        W_xx = solver_lsp(A_xx, b_xx)
        W_xy = solver_lsp(A_xy, b_xy)
        W_yx = solver_lsp(A_yx, b_yx)
        W_yy = solver_lsp(A_yy, b_yy)
        dist[i, 0] = np.array((W_xx - W_yx).T * (W_xx - W_yx))
        dist[i, 1] = np.array((W_xy - W_yy).T * (W_xy - W_yy))

    return dist, index_X, index_Y, index_I0


def NMRC(m1, m2, lamda, K, M, nm, k, yita):
    """
    lamda = [1.5, 1.5]
    K = 10
    M = 3
    nm = [0.2, 0.5, 0.5]  # 判断邻域是否相同的阈值
    k = 10  # 计算公共特征点的比率 分母
    :param m1: 原图像的匹配点对
    :param m2: 新图像的匹配点对
    :return: I: 去除错误匹配后的内点集
    """
    time_start = time.time()
    # 代入迭代滤波器计算用于构建邻域的点
    index = interation_neighbor(m1, m2, M, K, k, nm)
    time_end = time.time()
    time_c = time_end - time_start
    print('interation_neighbor time cost', time_c, 's')

    N = m1.shape[0]
    index_std = np.arange(0, N, 1, int)
    # 用UM计算距离
    time_start = time.time()

    """index_X = construct_neighbor(m1, K, index, index_std)
    index_Y = construct_neighbor(m2, K, index, index_std)"""
    dist, index_X, index_Y, index_I0 = calculate_dist(m1, m2, K, index, 1, yita)

    N = index_X.shape[0]
    # 计算初始内点集I0
    I0 = np.array(np.zeros(N), int)
    ti = -1
    for i in range(N):
        if dist[i, 0] < lamda[0] and dist[i, 1] < lamda[1]:
            ti = ti + 1
            I0[ti] = index_I0[i]
    I0 = I0[0:ti]
    N = I0.shape[0]
    # 用I0计算距离
    if K <= N:
        dist, index_X, index_Y, index_I0 = calculate_dist(m1, m2, K, I0, 1, yita)  # index_X 是用I0构建邻域的索引
    else:
        print("I0构成的邻域过小,K近邻大小为", N - 1)
        dist, index_X, index_Y, index_I0 = calculate_dist(m1, m2, N - 1, I0, 1, yita)  # index_X 是用I0构建邻域的索引

    time_end = time.time()
    time_c = time_end - time_start
    print('构造I0 time cost', time_c, 's')

    N = I0.shape[0]
    I = np.array(np.zeros(N), int)
    temp = np.array(np.zeros(N), int)
    ti = -1
    for i in range(N):
        if dist[i, 0] < lamda[0] and dist[i, 1] < lamda[1]:
            ti = ti + 1
            I[ti] = I0[i]
            temp[ti] = i
    I = I[0:ti]
    temp = temp[0:ti]
    index_X = index_X[temp, :]  # index_X是邻域在I0中的索引 即m1[I0[index_X]]
    index_Y = index_Y[temp, :]
    return I, index_X, index_Y


if __name__ == '__main__':
    img1 = cv.imread('test_images/VGG_graf/graf1.jpg')
    img2 = cv.imread('test_images/VGG_graf/graf2.jpg')
    m1, m2 = SIFT(img1, img2)

    time_start = time.time()
    I, index_X, index_Y = NMRC(m1, m2, [10.5, 10.5], 10, 3, [0.3, 0.5, 0.8], 10, 0.8)
    time_end = time.time()
    time_c = time_end - time_start
    print("NMRC总耗时", time_c, "s")

    match1 = m1[I, :]
    match2 = m2[I, :]
    H = np.mat(([[8.7976964e-01, 3.1245438e-01, -3.9430589e+01],
                 [-1.8389418e-01, 9.3847198e-01, 1.5315784e+02],
                 [1.9641425e-04, -1.6015275e-05, 1.0000000e+00]]))
    F_s = F_Score.calculation(m1, m2, match1, match2, H, img1)

    m2[:, 0] = m2[:, 0] + img1.shape[1]  # 对新图像特征点做一个平移
    match1 = m1[I, :]
    match2 = m2[I, :]

    imgs = np.hstack([img1, img2])
    # match2[:, 0] = match2[:, 0] + img1.shape[1]

    fig = plt.figure('imgs')
    ax = fig.add_subplot(111)
    ax.imshow(imgs)
    # 画匹配点对
    plt.plot(match1[:, 0], match1[:, 1], 'o')
    plt.plot(match2[:, 0], match2[:, 1], 'o')
    plt.plot([match1[:, 0], match2[:, 0]], [match1[:, 1], match2[:, 1]])
    # 画邻域拓扑结构
    N = I.shape[0]  # 内点个数
    L = index_X.shape[1]  # 邻域大小
    for i in range(N):
        for j in range(L):
            plt.plot([m1[index_X[i, j], 0], match1[i, 0]], [m1[index_X[i, j], 1], match1[i, 1]], 'b:')
            plt.plot([m2[index_Y[i, j], 0], match2[i, 0]], [m2[index_Y[i, j], 1], match2[i, 1]], 'b:')

    plt.show()



