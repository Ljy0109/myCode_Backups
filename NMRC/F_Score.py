"""
Average F Score 计算程序，一种度量图像匹配性能的参数
https://blog.csdn.net/cgwang_1580/article/details/68944319
"""
import numpy as np
import cv2 as cv


def calculation(m1, m2, match1, match2, H, img):
    h, w, _ = img.shape
    eip = 1.5

    """ones1 = np.ones((m1.shape[0], 1))
    ones2 = np.ones((m2.shape[0], 1))
    m1 = np.mat(np.hstack((m1, ones1)).T)
    m2 = np.mat(np.hstack((m2, ones2)).T)
    m1_H = np.array([H * m1[:, i] for i in range(m1.shape[1])]).reshape(m1.shape[1], 3).T
    m2_H = np.array([H.I * m2[:, i] for i in range(m2.shape[1])]).reshape(m2.shape[1], 3).T

    ones1 = np.ones((match1.shape[0], 1))
    ones2 = np.ones((match2.shape[0], 1))
    match1 = np.mat(np.hstack((match1, ones1)).T)
    match2 = np.mat(np.hstack((match2, ones2)).T)
    match1_H = np.array([H * match1[:, i] for i in range(match1.shape[1])]).reshape(match1.shape[1], 3).T
    match2_H = np.array([H.I * match2[:, i] for i in range(match2.shape[1])]).reshape(match2.shape[1], 3).T"""

    m1_re = np.float32(m1).reshape(-1, 1, 2)
    m1_H = cv.perspectiveTransform(m1_re, H).reshape(m1_re.shape[0], 2)
    match1_re = np.float32(match1).reshape(-1, 1, 2)
    match1_H = cv.perspectiveTransform(match1_re, H).reshape(match1_re.shape[0], 2)

    # 去掉不合格（计算结果超出在B图像坐标）的特征点，剩下的特征点数记为n1
    index1 = np.array(np.zeros(m1_H.shape[0]), int)
    ti = -1
    for i in range(m1_H.shape[0]):
        if m1_H[i, 0] > h or m1_H[i, 1] > w:
            ti = ti + 1
            index1[ti] = i
    if ti != -1:
        index1 = index1[0:ti]
        m1_n = np.delete(m1_H, index1, axis=0)  # 图A剩下的特征点由H1计算出在图B中的坐标
    # m1 = np.delete(m1, index1, axis=1)
        m1_n_m2 = np.delete(m2, index1, axis=0)
    else:
        m1_n = m1_H
        m1_n_m2 = m2
    n1 = m1_n.shape[0]

    """index2 = np.array(np.zeros(m1_H.shape[0]), int)
    ti = -1
    for i in range(m2_H.shape[0]):
        if m2_H[i, 0] > h or m2_H[i, 1] > w:
            ti = ti + 1
            index2[ti] = i
    if ti != -1:
        index2 = index2[0:ti]
        m2_n = np.delete(m2_H, index2, axis=0)
    else:
        m2_n = m2_H
    n2 = m2_n.shape[0]

    fen_mu = np.min([n1, n2])  # 计算重复率的分母"""

    # 计算重复特征点个数
    ti = -1
    index_m1n_m2 = np.array(np.zeros(m1_n.shape[0]), int)
    for i in range(m1_n.shape[0]):
        x1 = m1_n[i, 0]
        y1 = m1_n[i, 1]
        x2 = m1_n_m2[i, 0]
        y2 = m1_n_m2[i, 1]
        dist = ((x1 - x2)**2 + (y1 - y2)**2)**(1/2)
        if dist > eip:
            ti = ti + 1
            index_m1n_m2[ti] = i
    if ti != -1:
        index_m1n_m2 = index_m1n_m2[0:ti]
        repeat_point = np.delete(m1_n_m2, index_m1n_m2, axis=0)  # 以m2形式表示重复点
    else:
        repeat_point = m1_n_m2
    print('repeat point = ', repeat_point.shape[0])

    # 计算NMRC重复特征点个数
    ti = -1
    index_match1H = np.array(np.zeros(match1_H.shape[0]), int)
    for i in range(match1_H.shape[0]):
        x1 = match1_H[i, 0]
        y1 = match1_H[i, 1]
        x2 = match2[i, 0]
        y2 = match2[i, 1]
        dist = ((x1 - x2)**2 + (y1 - y2)**2)**(1/2)
        if dist > eip:
            ti = ti + 1
            index_match1H[ti] = i
    if ti != -1:
        index_match1H = index_match1H[0:ti]
        repeat_point_match = np.delete(match1_H, index_match1H, axis=0)  # 以m2形式表示重复点
    else:
        repeat_point_match = match1_H
    print('repeat point of matching = ', repeat_point_match.shape[0])

    """index_match2 = np.array(np.zeros(match2.shape[0]), int)
    ti = -1
    for i in range(match2.shape[0]):
        for j in range(repeat_point.shape[0]):
            if match2[i, 0] == repeat_point[j, 0] and match2[i, 1] == repeat_point[j, 1]:
                ti = ti + 1
                index_match2[ti] = i
                break
    index_match2 = index_match2[0:ti]
    correct_matches = ti + 1"""
    print('correct_match = ', repeat_point_match.shape[0])
    print('match.shape = ', match2.shape)
    false_matches = match2.shape[0] - repeat_point_match.shape[0]

    recall = repeat_point_match.shape[0] / repeat_point.shape[0]
    print('recall = ', recall)
    precision = repeat_point_match.shape[0] / (repeat_point_match.shape[0] + false_matches)
    print('precision = ', precision)
    F_s = (2*precision*recall)/(precision + recall)
    print('F_s = ', F_s)
    return F_s





