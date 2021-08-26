import numpy as np
from numpy import inf
from matplotlib import pyplot as plt
import random

def distEclud(vecA, vecB):
    "两个向量的欧式距离计算公式"
    return np.sqrt(sum(np.power(vecA - vecB, 2)))

def kmeans(dataset, k):
    """K-means 聚类算法

    Args:
        dataset ([ndarray]): 数据集，二维数组
        k ([int]): 聚簇数量
    """
    m = np.shape(dataset)[0]  # 样本个数
    
    # 1, 随机初始化聚类中心点
    center_indexs = random.sample(range(m), k)
    center = dataset[center_indexs,:]
    
    cluster_assessment = np.zeros((m, 2))
    cluster_assessment[:, 0] = -1  # 将所有的类别置为 -1
    cluster_changed = True 
    while cluster_changed:
        cluster_changed = False
        # 4-8，计算样本x_i与各聚类中心的距离，根据距离最近的聚类中心确定x_j的簇标记，并将对应样本x_i划入相应的簇
        for i in range(m):
            # 初始化样本和聚类中心的距离，及样本对应簇
            min_dist = inf
            c = 0
            # 确定每一个样本离哪个中心点最近，和属于哪一簇
            for j in range(k):
                dist = distEclud(dataset[i,:], center[j,:])
                if dist < min_dist:
                    min_dist = dist
                    c = i
            # 更新样本所属簇
            if cluster_assessment[i, 0] != c:  # 仍存在数据在前后两次计算中有类别的变动，未达到迭代停止要求
                cluster_assessment[i, :] = c, min_dist
                cluster_changed = True
        # 9-15 更新簇中心点位置
        for j in range(k):
            changed_center = dataset[cluster_assessment[:,0] == j].mean(axis=0)
            center[j,:] = changed_center
            
    return cluster_assessment, center

def show_cluster(dataSet, k, centroids, clusterAssement):
    """
    针对二维数据进行聚类结果绘图
    """
    numSamples, dim = dataSet.shape
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', '<r', 'pr']
    center_mark = ['*r', '*b', '*g', '*k', '*r', '*r', '*r', '*r']

    for i in range(numSamples):
        markIndex = int(clusterAssement[i,0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex], markersize=2)
    for j in range(k):
        plt.plot(centroids[j, 0], centroids[j, 1], center_mark[j], markersize=12)
    plt.show()
    

if __name__ == '__main__':
    x1 = np.random.randint(0, 50, (50, 2))
    x2 = np.random.randint(40, 100, (50, 2))
    x3 = np.random.randint(90, 120, (50, 2))
    x4 = np.random.randint(110, 160, (50, 2))
    test = np.vstack((x1, x2, x3, x4))

    # 对特征进行聚类
    result, center = kmeans(test, 4, is_kmeans=False, is_random=False)
    print(center)
    # show_cluster(test, 4, center, result)  # 报错