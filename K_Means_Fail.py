from learning_curve import plot_learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans


digits = load_digits()
X = digits.data
y = digits.target


def show(image):
    test = image.reshape((8, 8))  # 从一维变为二维，这样才能被显示
    print(test.shape)  # 查看是否是二维数组
    # print(test)
    plt.imshow(test, cmap=plt.cm.gray)  # 显示灰色图像
    plt.show()


def standard_demo(data):
    transfer = StandardScaler()
    data_new = transfer.fit_transform(data)
    print(data_new)
    return data_new


def pca_demo(data):
    transfer = PCA(n_components=0.92)
    data_new = transfer.fit_transform(data)
    print(data_new)
    return data_new


def k_Means_demo(data, label):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=6)
    # 训练模型
    estimate = KMeans(n_clusters=10,
                      max_iter=100,
                      tol=0.01,  # 两次迭代中心的距离之差
                      verbose=1,
                      n_init=30)  # 进行三次初始化，找一个最好的
    estimate.fit(data)  # 模型构建好了
    # 模型评估的两种方法：1：直接比对预测值与真实值；
    y_predict = estimate.predict(data)
    print(silhouette_score(data, y_predict))
    # print("直接比对预测值与真实值：\n", y_test == y_predict)
    # # 2：计算准确率
    # score = estimate.score(X_test, y_test)
    # print("准确率为：\n", score)
    # # 绘制学习曲线!!!!
    # fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=144)
    # cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=6)  # 交叉验证 cross validation
    # plot_learning_curve(ax, estimate, "Learning_Curve", data, label, ylim=(0.8, 1.01), cv=cv)
    # # 混淆矩阵
    # cm = confusion_matrix(y_test, y_predict)
    # print(cm)
    # # 可视化显示混淆矩阵!!!!
    # fig, ax = plt.subplots(1, 1, figsize=(40, 40), dpi=140)
    # ax.spines['right'].set_color('none')
    # ax.spines['left'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    # ax.set_title('Visualization of Confusion Matrix', size=30)
    # ax.tick_params(labelsize=25)
    # ax0 = ax.matshow(cm, cmap=plt.cm.viridis)
    # fig.colorbar(ax0, ax=ax)
    # 计算精确率与召回率
    # report = classification_report(y_test, y_predict, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #                                target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    # print(report)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # 查看数据集，以图片显示
    print(X.shape, y.shape)
    print(X, y)
    # show(X[1795])
    # 数据集预处理（标准化、特征降维）
    X_new = standard_demo(X)
    X_new = pca_demo(X_new)
    print(X_new.shape)  # 从64维降到了40维
    # 机器学习建模
    # 调参
    # 模型评估
    # 学习曲线
    k_Means_demo(X_new, y)  # 数据已经准备好了
