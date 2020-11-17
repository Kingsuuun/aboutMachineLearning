from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def datasets_demo():
    '''
    sklearn数据集使用
    '''
    # 获取数据
    iris = load_iris()
    print("鸢尾花数据集：\n", iris)
    print("鸢尾花的数据描述：\n", iris.DESCR)
    print("鸢尾花的特征值的名字：\n", iris.feature_names)
    print("查看特征值：\n", iris.data, iris.data.shape) # .shape 是直接查看几行几列

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.2, random_state=22)
    print("训练集的特征值：\n", x_train, x_train.shape)
    
    return None

def dict_demo():
    '''
    字典特征提取 
    '''
    data = [{'city': '北京','temperature':100}, {'city': '上海','temperature':60}, {'city': '深圳','temperature':30}]
    # 1、实例化一个转换器类
    transfer = DictVectorizer(sparse = False) # 默认输出的是稀疏矩阵
    transfer1 = DictVectorizer()
    # 2、调用fit_tansform()
    data_new = transfer.fit_transform(data)
    # data_two = transfer1.fit_transform(data)
    print("data_new:\n", data_new)
    print("特征名字：\n", transfer.get_feature_names())
    # print("data_two:\n", data_two)
    return None

def count_demo():
    '''
    # 文本特征抽取： CountVectorrizer
    '''
    data = ["life is short,i like python", "life is too long,i dislike python"]
    # 1、 实例化一个转换类
    transfer = CountVectorizer()
    # 2、 调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray(), type(data_new)) # .toarray() 可以使得稀疏sparse矩阵变成二阶矩阵
    print("特征名字：\n", transfer.get_feature_names())
    return None

def count_chinese_demo():
    '''
    # 文本特征抽取： CountVectorrizer
    '''
    data = ["人生苦短，我喜欢Python" "生活太长久，我不喜欢Python"]
    # 1、 实例化一个转换类
    transfer = CountVectorizer()
    # 2、 调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray(), type(data_new)) # .toarray() 可以使得稀疏sparse矩阵变成二阶矩阵
    print("特征名字：\n", transfer.get_feature_names())
    return None
    
if __name__ == "__main__":
    # 代码1: sklearn数据集使用
    # datasets_demo()
    # 代码2: 字典特征提取
    # dict_demo()
    # 代码3: 文本的特征抽取
    # count_demo()
    # 代码4: 中文文本特征抽取
    count_chinese_demo()