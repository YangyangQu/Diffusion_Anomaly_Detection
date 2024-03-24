import numpy as np
from sklearn.preprocessing import StandardScaler


class OutlierSampler(object):
    def __init__(self, data_path=r'mnist.npz'):
        data_dic = np.load(data_path, allow_pickle=True)

        self.X_train, self.data_test_true, self.data_test_false, self.data_label_test = self.normalize(data_dic)
        self.Y = None
        self.nb_train = self.X_train.shape[0]
        self.mean = 0
        self.sd = 0

    def normalize(self, data_dic):

        data = data_dic['arr_0']
        label = data_dic['arr_1']

        # 筛选标签为0的索引
        index = np.where(label == 0)
        index1 = np.where(label == 1)

        # 根据索引筛选
        data_0 = data[index]
        label_0 = label[index]
        data_1 = data[index1]
        label_1 = label[index1]

        # 按比例分割
        train_num = int(0.9 * len(data_0))
        test_num = int(0.1 * len(data_1))
        data_train = data_0[:train_num]

        # selected_indices = np.random.choice(len(data_0[train_num:]), 100, replace=False)
        # data_test_true = data_0[train_num:][selected_indices]
        # data_test = data_0[train_num:] + data_1
        data_test_true = data_0[train_num:]
        
        # selected_indices_false = np.random.choice(len(data_1), size=100, replace=False)
        # # Use the selected indices to create data_test_false
        # data_test_false = data_1[selected_indices_false]

        data_test_false = data_1
        # data_test_false= data_1[:test_num]
        print("data_test_true",data_test_true.shape)
        print("data_test_false",data_test_false.shape)
        data_test = np.concatenate([data_0[train_num:], data_1], axis=0)
        label_train = label_0[:train_num]
        # label_test = label_0[train_num:] + label_1
        label_test = np.concatenate([label_0[train_num:], label_1], axis=0)
        data_label_test = (data_test, label_test)
        # Standardize the data
        scaler = StandardScaler()
        data_train = scaler.fit_transform(data_train)
        data_test = scaler.transform(data_test)
        data_test_true = scaler.transform(data_test_true)
        data_test_false = scaler.transform(data_test_false)
        return data_train, data_test_true, data_test_false, data_label_test

    def train(self, batch_size, label=False):
        indx = np.random.randint(low=0, high=self.nb_train, size=batch_size)
        if label:
            return self.X_train[indx, :], self.Y[indx]
        else:
            return self.X_train[indx, :]

    def load_all(self):
        return self.X_train, None
