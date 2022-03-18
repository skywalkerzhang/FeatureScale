# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np


class MyDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class Net(nn.Module):
    def __init__(self, n_feature, n_output):
        # 初始化数组，参数分别是初始化信息，特征数，隐藏单元数，输出单元数
        super(Net, self).__init__()
        # 此步骤是官方要求
        self.bn = nn.BatchNorm1d(n_feature)
        self.hidden1 = nn.Linear(n_feature, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.hidden3 = nn.Linear(128, 64)
        self.hidden4 = nn.Linear(64, 16)
        self.hidden5 = nn.Linear(16, 8)
        # 设置输入层到隐藏层的函数
        self.predict = nn.Linear(8, n_output)
        # 设置隐藏层到输出层的函数

    def forward(self, x):
        # 定义向前传播函数
        x = self.bn(x)
        x = F.relu(self.hidden1(x))
        # 给x加权成为a，用激励函数将a变成特征b
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = self.predict(x)
        # 给b加权，预测最终结果
        return x


def get_train_data(path, sep=','):
    df = pd.read_csv(path, sep=sep)
    # 每个仓库根据月份求日销售的和
    # new_df = pd.DataFrame(columns=['shop_id', 'item_id', 'date_block_num', 'item_cnt_month'])
    df = feature_link(df)
    data_group = df.groupby(['shop_id', 'item_id', 'date_block_num'])['item_cnt_day'].sum()
    new_df = data_group.reset_index()
    new_df = new_df.rename(columns={'item_cnt_day': 'item_cnt_month'})
    # 后续考虑是否需要item price
    return new_df


def generate_testset(path, sep=','):
    df = pd.read_csv(path, sep=sep)
    df = feature_link(df)
    return df


def get_tensor(X):
    return torch.tensor(X.values, dtype=torch.float32)


# link the feature for both training set and test set
def feature_link(data: pd.DataFrame):
    # all names need one-hot encoding
    # Link name
    shop_info = pd.read_csv(f'./competitive-data-science-predict-future-sales/shops.csv')
    shop_info = shop_info.set_index(['shop_id'])
    shop_info = shop_info['shop_name']
    shop_info = shop_info.to_dict()
    data['shop_name'] = data['shop_id'].apply(lambda x: shop_info[x])

    # Link item name and cid
    item_info = pd.read_csv(f'./competitive-data-science-predict-future-sales/items.csv')
    item_info = item_info.set_index(['item_id'])
    item_name = item_info['item_name']
    item_cid = item_info['item_category_id']
    item_name = item_name.to_dict()
    item_cid = item_cid.to_dict()
    data['item_name'] = data['item_id'].apply(lambda x: item_name[x])
    data['item_category_id'] = data['item_id'].apply(lambda x: item_cid[x])

    # Link item c name
    item_categories_info = pd.read_csv(f'./competitive-data-science-predict-future-sales/item_categories.csv')
    item_categories_info = item_categories_info.set_index(['item_category_id'])
    item_categories_info = item_categories_info['item_category_name']
    item_categories_info = item_categories_info.to_dict()
    data['item_category_name'] = data['item_category_id'].apply(lambda x: item_categories_info[x])
    return data


# 给测试集用的，用来做商品价格预测，看是否影响销量
def predict_prices(data):
    pass


# 使用相对误差来计算Loss，因为房价相差很大
def log_rmse(net, features, labels):
    # 将输入input张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量。
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    loss = nn.MSELoss()
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


def train_val_split(X, ratio=0.7):
    return train_test_split(X, train_size=ratio)


def train(my_train_set, device, features, epochs=100, lr=0.01, weight_decay=1e-5):
    net = Net(features, 1)
    net.to(device)
    # 使用adam作为有偶花旗
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    # train_X = train_X.to(device)
    # train_y = train_y.to(device)
    # 下面的会报错，不能这么写
    # my_train_set = my_train_set.to(device)

    for epoch in range(epochs):
        running_loss = 0.00
        # enumerate里前面是idx，后面是data
        for i, data in enumerate(my_train_set):
            # 不能直接to device
            # list' object has no attribute 'to'
            x, y = data
            x = x.to(device)
            y = y.to(device)
        # for x, y in zip(train_X, train_y):
            optimizer.zero_grad()
            # 获得输出
            output = net(x)
            # 计算loss
            loss_fn = nn.MSELoss()
            # RMSE_loss = torch.sqrt(loss_fn(output, y))
            # loss反向传播，计算梯度
            # RMSE_loss.backward()
            loss = loss_fn(output, y)
            loss.backward()
            # 更新参数
            optimizer.step()

            # running_loss += RMSE_loss.item()
            running_loss += loss
            # 定期输出梯度
            if (i + 1) % 2000 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.00
    return net


def test(test_x, net, device):
    # 不需要梯度
    test_x = test_x.to(device)
    net.to(device)
    res = net(test_x).detach()
    return res


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # set random state
    torch.cuda.manual_seed(1)

    train_set = get_train_data(f'./competitive-data-science-predict-future-sales/sales_train.csv')

    # 划分训练集和测试集
    train_set, val_set = train_val_split(train_set)

    train_X, train_y = get_tensor(train_set.iloc[:, :-1]), get_tensor(train_set.iloc[:, -1])
    train_y = torch.unsqueeze(train_y, dim=1)
    train_set = MyDataset(train_X, train_y)
    val_X, val_y = get_tensor(val_set.iloc[:, :-1]), get_tensor(val_set.iloc[:, -1])

    train_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=0)

    test_set = generate_testset(f'./competitive-data-science-predict-future-sales/test.csv')
    test_set = get_tensor(test_set)

    # loss nan学习率太高
    # cur best epochs=10, lr=0.001, batch_size=256
    features = 7
    net = train(train_loader, device, features, epochs=15, lr=0.001)
    net.to(device)

    # 验证
    val_preds = test(val_X, net, device)
    val_preds = torch.squeeze(val_preds, dim=1).detach()
    val_y = val_y.to(device).detach()
    loss = F.mse_loss(val_preds.float(), val_y.float(), reduction='mean')
    print(loss)
    preds = test(test_set, net, device)
    preds = torch.squeeze(preds, dim=1).detach()
    preds = preds.to("cpu")
    out = pd.DataFrame(preds.numpy(), columns=['item_cnt_month'])
    idx = pd.DataFrame(np.arange(0, len(preds)), columns=['ID'])
    out = pd.concat([idx, out], axis=1)
    out.to_csv("test.csv", index=False)