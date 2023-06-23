import sys

sys.path.append('/code')
import numpy as np
import torch
from torch.autograd import Variable
import mat73

def getDatasetTrainORTestDG(path, subject_ids):
    total_datas = torch.Tensor([])
    total_labels = torch.Tensor([])
    flag = True
    all_train_data = mat73.loadmat(path)
    for subject_id in subject_ids:
        train_datas = np.transpose(all_train_data['data' + str(subject_id) + '_X'], (2, 1, 0))
        train_labels = np.squeeze(all_train_data['data' + str(subject_id) + '_y'])
        train_datas = Variable(torch.from_numpy(train_datas))
        train_datas = torch.unsqueeze(train_datas, dim=3)
        train_labels = Variable(torch.from_numpy(train_labels))
        train_labels -= train_labels.min()
        if flag:
            total_datas = train_datas
            total_labels = train_labels
            flag = False
        else:
            total_datas = torch.cat((total_datas, train_datas), dim=0)
            total_labels = torch.cat((total_labels, train_labels), dim=0)

    return total_datas, total_labels


def getDatasetAugAndAugDG(args, unseen):
    augment1_all_train_data = mat73.loadmat(args.train_data_augment1)
    augment1_train_datas = np.transpose(augment1_all_train_data['unseenDatas'][unseen - 1][0], (2, 1, 0))
    augment1_train_labels = np.transpose(augment1_all_train_data['unseenlabels'][unseen - 1][0])
    augment1_train_datas = Variable(torch.from_numpy(augment1_train_datas))
    augment1_train_datas = torch.unsqueeze(augment1_train_datas, dim=3)
    augment1_train_labels = Variable(torch.from_numpy(augment1_train_labels))
    augment1_train_labels -= augment1_train_labels.min()
    augment1_train_labels = augment1_train_labels.reshape(-1)

    augment2_all_train_data = mat73.loadmat(args.train_data_augment2)
    augment2_train_datas = np.transpose(augment2_all_train_data['unseenDatas'][unseen - 1][0], (2, 1, 0))
    augment2_train_labels = np.transpose(augment2_all_train_data['unseenlabels'][unseen - 1][0])
    augment2_train_datas = Variable(torch.from_numpy(augment2_train_datas))
    augment2_train_datas = torch.unsqueeze(augment2_train_datas, dim=3)
    augment2_train_labels = Variable(torch.from_numpy(augment2_train_labels))
    augment2_train_labels -= augment2_train_labels.min()
    augment2_train_labels = augment2_train_labels.reshape(-1)

    assert np.where((augment1_train_labels - augment2_train_labels) != 0)[0].shape[0] == 0
    total_datas = torch.cat((augment1_train_datas, augment2_train_datas), dim=1)
    return total_datas, augment1_train_labels
