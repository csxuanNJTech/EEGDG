import sys

sys.path.append('/code')
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from braindecode.torch_ext.util import np_to_var
from model.shallow import ShallowFBCSPNet
from model.maxNorm import MaxNormDefaultConstraint
from until.getDataset import getDatasetTrainORTestDG
from until.getDataset import getDatasetAugAndAugDG
from until.reweighting import *
from until.util import softmax_mse_loss
from until.util import update_ema_variables
from until.util import setDataPath

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--lr', type=int, default=0.001, help='lr')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--max_epochs', type=int, default=2000, help='epoch number')
parser.add_argument('--test_interval', type=int, default=1, help='')
parser.add_argument('--cuda', type=bool, default=True, help='')
parser.add_argument('--seed', type=int, default=20190706, help='')
parser.add_argument('--optim', type=str, default='sgd', help='')
parser.add_argument('--alpha', type=int, default=0.001)

parser.add_argument('--n_levels', type=int, default=1, help='')
parser.add_argument('--cos', default=1, type=int, help='')
parser.add_argument('--lrbl', type=float, default=1.0, help='')
parser.add_argument('--epochb', type=int, default=20, help='')
parser.add_argument('--num_f', type=int, default=2, help='')
parser.add_argument('--sum', type=bool, default=False)
parser.add_argument('--decay_pow', type=float, default=2)
parser.add_argument('--presave_ratio', type=float, default=0.9)
parser.add_argument('--lambdap', type=float, default=70.0)
parser.add_argument('--min_lambda_times', type=float, default=0.01)
parser.add_argument('--lambda_decay_epoch', type=int, default=5)
parser.add_argument('--lambda_decay_rate', type=float, default=1)
parser.add_argument('--first_step_cons', type=float, default=1)
parser.add_argument('--ema_decay', type=float, default=0.99)

args = parser.parse_args()

for test_object in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    object_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    object_list.remove(test_object)
    setDataPath(args)
    print("test_object: {}".format(test_object))
    trainingdata, traininglabel = getDatasetAugAndAugDG(args, test_object)

    trainset = TensorDataset(trainingdata, traininglabel)
    train_iterator = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True)

    testingdata, testinglabel = getDatasetTrainORTestDG(args.test_data, [test_object])

    # obtaining network parm
    classes = torch.unique(traininglabel, sorted=False).numpy()
    n_classes = len(classes)
    input_time_length = trainingdata.shape[2]
    n_chans = int(trainingdata.shape[1] / 2)

    model = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                            final_conv_length='auto', args=args)
    ema_model = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                                final_conv_length='auto', args=args)
    for param in ema_model.parameters():
        param.detach_()
    iter_num = 0

    if args.cuda:
        model.cuda()
        ema_model.cuda()
    model_constraint = MaxNormDefaultConstraint()
    ## set loss and optimizer
    loss_function = nn.CrossEntropyLoss(reduce=False)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    ## model train
    for i in range(args.max_epochs):
        model.train()

        lr_setter(optimizer, i, args)

        for step, (inputs, labels) in enumerate(train_iterator):
            optimizer.zero_grad()

            all_input_vars, label_vars = Variable(inputs).cuda().float(), Variable(labels).cuda().long()
            input_vars = all_input_vars[:, :n_chans, :, :]
            ema_input_vars = all_input_vars[:, n_chans:, :, :]

            preds, currfeatures = model(input_vars)

            pre_features = model.pre_features
            pre_weight1 = model.pre_weight1
            weight1, pre_features, pre_weight1 = weight_learner(currfeatures, pre_features, pre_weight1,
                                                                args, i, step)
            model.pre_features.data.copy_(pre_features)
            model.pre_weight1.data.copy_(pre_weight1)

            with torch.no_grad():
                ema_preds, _ = ema_model(ema_input_vars)
            loss_consistency = torch.sum(weight1 * softmax_mse_loss(preds, ema_preds)) / \
                               label_vars.size()[0]

            loss_cross = loss_function(preds, label_vars).view(1, -1).mm(weight1).view(1) / \
                         label_vars.size()[0]
            loss = args.alpha * loss_consistency + loss_cross

            loss.backward()
            optimizer.step()

            if model_constraint is not None:
                model_constraint.apply(model)

            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            iter_num = iter_num + 1

        if i % args.test_interval == 0:
            model.eval()
            b_x = testingdata.float()
            b_y = testinglabel.long()
            with torch.no_grad():
                input_vars = np_to_var(b_x, pin_memory=False).float()
                labels_vars = np_to_var(b_y, pin_memory=False).type(torch.LongTensor)
                input_vars = input_vars.cuda()
                labels_vars = labels_vars.cuda()

                outputs, _ = model(input_vars)
                y_test_pred = torch.max(outputs, 1)[1].cpu().data.numpy().squeeze()

                acc = metrics.accuracy_score(labels_vars.cpu().data.numpy(), y_test_pred)
                recall = metrics.recall_score(labels_vars.cpu().data.numpy(), y_test_pred,
                                              average='macro')
                f1 = metrics.f1_score(labels_vars.cpu().data.numpy(), y_test_pred, average='macro')
                preci = metrics.precision_score(labels_vars.cpu().data.numpy(), y_test_pred,
                                                average='macro')
                kappa = metrics.cohen_kappa_score(labels_vars.cpu().data.numpy(), y_test_pred)
                print("unseen:" + str(test_object) + " epoch:" + str(i) + " acc: %.4f " % (acc * 100))
            with open('res/res.txt', 'a') as res:
                res.writelines("unseen:" + str(test_object) + " epoch:" + str(i) + " acc: %.4f " % (
                        acc * 100) + " recall: %.4f " % (recall * 100) + " f1: %.4f " % (
                                       f1 * 100) + " preci: %.4f " % (
                                       preci * 100) + " kappa: %.4f " % (
                                       kappa * 100) + '\n')
