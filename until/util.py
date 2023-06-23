from torch.nn import functional as F
import warnings


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def update_ema_variables_2(model, ema_model, m):
    for p1, p2 in zip(model.parameters(), ema_model.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


# consistency loss
def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax) ** 2
    return mse_loss


def setDataPath(args):
    args.test_data = "./data/data_2a/testPython_2a_4_38_1125_my.mat"
    args.train_data = "./data/data_2a/trainPython_2a_4_38_1125_my.mat"
    args.train_data_augment1 = './data/data_2a/Train_2a.mat'
    args.train_data_augment2 = './data/data_2a/EMATrain_2a.mat'
