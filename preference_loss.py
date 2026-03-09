import torch


def preference_loss(r_chosen, r_rejected):

    return -torch.log(
        torch.sigmoid(r_chosen - r_rejected)
    ).mean()
