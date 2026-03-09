import torch


def ppo_step(policy, reward_model, input_ids):

    logits = policy(input_ids)

    probs = torch.softmax(logits,dim=-1)

    actions = torch.multinomial(probs.view(-1,probs.size(-1)),1)

    reward = reward_model(actions)

    loss = -reward.mean()

    return loss
