import torch


def prepare_training_inputs(sampled_exps, device='cpu'):

    states = torch.cat(sampled_exps[0], dim=0)
    actions = torch.cat(sampled_exps[1], dim=0)
    rewards = torch.cat(sampled_exps[2], dim=0)
    next_states = torch.cat(sampled_exps[3], dim=0)
    dones = torch.cat(sampled_exps[4], dim=0)
    
    return states, actions, rewards, next_states, dones


def soft_update(net, net_target, tau):
    
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
