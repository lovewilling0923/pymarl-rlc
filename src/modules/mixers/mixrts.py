import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from ..agents.rtc_agent import RecurrentTreeCell


class MixingTree(nn.Module):
    def __init__(self, args):
        super(MixingTree, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.hyper_states = nn.Linear(self.state_dim, args.mixing_embed_dim)
        self.depth = args.mix_q_tree_depth
        self.beta = 0  #args.beta
        self.mix_tree = RecurrentTreeCell(self.args.n_agents, args.mixing_embed_dim, self.args.n_agents,
                                       tree_depth=self.depth, beta=self.beta, kernel=args.kernel)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, agent_qs, states):  # states.shape: (episode_num, max_episode_len, state_shape)
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        states_ = self.hyper_states(states).view(-1, self.args.mixing_embed_dim)

        q_tot_p, _, _ = self.mix_tree(agent_qs, states_)  # The final w is produced by states with a layer tree
        q_tot_p = q_tot_p.view(-1, self.n_agents, 1)
        if self.args.softmax_w:
            q_tot_p = self.softmax(q_tot_p).view(-1, self.n_agents, 1)
        q_tot = torch.bmm(agent_qs, q_tot_p)
        # print(q_values[0], q_tot_p[0])
        q_tot = q_tot.view(bs, -1, 1)
        return q_tot
