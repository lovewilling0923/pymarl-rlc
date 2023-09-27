import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class QAMixer(nn.Module):
    def __init__(self, args):
        super(QAMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.u_dim = int(args.emb)
        # print(args)
        self.attention = SelfAttention(self.n_agents, heads=args.heads, mask=False)
        # self.q_embedding = nn.Linear(self.n_agents,args.emb)


        self.n_query_embedding_layer1 = args.n_query_embedding_layer1
        self.n_query_embedding_layer2 = args.emb
        self.n_key_embedding_layer1 = args.emb
        self.n_head_embedding_layer1 = args.n_head_embedding_layer1
        self.n_head_embedding_layer2 = args.n_head_embedding_layer2
        self.n_attention_head = args.n_attention_head
        self.n_constrant_value = args.n_constrant_value

        self.query_embedding_layers = nn.ModuleList()
        for i in range(self.n_attention_head):
            self.query_embedding_layers.append(nn.Sequential(nn.Linear(self.state_dim, self.n_query_embedding_layer1),
                                                           nn.ReLU(),
                                                           nn.Linear(self.n_query_embedding_layer1, self.n_query_embedding_layer2)))
        
        self.key_embedding_layers = nn.ModuleList()
        for i in range(self.n_attention_head):
            self.key_embedding_layers.append(nn.Linear(self.u_dim, self.n_key_embedding_layer1))


        self.scaled_product_value = np.sqrt(args.n_query_embedding_layer2)

        self.head_embedding_layer = nn.Sequential(nn.Linear(self.state_dim, self.n_head_embedding_layer1),
                                                  nn.ReLU(),
                                                  nn.Linear(self.n_head_embedding_layer1, self.n_head_embedding_layer2))
        
        self.constrant_value_layer = nn.Sequential(nn.Linear(self.state_dim, self.n_constrant_value),
                                                  nn.ReLU(),
                                                  nn.Linear(self.n_constrant_value, 1))


    def forward(self, agent_qs, states, utility):
        # print(agent_qs.size())
        # print(states.size())
        # print(utility.size())
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        us = utility.reshape(-1,self.u_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        q_lambda_list = []
        for i in range(self.n_attention_head):
            state_embedding = self.query_embedding_layers[i](states)
            u_embedding = self.key_embedding_layers[i](us)

            # shape: [-1, 1, state_dim]
            state_embedding = state_embedding.reshape(-1, 1, self.n_query_embedding_layer2)
            # shape: [-1, state_dim, n_agent]
            u_embedding = u_embedding.reshape(-1, self.n_agents, self.n_key_embedding_layer1)
            u_embedding = u_embedding.permute(0, 2, 1)

            # shape: [-1, 1, n_agent]
            raw_lambda = th.matmul(state_embedding, u_embedding) / self.scaled_product_value
            q_lambda = F.softmax(raw_lambda, dim=-1)

            q_lambda_list.append(q_lambda)

        # shape: [-1, n_attention_head, n_agent]
        q_lambda_list = th.stack(q_lambda_list, dim=1).squeeze(-2)

        # shape: [-1, n_agent, n_attention_head]
        q_lambda_list = q_lambda_list.permute(0, 2, 1)

        # shape: [-1, 1, n_attention_head]
        q_h = th.matmul(agent_qs, q_lambda_list)

        if self.args.type == 'weighted':
            # shape: [-1, n_attention_head, 1]
            w_h = th.abs(self.head_embedding_layer(states))
            w_h = w_h.reshape(-1, self.n_head_embedding_layer2, 1)

            # shape: [-1, 1]
            sum_q_h = th.matmul(q_h, w_h)
            sum_q_h = sum_q_h.reshape(-1, 1)
        else:
            # shape: [-1, 1]
            sum_q_h = q_h.sum(-1)
            sum_q_h = sum_q_h.reshape(-1, 1)

        c = self.constrant_value_layer(states)
        q_tot = sum_q_h + c
        q_tot = q_tot.view(bs, -1, 1)
        return q_tot

    def _get_us(self, states):
        agent_own_state_size = self.args.agent_own_state_size
        with th.no_grad():
            us = states[:, :agent_own_state_size*self.n_agents].reshape(-1, agent_own_state_size)
        return us


class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x, mask):

        b, t, e = x.size()
        h = self.heads
        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * h, t, t)

        if self.mask:  # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        if mask is not None:
            dot = dot.masked_fill(mask == 0, -1e9)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)

def mask_(matrices, maskval=0.0, mask_diagonal=True):

    b, h, w = matrices.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval