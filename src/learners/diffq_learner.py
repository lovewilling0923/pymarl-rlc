import copy

import numpy as np
import torch as th
import torch.nn.functional as F
from components.episode_buffer import EpisodeBatch
from modules.mixers.qnam import QNAMer
from modules.intrinsic.diffusion_network import Diffusion, VAE
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.optim import RMSprop, Adam


class DiffQ_Learner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "qnam":
                self.mixer = QNAMer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            # self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.eval_diff_network = VAE(args.rnn_hidden_dim, args.obs_shape)
        self.target_diff_network = VAE(args.rnn_hidden_dim, args.obs_shape)
        self.params_mixer = list(self.mixer.parameters())
        self.params_mixer += list(self.eval_diff_network.parameters())

        if self.args.use_cuda:
            self.eval_diff_network.cuda()
            self.target_diff_network.cuda()

        self.target_diff_network.load_state_dict(
            self.eval_diff_network.state_dict())

        self.optimiser = RMSprop(
            params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.optimiser_diff = Adam(params=self.params_mixer, lr=args.lr, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]
        last_actions_onehot = th.cat([th.zeros_like(
            actions_onehot[:, 0].unsqueeze(1)), actions_onehot], dim=1)  # last_actions

        self.mac.init_hidden(batch.batch_size)
        initial_hidden = self.mac.hidden_states.clone().detach()
        initial_hidden = initial_hidden.reshape(
            -1, initial_hidden.shape[-1]).to(self.args.device)
        input_here = th.cat((batch["obs"], last_actions_onehot),
                            dim=-1).permute(0, 2, 1, 3).to(self.args.device)

        mac_out, hidden_store, local_qs = self.mac.agent.forward(
            input_here.clone().detach(), initial_hidden.clone().detach())   # (bs*n, t, h_dim)
        # (bs, t, n, h_dim)
        hidden_store = hidden_store.reshape(
            -1, input_here.shape[1], hidden_store.shape[-2], hidden_store.shape[-1]).permute(0, 2, 1, 3)

        input_var_here = batch["obs"][:, :-1]
        recon, mean, std = self.eval_diff_network(hidden_store[:, :-1])
        output_vae_here = th.einsum("btnd, btnd->btnd", recon, input_var_here)
        recon_loss = F.mse_loss(output_vae_here, input_var_here)
        KL_loss = -0.5 * (1 + th.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        # entropy_loss = - (recon * th.log2(recon)).sum(-1).mean()/self.n_agents
        entropy_loss = F.l1_loss(recon, target=th.zeros_like(recon), size_average=True)
        vae_loss = 0.5 * recon_loss + KL_loss + 0.7 * entropy_loss

        #vae_loss /= batch.max_seq_length

        with th.no_grad():
            latent, _, _ = self.eval_diff_network.encode(hidden_store[:, :-1])

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(
            mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        self.target_mac.init_hidden(batch.batch_size)
        initial_hidden_target = self.target_mac.hidden_states.clone().detach()
        initial_hidden_target = initial_hidden_target.reshape(
            -1, initial_hidden_target.shape[-1]).to(self.args.device)
        target_mac_out, target_hidden_store, _ = self.target_mac.agent.forward(
            input_here.clone().detach(), initial_hidden_target.clone().detach())

        target_mac_out = target_mac_out[:, 1:]

        target_hidden_store = target_hidden_store.reshape(
            -1, input_here.shape[1], target_hidden_store.shape[-2], target_hidden_store.shape[-1]).permute(0, 2, 1, 3)
        with th.no_grad():
            target_latent, _, _ = self.target_diff_network.encode(target_hidden_store[:, 1:])

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], latent)
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], target_latent)

        N = getattr(self.args, "n_step", 1)
        if N == 1:
            # Calculate 1-step Q-Learning targets
            targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        else:
            # N step Q-Learning targets
            n_rewards = th.zeros_like(rewards)
            gamma_tensor = th.tensor([self.args.gamma**i for i in range(N)], dtype=th.float, device=n_rewards.device)
            steps = mask.flip(1).cumsum(dim=1).flip(1).clamp_max(N).long()
            for i in range(batch.max_seq_length - 1):
                n_rewards[:,i,0] = ((rewards * mask)[:,i:i+N,0] * gamma_tensor[:(batch.max_seq_length - 1 - i)]).sum(dim=1)
            indices = th.linspace(0, batch.max_seq_length-2, steps=batch.max_seq_length-1, device=steps.device).unsqueeze(1).long()
            n_targets_terminated = th.gather(target_max_qvals*(1-terminated),dim=1,index=steps.long()+indices-1)
            targets = n_rewards + th.pow(self.args.gamma, steps.float()) * n_targets_terminated

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()
        #
        # norm_loss = F.l1_loss(local_qs, target=th.zeros_like(
        #     local_qs), size_average=True)
        # loss += norm_loss / 10
        loss += vae_loss * 0.02

        # Optimise
        self.optimiser.zero_grad()
        self.optimiser_diff.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        grad_norm_diff = th.nn.utils.clip_grad_norm_(self.params_mixer, self.args.grad_norm_clip)
        self.optimiser.step()
        self.optimiser_diff.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("vae_loss", vae_loss.item(), t_env)
            self.logger.log_stat("recon_loss", recon_loss.item(), t_env)
            self.logger.log_stat("entropy_loss", entropy_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("grad_norm_diff", grad_norm_diff, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_diff_network.load_state_dict(
            self.eval_diff_network.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.to(self.args.device)
            self.target_mixer.to(self.args.device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        th.save(self.optimiser_mixer.state_dict(), "{}/opt.th".format(path))
        th.save(self.eval_diff_network.state_dict(),
                "{}/pid.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(
                th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.eval_diff_network.load_state_dict(
            th.load("{}/pid.th".format(path), map_location=lambda storage, loc: storage))
        self.target_diff_network.load_state_dict(
            th.load("{}/pid.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser_mixer.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))