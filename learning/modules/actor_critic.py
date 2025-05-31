import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
import torch.nn.functional as F

from .utils import create_MLP
from .actor import Actor
from .critic import Critic
from .estimator import HIMEstimator
import os
import copy

class ActorCritic(nn.Module):
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_one_step_obs,  # 新增
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        normalize_obs=False,
        **kwargs,
    ):

        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(ActorCritic, self).__init__()
        self.num_actor_obs = num_actor_obs
        self.history_size = int(num_actor_obs / num_one_step_obs)
        self.num_one_step_obs = num_one_step_obs
        mlp_input_dim_a = num_one_step_obs + 3 + 16
        self.actor = Actor(
            mlp_input_dim_a,
            num_actions,
            actor_hidden_dims,
            activation,
            init_noise_std,
            normalize_obs,
        )

        self.critic = Critic(
            num_critic_obs, critic_hidden_dims, activation, normalize_obs
        )
        self.estimator = HIMEstimator(
            temporal_steps=self.history_size, num_one_step_obs=num_one_step_obs
        )
        print(f"Actor MLP: {self.actor.mean_NN}")
        print(f"Critic MLP: {self.critic.NN}")
        print(f'Estimator: {self.estimator.encoder}')

    @property
    def action_mean(self):
        return self.actor.action_mean

    @property
    def action_std(self):
        return self.actor.action_std

    @property
    def entropy(self):
        return self.actor.entropy

    @property
    def std(self):
        return self.actor.std

    def update_distribution(self, observations):
        with torch.no_grad():
            vel, latent = self.estimator(observations)
        actor_input = torch.cat((observations[:,:self.num_one_step_obs], vel, latent), dim=-1)
        self.actor.update_distribution(actor_input)

    def act(self, observations, **kwargs):
        vel, latent = self.estimator(observations)
        actor_input = torch.cat((observations[:,:self.num_one_step_obs], vel, latent), dim=-1)
        return self.actor.act(actor_input)

    def get_actions_log_prob(self, actions):
        return self.actor.get_actions_log_prob(actions)

    def act_inference(self, observations):
        vel, latent = self.estimator(observations)
        actor_input = torch.cat((observations[:,:self.num_one_step_obs], vel, latent), dim=-1)
        return self.actor.act_inference(actor_input)

    def evaluate(self, critic_observations, actions=None, **kwargs):
        return self.critic.evaluate(critic_observations, actions)

    def export_policy(self, path):
        self.export(path)

    def export(self, path):
        class PolicyExporterHIM(nn.Module):
            def __init__(self, actor, est,num_one_step_obs,num_actor_obs):
                super().__init__()
                self.actor = copy.deepcopy(actor)
                self.estimator = copy.deepcopy(est)
                self.num_one_step_obs = num_one_step_obs
                self.num_actor_obs = num_actor_obs

            def forward(self, obs_history):
                vel, z = self.estimator(obs_history)
                z = F.normalize(z, dim=-1, p=2.0)
                actor_input = torch.cat((obs_history[:, 0:self.num_one_step_obs], vel, z), dim=1)
                return self.actor.act_inference(actor_input)
            
            def export(self):
                os.makedirs(path, exist_ok=True)
                path_ts = os.path.join(path, 'policy.pt')  # TorchScript path
                path_onnx = os.path.join(path, 'policy.onnx')  # ONNX path
                self.to('cpu')
                # 使用TorchScript导出


                # 使用ONNX导出
                dummy_input = torch.rand(1, self.num_actor_obs)  # 请根据实际输入维度修改
                traced_script_module = torch.jit.trace(self,dummy_input)
                traced_script_module.save(path_ts)
                torch.onnx.export(traced_script_module, dummy_input, path_onnx, input_names=['obs_history'], output_names=['actions'])
        md = PolicyExporterHIM(self.actor,self.estimator,self.num_one_step_obs,self.num_actor_obs)
        md.export()
        
        # if self._normalize_obs:
        #     class NormalizedActor(nn.Module):
        #         def __init__(self, actor, obs_rms):
        #             super().__init__()
        #             self.actor = actor
        #             self.obs_rms = obs_rms
        #         def forward(self, obs):
        #             obs = self.obs_rms(obs)
        #             return self.actor(obs)
        #     model = NormalizedActor(copy.deepcopy(self.mean_NN), copy.deepcopy(self.obs_rms)).to('cpu')
        
        # else:
        #     model = copy.deepcopy(self.mean_NN).to('cpu')

        # dummy_input = torch.rand(self.mean_NN[0].in_features,)
        # model_traced = torch.jit.trace(model, dummy_input)
        # torch.jit.save(model_traced, path_TS)
        # torch.onnx.export(model_traced, dummy_input, path_onnx)
