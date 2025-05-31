
learning/runners/on_policy_runner.py 中的初始化提到actor_obs是从train_cfg["policy"]["actor_obs"]中得到的。

电机的执行action通过 `actions = self.alg.act(actor_obs, critic_obs)`得到,执行是通过 `set_actions`

`actor_obs` 的得到是通过 `get_noisy_obs -> get_obs`

```python
class OnPolicyRunner:
    def __init__(self,env: VecEnv,train_cfg,log_dir=None,device='cpu'):
	...
        self.policy_cfg = train_cfg["policy"]
	...
	self.num_actor_obs = self.get_obs_size(self.policy_cfg["actor_obs"])
        self.num_critic_obs = self.get_obs_size(self.policy_cfg["critic_obs"])
        self.num_actions = self.get_action_size(self.policy_cfg["actions"])
	...
    def learn(self, num_learning_iterations=None, init_at_random_ep_len=False):
        ...
        actor_obs = self.get_noisy_obs(self.policy_cfg["actor_obs"])
        critic_obs = self.get_obs(self.policy_cfg["critic_obs"])
	...
        self.num_learning_iterations = num_learning_iterations
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            ...
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(actor_obs, critic_obs)
                    self.set_actions(actions)
                    self.env.step()
		    ...
                    actor_obs = self.get_noisy_obs(self.policy_cfg["actor_obs"])
                    critic_obs = self.get_obs(self.policy_cfg["critic_obs"])
		    ...
    def set_actions(self, actions):
        if hasattr(self.env.cfg.scaling, "clip_actions"):
            actions = torch.clip(actions, -self.env.cfg.scaling.clip_actions, self.env.cfg.scaling.clip_actions)
        self.env.set_states(self.policy_cfg["actions"], actions)

    def get_obs(self, obs_list):
        self.env._set_obs_variables()
        observation = self.env.get_states(obs_list).to(self.device)
        return observation
  
    def get_noisy_obs(self, obs_list):
        observation = self.get_obs(obs_list)
        return observation + (2*torch.rand_like(observation) - 1) * self.obs_noise_vec
```

追溯 OnPolicyRunner类 的初始化入参train_cfg

gym/utils/task_registry.py 在`TaskRegistry`类的`make_alg_runner`函数当中动态地实例化一个 `OnPolicyRunner` 类的对象

```python
class TaskRegistry():
    ...
    def get_cfgs(self, args) -> Tuple[LeggedRobotCfg, LeggedRobotRunnerCfg]:
        if args.load_files:
            # Update task_classes, env_cfgs, train_cfgs to original files
            self.set_registry_to_original_files(args)

        name = args.task
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[Union[OnPolicyRunner], LeggedRobotRunnerCfg]:
        ...
	if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(args)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
	...
        train_cfg_dict = class_to_dict(train_cfg)
        runner: Union[OnPolicyRunner] = eval(train_cfg_dict["runner_class_name"])(env, train_cfg_dict, log_dir, device=args.rl_device)
	...
	return runner, train_cfg
```

追溯 `OnPolicyRunner`类 的 `set_actions` 函数中的 `self.env.set_states(self.policy_cfg["actions"], actions)`

gym/envs/base/base_task.py

```python
class BaseTask:
    ...
    def set_states(self, state_list, values):
        idx = 0
        for state in state_list:
            state_dim = getattr(self, state).shape[1]
            self.set_state(state, values[:, idx:idx+state_dim])
            idx += state_dim
        assert(idx == values.shape[1]), "Actions don't equal tensor shapes"
    def set_state(self, name, value):
        try:
            if name in self.scales.keys():
                setattr(self, name, value/self.scales[name])
            else:
                setattr(self, name, value)
        except AttributeError:
            print("Value for " + name + " does not match tensor shape")

```

追溯 `OnPolicyRunner`类 的 `get_obs` 函数中的 `self.env._set_obs_variables()` 和 `self.env.get_states(obs_list).to(self.device)`

gym/envs/hi01/hi01_controller.py

```python
class Hi01Controller(LeggedRobot):
    cfg: Hi01ControllerCfg
    ...
    def _set_obs_variables(self):
        self.foot_states_right[:, :3] = ...
        self.foot_states_left[:, :3] = ...
        self.foot_states_right[:, 3] = ...
        self.foot_states_left[:, 3] = ...

        self.step_commands_right[:, :3] = ...
        self.step_commands_left[:, :3] = ...
        self.step_commands_right[:, 3] = ...
        self.step_commands_left[:, 3] = ...

        self.phase_sin = ...
        self.phase_cos = ...

        self.base_lin_vel_world = ...

```

gym/envs/base/base_task.py 

```python
class BaseTask:
    def get_states(self, obs_list):
        return torch.cat([self.get_state(obs) for obs in obs_list], dim=-1)

    def get_state(self, name):
        if name in self.scales.keys():
            return getattr(self, name)*self.scales[name]
        else:
            return getattr(self, name)

    def _parse_cfg(self):
        ...
        self.scales = class_to_dict(self.cfg.scaling)
        ...
```

追溯 self.cfg.scaling

gym/envs/hi01/hi01_controller_config.py

```python
class Hi01ControllerCfg(LeggedRobotCfg):
    ...
    class scaling(LeggedRobotCfg.scaling):
        base_height = 1.
        base_lin_vel = 1. #.5
        base_ang_vel = 1. #2.
        projected_gravity = 1.
        foot_states_right = 1.
        foot_states_left = 1.
        dof_pos = 1.
        dof_vel = 1. #.1
        dof_pos_target = dof_pos  # scale by range of motion

        # Action scales
        commands = 1.
        clip_actions = 10.
```
