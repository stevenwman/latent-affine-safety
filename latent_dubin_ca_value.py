import sys

sys.path.append("PytorchReachability")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from PIL import Image
import numpy as np
import torch
from collections import defaultdict
import ruamel.yaml as yaml
from PyHJ.data import Batch
from PyHJ.exploration import GaussianNoise
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
from PyHJ.policy import avoid_DDPGPolicy_annealing as DDPGPolicy
import pathlib
import argparse
from pathlib import Path

import gym

# replace these accordingly
parent_dir = "."
dreamer_dir = str(Path(__file__).resolve().parent) + "/eais_hw2/dreamerv3-torch"
ckpt_path = str(Path(__file__).resolve().parent) + "/eais_hw2/best_pretrain_joint_0_12.pt"
policy_path = str(Path(__file__).resolve().parent) + "/eais_hw2/policy.pth"
# policy_path = ("dreamer_l2_lessrand/lcrl/0411/142538/lcrl/dubins-wm/"
#                 "wm_actor_activation_ReLU_critic_activation_ReLU_game_"
#                 "gd_steps_1_tau_0.005_training_num_1_buffer_size_40000"
#                 "_c_net_512_4_a1_512_4_a2_512_4_gamma_0.95/noise_0.1_"
#                 "actor_lr_0.0001_critic_lr_0.001_batch_512_step_per_"
#                 "epoch_40000_kwargs_{}_seed_0/epoch_id_40/policy.pth")
HJconfig_path = str(Path(__file__).resolve().parent) + "/eais_hw2/HJconfig.yaml"

sys.path.append(parent_dir)
sys.path.append(dreamer_dir)
# print(sys.path)
import models
import tools


def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value

def get_args():
    yml = yaml.YAML(typ="safe", pure=True)
    configs = yml.load((pathlib.Path(sys.argv[0]).parent.parent.parent.parent.parent / HJconfig_path).read_text())
    name_list = ["defaults"]
    defaults = {}

    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    final_config = parser.parse_args([])
    return final_config


class latent_dubin_ca_value:
    """
    Class to calculate the latent Dubin Continuous Action value function.
    """

    def __init__(self):
        """
        Init function
        """
        self.param = {
            'x_min': -1.1,
            'x_max': 1.1,
            'y_min': -1.1,
            'y_max': 1.1,
            'u_max': 1.25,
            'radius': 0.5,
            'dt': 0.05,
            'v': 1.,
            'dpi': 128
            }
        self.np_expdim = lambda x: np.expand_dims(x, axis=0)
        self._init_wm()

    def dyn_step_back(self, s0: torch.Tensor) -> torch.Tensor:
        """
        Take a single timestep backward. Because the WM requires 
        a single forward step to compute latent and features.
        """
        v = self.param['v']
        dt = self.param['dt']
        bs = s0.shape[0]
        s, s_prev = torch.zeros(bs, 3), torch.zeros(bs, 3)
        s0 = torch.tensor(s0, dtype=torch.float32)
        s[:,0], s[:,1], s[:,2] = s0[:,0], s0[:,1], s0[:,2]
        s_prev[:,0] = s[:,0] - v*dt*torch.cos(s[:,2])
        s_prev[:,1] = s[:,1] - v*dt*torch.sin(s[:,2])
        s_prev[:,2] = s[:,2]
        return s_prev
    
    def state_to_data(self, s0: torch.Tensor) -> dict:
        """
        Convert the state to data for the latent Dubin environment.
        """
        state_obs, img_obs, state_gt, dones, acs = ([] for _ in range(5))
        
        for i in range(s0.shape[0]):
            # print(i, " ", s0.shape[0])
            s = s0[i]
            ac = 0 * torch.rand(1)
            state_obs.append(s[2].numpy()) # get to observe theta
            state_gt.append(s.numpy()) # gt state
            dones.append(1)
            acs.append(ac)

            dt = self.param['dt']
            v = self.param['v']
            center = (0.0, 0.0)

            fig,ax = plt.subplots()
            # hardcoded with existing limits
            plt.xlim([-1.1, 1.1]) 
            plt.ylim([-1.1, 1.1])
            plt.axis('off')
            fig.set_size_inches( 1, 1 )
            # Create the circle patch
            circle = patches.Circle(center, self.param['radius'], 
                                    edgecolor=(1,0,0), facecolor='none')
            ax.add_patch(circle)

            plt.quiver(s[0], s[1], dt*v*torch.cos(s[2]), dt*v*torch.sin(s[2]),
                       angles='xy', scale_units='xy', minlength=0,width=0.1, 
                       scale=0.18,color=(0,0,1), zorder=3)
            plt.scatter(s[0], s[1],s=20, color=(0,0,1), zorder=3)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=self.param['dpi'])
            buf.seek(0)

            # Load the buffer content as an RGB image
            img = Image.open(buf).convert('RGB')
            img_array = np.array(img)
            img_obs.append(img_array)
            plt.close()

        demo = {}
        demo['obs'] = {'image': img_obs, 'state': state_obs, 'priv_state': state_gt}
        demo['actions'] = acs
        demo['dones'] = dones

        return demo
    
    def demo_to_traj(self, demos: dict) -> dict:       
        """
        Convert demo dictionary to trajectory ready to be pre-processed
        """
        pixel_keys = sorted(['image'])
        state_keys = sorted(['state'])

        traj = demos
        traj_to_pp = {}

        for t in range(len(traj["obs"][pixel_keys[0]])):
            transition = defaultdict(np.array)
            for obs_key in pixel_keys:
                transition[obs_key] = traj["obs"][obs_key][t]

            if len(state_keys) != 0:
                curr_obs_state_vec = [traj["obs"][obs_key][t] for obs_key in state_keys]
                transition["state"] = curr_obs_state_vec
                
            transition["privileged_state"] = traj['obs']['priv_state'][t]
            transition["obs_state"] = [np.cos(traj['obs']['state'][t]), 
                                       np.sin(traj['obs']['state'][t])]
            transition["reward"] = np.array(0, dtype=np.float32)
            transition["is_first"] = np.array(t == 0, dtype=np.bool_)
            transition["is_last"] = np.array(traj["dones"][t], dtype=np.bool_)
            transition["is_terminal"] = np.array(traj["dones"][t], dtype=np.bool_)
            transition["discount"] = np.array(1, dtype=np.float32)
            transition["action"] = np.array(traj["actions"][t], dtype=np.float32)

            if t == 0: 
                traj_to_pp = {k:self.np_expdim(self.np_expdim(v)) for k,v in transition.items()}
            else: 
                for k,v in traj_to_pp.items():
                    traj_to_pp[k] = np.append(v, self.np_expdim(self.np_expdim(transition[k])), axis=0)
        
        return traj_to_pp
    
    def find_a(self, state):
        """
        Use the policy to find the safe action for a given state.
        """
        tmp_obs = np.array(state).reshape(state.shape[0],state.shape[-1])
        tmp_batch = Batch(obs = tmp_obs, info = Batch())
        tmp = self.policy(tmp_batch, model = "actor_old").act
        act = self.policy.map_action(tmp).cpu().detach().numpy().flatten()
        return act

    def evaluate_V(self, state):
        """
        Use the safe value function to find the value for a given state.
        """
        tmp_obs = np.array(state).reshape(state.shape[0],state.shape[-1])
        tmp_batch = Batch(obs = tmp_obs, info = Batch())
        tmp = self.policy.critic(tmp_batch.obs, self.policy(tmp_batch, model="actor_old").act)
        return tmp.cpu().detach().numpy().flatten()
    
    def _init_wm(self):
        """
        Initialize world model and value function.
        """
        args=get_args()
        config=args

        # set up the environment spaces
        image_size = config.size[0] #128
        img_obs_space = gym.spaces.Box(low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8)
        obs_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        # hardcoded env bounds
        high = np.array([1.1, 1.1, 2*np.pi,])
        low = np.array([-1.1, -1.1, 0.,])
        gt_observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        observation_space = gym.spaces.Dict({'obs_state': obs_space,
                                            'image': img_obs_space,
                                            'state': gt_observation_space,})
        u_max = 1.25
        action_space = gym.spaces.Box(low=-u_max, high=u_max, shape=(1,), dtype=np.float32)
        config.num_actions = action_space.n if hasattr(action_space, "n") else action_space.shape[0]

        # load 
        config.eval_state_mean = True
        wm = models.WorldModel(observation_space, action_space, 0, config)
        wm.to(config.device)
        checkpoint = torch.load(ckpt_path)
        state_dict = {k[14:]:v for k,v in checkpoint['agent_state_dict'].items() if '_wm' in k}
        wm.load_state_dict(state_dict)
        
        # hardcoded latent dim
        args.state_shape = (1,1,544,)
        args.action_shape = args.action1_shape = (1,)
        args.max_action = args.max_action1 = u_max

        # seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        activation_map = {
            'ReLU': torch.nn.ReLU,
            'Tanh': torch.nn.Tanh,
            'Sigmoid': torch.nn.Sigmoid,
            'SiLU': torch.nn.SiLU
        }
        actor_activation = activation_map.get(args.actor_activation)
        critic_activation = activation_map.get(args.critic_activation)

        assert args.critic_net is not None, "Please provide critic_net!"
        critic_net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.critic_net,
            activation=critic_activation,
            concat=True,
            device=args.device
            )

        critic = Critic(critic_net, device=args.device).to(args.device)
        critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

        print("DDPG under the Avoid annealed Bellman equation with no Disturbance has been loaded!")

        actor1_net = Net(args.state_shape, hidden_sizes=args.control_net, activation=actor_activation, device=args.device)
        actor1 = Actor(actor1_net, args.action1_shape, max_action=args.max_action1, device=args.device).to(args.device)
        actor1_optim = torch.optim.Adam(actor1.parameters(), lr=args.actor_lr)

        policy = DDPGPolicy(
            critic,
            critic_optim,
            tau=args.tau,
            gamma=args.gamma_lcrl,
            exploration_noise=GaussianNoise(sigma=args.exploration_noise),
            reward_normalization=args.rew_norm,
            estimation_step=args.n_step,
            action_space=action_space,
            actor1=actor1,
            actor1_optim=actor1_optim,
            actor_gradient_steps=args.actor_gradient_steps,
            )

        # load policy
        policy.load_state_dict(torch.load(policy_path))

        self.wm = wm
        self.policy = policy

    def state_action_to_V(self, state, action):
        data_pts = self.state_to_data(state)
        traj = self.demo_to_traj(data_pts)

        bs = traj['state'].shape[0] # batch size
        is_first = torch.ones((bs,1), device='cuda:0')

        proc_data = self.wm.preprocess(traj)
        latent,_ = self.wm.dynamics.observe(self.wm.encoder(proc_data), action, is_first)
        # latent['stoch'] = latent['mean']
        for k, v in latent.items(): latent[k] = v[:, [-1]]
        feat = self.wm.dynamics.get_feat(latent).detach().cpu().numpy() 
        value = self.evaluate_V(feat)
        act = self.find_a(feat)
        pr_state = proc_data['privileged_state'][0,0].cpu()

        return value

    def state_to_V(self, state):
        s_curr = state
        s_prev = self.dyn_step_back(s_curr)
        data_pts = self.state_to_data(s_prev)
        traj = self.demo_to_traj(data_pts)

        bs = traj['state'].shape[0] # batch size
        action = torch.zeros((bs,1,1), device='cuda:0')
        is_first = torch.ones((bs,1), device='cuda:0')

        proc_data = self.wm.preprocess(traj)
        latent,_ = self.wm.dynamics.observe(self.wm.encoder(proc_data), action, is_first)
        latent['stoch'] = latent['mean']
        for k, v in latent.items(): latent[k] = v[:, [-1]]
        feat = self.wm.dynamics.get_feat(latent).detach().cpu().numpy() 
        value = self.evaluate_V(feat)
        act = self.find_a(feat)
        pr_state = proc_data['privileged_state'][0,0].cpu()
        # print("privileged state: ", pr_state)
        # print("safe action: ", act)
        # print("safe value: ", value)    

        return value

def main():
    s_curr = np.array([[0., 0., 0.],[0.1, 0., 0.],[0.,0.,0.]])
    val_fn = latent_dubin_ca_value()
    val_fn.state_to_V(s_curr)


if __name__ == "__main__":
    main()