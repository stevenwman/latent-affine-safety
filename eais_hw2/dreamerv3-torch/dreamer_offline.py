import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd
import collections

from tqdm import trange
from termcolor import cprint
import matplotlib.pyplot as plt
import gym
from io import BytesIO
from PIL import Image
import matplotlib.patches as patches
import io

to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

        self._make_pretrain_opt()

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

    def _make_pretrain_opt(self):
        config = self._config
        use_amp = True if config.precision == 16 else False
        if (
            config.pretrain_steps > 0
            or config.from_ckpt is not None
        ):
            # have separate lrs/eps/clips for actor and model
            # https://pytorch.org/docs/master/optim.html#per-parameter-options
            standard_kwargs = {
                "lr": config.model_lr,
                "eps": config.opt_eps,
                "clip": config.grad_clip,
                "wd": config.weight_decay,
                "opt": config.opt,
                "use_amp": use_amp,
            }
            model_params = {
                "params": list(self._wm.encoder.parameters())
                + list(self._wm.dynamics.parameters())
            }
            model_params["params"] += list(self._wm.heads["decoder"].parameters())
            actor_params = {
                "params": list(self._task_behavior.actor.parameters()),
                "lr": config.actor["lr"],
                "eps": config.actor["eps"],
                "clip": config.actor["grad_clip"],
            }
            self.pretrain_params = list(model_params["params"]) + list(
                actor_params["params"]
            )
            self.pretrain_opt = tools.Optimizer(
                "pretrain_opt", [model_params, actor_params], **standard_kwargs
            )
            self.actor_params = list(self._task_behavior.actor.parameters())
            
            print(
                f"Optimizer pretrain has {sum(param.numel() for param in self.pretrain_params)} variables."
            )

    def _update_running_metrics(self, metrics):
        for name, value in metrics.items():
            if name not in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

    def _maybe_log_metrics(self, video_pred_log=False):
        if self._logger is not None:
            logged = False
            if self._should_log(self._step):
                for name, values in self._metrics.items():
                    if not np.isnan(np.mean(values)):
                        self._logger.scalar(name, float(np.mean(values)))
                        self._metrics[name] = []
                logged = True

            if video_pred_log and self._should_log_video(self._step):
                video_pred, video_pred2 = self._wm.video_pred(next(self._dataset))
                self._logger.video("train_openl_agent", to_np(video_pred))
                self._logger.video("train_openl_hand", to_np(video_pred2))
                logged = True

            if logged:
                self._logger.write(fps=True)

    def pretrain_model_only(self, data, step=None):
        metrics = {}
        wm = self._wm
        actor = self._task_behavior.actor
        data = wm.preprocess(data)
        
        with tools.RequiresGrad(wm), tools.RequiresGrad(actor):
            with torch.cuda.amp.autocast(wm._use_amp):
                embed = wm.encoder(data)
                # post: z_t, prior: \hat{z}_t
                post, prior = wm.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                # note: kl_loss is already sum of dyn_loss and rep_loss
                kl_loss, kl_value, dyn_loss, rep_loss = wm.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape

                losses = {}
                feat = wm.dynamics.get_feat(post)

                if (
                    step <= self._config.pretrain_steps
                ):
                    # preds is dictionary of all all MLP+CNN keys
                    preds = wm.heads["decoder"](feat)
                    for name, pred in preds.items():
                        loss = -pred.log_prob(data[name])
                        assert loss.shape == embed.shape[:2], (name, loss.shape)
                        losses[name] = loss
                    recon_loss = sum(losses.values())
                else:
                    recon_loss = 0

                model_loss = kl_loss + recon_loss
                metrics = self.pretrain_opt(
                    torch.mean(model_loss), self.pretrain_params
                )
        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_loss"] = to_np(kl_loss)
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl_value"] = to_np(torch.mean(kl_value))

        with torch.cuda.amp.autocast(wm._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(wm.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(wm.dynamics.get_dist(post).entropy())
            )
        metrics = {
            f"model_only_pretrain/{k}": v for k, v in metrics.items()
        }  # Add prefix model_pretrain to all metrics
        self._update_running_metrics(metrics)
        self._maybe_log_metrics()
        self._step += 1
        self._logger.step = self._step
    def pretrain_regress_obs(self, data, obs_mlp, obs_opt, eval=False):
        wm = self._wm
        actor = self._task_behavior.actor
        data = wm.preprocess(data)
        if eval:
            obs_mlp.eval()
        with tools.RequiresGrad(obs_mlp):
            with torch.cuda.amp.autocast(wm._use_amp):
                embed = self._wm.encoder(data)
                post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])

                feat = self._wm.dynamics.get_feat(prior).detach() # want the imagined prior to be strong
                target = torch.Tensor(data["privileged_state"]).to(self._config.device)
                pred_state = obs_mlp(feat)
                obs_loss = torch.mean((pred_state - target) ** 2)
            if not eval:
                obs_opt(torch.mean(obs_loss), obs_mlp.parameters())
            else:
                obs_mlp.train()
        return obs_loss.item()
    def get_latent(self, xs, ys, thetas, imgs, lx_mlp):
        states = np.expand_dims(np.expand_dims(thetas,1),1)
        imgs = np.expand_dims(imgs, 1)
        dummy_acs = np.zeros((np.shape(xs)[0], 1, 3))
        rand_idx = 1 #np.random.randint(0, 3, np.shape(xs)[0])
        dummy_acs[np.arange(np.shape(xs)[0]), :, rand_idx] = 1
        firsts = np.ones((np.shape(xs)[0], 1))
        lasts = np.zeros((np.shape(xs)[0], 1))
        
        cos = np.cos(states)
        sin = np.sin(states)
        states = np.concatenate([cos, sin], axis=-1)
        data = {'obs_state': states, 'image': imgs, 'action': dummy_acs, 'is_first': firsts, 'is_terminal': lasts}

        data = self._wm.preprocess(data)
        embed = self._wm.encoder(data)

        post, prior = self._wm.dynamics.observe(
            embed, data["action"], data["is_first"]
            )
        feat = self._wm.dynamics.get_feat(post).detach()
        with torch.no_grad():  # Disable gradient calculation
            g_x = lx_mlp(feat).detach().cpu().numpy().squeeze()
        feat = self._wm.dynamics.get_feat(post).detach().cpu().numpy().squeeze()
        return g_x, feat, post
    def capture_image(self, state=None):
        """Captures an image of the current state of the environment."""
        # For simplicity, we create a blank image. In practice, this should render the environment.
        fig,ax = plt.subplots()
        plt.xlim([self._config.x_min,self._config.x_max])
        plt.ylim([self._config.y_min,self._config.y_max])
        plt.axis('off')
        fig.set_size_inches( 1, 1 )
        # Create the circle patch
        circle = patches.Circle([0,0], self._config.obs_r, edgecolor=(1,0,0), facecolor='none')
        # Add the circle patch to the axis
        dt = self._config.dt
        v = self._config.speed
        ax.add_patch(circle)
        
        plt.quiver(state[0], state[1], dt*v*np.cos(state[2]), dt*v*np.sin(state[2]), angles='xy', scale_units='xy', minlength=0,width=0.05, scale=0.2,color=(0,0,1), zorder=3)

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        #plt.savefig('logs/tests/test_rarl.png', dpi=dpi)
        plt.savefig(buf, format='png', dpi=self._config.size[0])
        buf.seek(0)

        # Load the buffer content as an RGB image
        img = Image.open(buf).convert('RGB')
        img_array = np.array(img)
        plt.close()
        return img_array
    def get_eval_plot(self, obs_mlp, theta):
        nx, ny, nz = 41, 41, 5

        v = np.zeros((nx, ny, nz))
        xs = np.linspace(-1, 1, nx)
        ys = np.linspace(-1, 1, ny)
        thetas= np.linspace(0, 2*np.pi, nz, endpoint=True)
        tn, tp, fn, fp = 0, 0, 0, 0
        it = np.nditer(v, flags=['multi_index'])
        ###
        idxs = []  
        imgs = []
        labels = []
        it = np.nditer(v, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            x = xs[idx[0]]
            y = ys[idx[1]]
            theta = thetas[idx[2]]
            if (x**2 + y**2) < (self._config.obs_r**2):
                labels.append(1) # unsafe
            else:
                labels.append(0) # safe
            x = x - np.cos(theta)*1*0.05
            y = y - np.sin(theta)*1*0.05
            imgs.append(self.capture_image(np.array([x, y, theta])))
            idxs.append(idx)        
            it.iternext()
        idxs = np.array(idxs)
        safe_idxs = np.where(np.array(labels) == 0)
        unsafe_idxs = np.where(np.array(labels) == 1)
        x_lin = xs[idxs[:,0]]
        y_lin = ys[idxs[:,1]]
        theta_lin = thetas[idxs[:,2]]
        
        g_x = []
        ## all of this is because I can't do a forward pass with 128x128 images in one go
        num_c = 5
        chunk = int(np.shape(x_lin)[0]/num_c)
        for k in range(num_c):
            g_xlist, _, _ = self.get_latent(x_lin[k*chunk:(k+1)*chunk], y_lin[k*chunk:(k+1)*chunk], theta_lin[k*chunk:(k+1)*chunk], imgs[k*chunk:(k+1)*chunk], obs_mlp)
            g_x = g_x + g_xlist.tolist()
        g_x = np.array(g_x)
        v[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = g_x

        #g_x, _, _ = self.get_latent(x_lin, y_lin, theta_lin, imgs, obs_mlp)
        #v[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = g_x
        tp  = np.where(g_x[safe_idxs] > 0)
        fn  = np.where(g_x[safe_idxs] <= 0)
        fp  = np.where(g_x[unsafe_idxs] > 0)
        tn  = np.where(g_x[unsafe_idxs] <= 0)
        
        vmax = round(max(np.max(v), 0),1)
        vmin = round(min(np.min(v), -vmax),1)
        
        fig, axes = plt.subplots(nz, 2, figsize=(12, nz*6))
        
        for i in range(nz):
            ax = axes[i, 0]
            im = ax.imshow(
                v[:, :, i].T, interpolation='none', extent=np.array([
                self._config.x_min, self._config.x_max, self._config.y_min, self._config.y_max, ]), origin="lower",
                cmap="seismic", vmin=vmin, vmax=vmax, zorder=-1
            )
            cbar = fig.colorbar(
                im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
            )
            cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
            ax.set_title(r'$g(x)$', fontsize=18)

            ax = axes[i, 1]
            im = ax.imshow(
                v[:, :, i].T > 0, interpolation='none', extent=np.array([
                self._config.x_min, self._config.x_max, self._config.y_min, self._config.y_max, ]), origin="lower",
                cmap="seismic", vmin=-1, vmax=1, zorder=-1
            )
            cbar = fig.colorbar(
                im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
            )
            cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
            ax.set_title(r'$v(x)$', fontsize=18)
            fig.tight_layout()
            circle = plt.Circle((0, 0), self._config.obs_r, fill=False, color='blue', label = 'GT boundary')

            # Add the circle to the plot
            axes[i,0].add_patch(circle)
            axes[i,0].set_aspect('equal')
            circle2 = plt.Circle((0, 0), self._config.obs_r, fill=False, color='blue', label = 'GT boundary')

            axes[i,1].add_patch(circle2)
            axes[i,1].set_aspect('equal')

        fp_g = np.shape(fp)[1]
        fn_g = np.shape(fn)[1]
        tp_g = np.shape(tp)[1]
        tn_g = np.shape(tn)[1]
        tot = fp_g + fn_g + tp_g + tn_g
        fig.suptitle(r"$TP={:.0f}\%$ ".format(tp_g/tot * 100) + r"$TN={:.0f}\%$ ".format(tn_g/tot * 100) + r"$FP={:.0f}\%$ ".format(fp_g/tot * 100) +r"$FN={:.0f}\%$".format(fn_g/tot * 100),
            fontsize=10,)
        buf = BytesIO()

        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        plot = Image.open(buf).convert("RGB")
        return np.array(plot), tp, fn, fp, tn
    def train_lx(self, data, lx_mlp, lx_opt, eval=False):
        wm = self._wm
        wm.dynamics.sample = False
        actor = self._task_behavior.actor
        data = wm.preprocess(data)
        R = self._config.obs_r
        with tools.RequiresGrad(lx_mlp):
            if not eval:
                with torch.cuda.amp.autocast(wm._use_amp):
                    embed = self._wm.encoder(data)
                    post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])
                    feat = self._wm.dynamics.get_feat(post).detach() 
                    
                    x, y, theta = data["privileged_state"][:,:,0], data["privileged_state"][:,:,1], data["privileged_state"][:,:, 2]

                    safety_data = (x**2 + y**2) - R**2
                    safe_data = torch.where(safety_data > 0)
                    unsafe_data = torch.where(safety_data <= 0)

                    safe_dataset = feat[safe_data]
                    unsafe_dataset = feat[unsafe_data]

                    pos = lx_mlp(safe_dataset)
                    neg = lx_mlp(unsafe_dataset)
                    
                    
                    gamma = self._config.gamma_lx
                    lx_loss = (1/pos.size(0))*torch.sum(torch.relu(gamma - pos)) #penalizes safe for being negative
                    lx_loss +=  (1/neg.size(0))*torch.sum(torch.relu(gamma + neg)) # penalizes unsafe for being positive
                    
                    lx_loss = lx_loss
            
                    lx_opt(torch.mean(lx_loss), lx_mlp.parameters())
                    plot_arr = None
                    score = 0
            else:
                lx_mlp.eval()
                plot_arr, tp, fn, fp, tn = self.get_eval_plot(lx_mlp, 0)
               
                lx_mlp.train()
                fp_num = np.shape(fp)[1]
                fn_num = np.shape(fn)[1]
                tp_num = np.shape(tp)[1]
                tn_num = np.shape(tn)[1]
                print('TP: ', tp_num)
                print('FN: ', fn_num)

                print('TN: ', tn_num)
                print('FP: ', fp_num)
            
                score = (fp_num + fn_num) / (fp_num + fn_num + tp_num + tn_num)

        return score, plot_arr
    


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    
    action_space = gym.spaces.Discrete(3)

    bounds = np.array([[config.x_min, config.x_max], [config.y_min, config.y_max], [0, 2 * np.pi]])
    low = bounds[:, 0]
    high = bounds[:, 1]
    midpoint = (low + high) / 2.0
    interval = high - low
    gt_observation_space = gym.spaces.Box(
        np.float32(midpoint - interval/2),
        np.float32(midpoint + interval/2),
    )
    image_size = config.size[0] #128
    image_observation_space = gym.spaces.Box(
        low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
    )

    
    obs_observation_space = gym.spaces.Box(
        low=-1, high=1, shape=(2,), dtype=np.float32
    )
    observation_space = gym.spaces.Dict({
            'state': gt_observation_space,
            'obs_state': obs_observation_space,
            'image': image_observation_space
        })


    print("Action Space", action_space)
    config.num_actions = action_space.n if hasattr(action_space, "n") else action_space.shape[0]

    
    expert_eps = collections.OrderedDict()
    print(expert_eps)
    tools.fill_expert_dataset_dubins(config, expert_eps)
    expert_dataset = make_dataset(expert_eps, config)
    # validation replay buffer
    expert_val_eps = collections.OrderedDict()
    tools.fill_expert_dataset_dubins(config, expert_val_eps, is_val_set=True)
    eval_dataset = make_dataset(expert_eps, config)

    print(len(expert_eps))
    print(len(expert_val_eps))

    print("Simulate agent.")
    agent = Dreamer(
        observation_space,
        action_space,
        config,
        logger,
        expert_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    def log_plot(title, data):
        buf = BytesIO()
        plt.plot(np.arange(len(data)), data)
        plt.title(title)
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        plot = Image.open(buf).convert("RGB")
        plot_arr = np.array(plot)
        logger.image("pretrain/" + title, np.transpose(plot_arr, (2, 0, 1)))
    def eval_obs_recon():
        recon_steps = 101
        obs_mlp, obs_opt = agent._wm._init_obs_mlp(config, 3)
        train_loss = []
        eval_loss = []
        for i in range(recon_steps):
            if i % int(recon_steps/4) == 0:
                new_loss = agent.pretrain_regress_obs(
                    next(eval_dataset), obs_mlp, obs_opt, eval=True
                )
                eval_loss.append(new_loss)
            else:
                new_loss = agent.pretrain_regress_obs(
                    next(expert_dataset), obs_mlp, obs_opt
                )
                train_loss.append(new_loss)
        log_plot("train_recon_loss", train_loss)
        log_plot("eval_recon_loss", eval_loss)
        logger.scalar("pretrain/train_recon_loss_min", np.min(train_loss))
        logger.scalar("pretrain/eval_recon_loss_min", np.min(eval_loss))
        logger.write(step=logger.step)
        del obs_mlp, obs_opt  # dont need to keep these
        return np.min(eval_loss)
    def train_lx(ckpt_name, log_dir):
        recon_steps = 2501
        best_pretrain_success_classifier = float("inf")
        lx_mlp, lx_opt = agent._wm._init_lx_mlp(config, 1)
        train_loss = []
        eval_loss = []
        for i in range(recon_steps):
            if i % 250 == 0:
                print('eval')
                new_loss, eval_plot = agent.train_lx(
                    next(eval_dataset), lx_mlp, lx_opt, eval=True
                )
                eval_loss.append(new_loss)
                logger.image("classifier", np.transpose(eval_plot, (2, 0, 1)))
                logger.write(step=i+100000)
                best_pretrain_success_classifier = tools.save_checkpoint(
                    ckpt_name, i, new_loss, best_pretrain_success_classifier, lx_mlp, logdir
                )

            else:
                new_loss, _ = agent.train_lx(
                    next(expert_dataset), lx_mlp, lx_opt
                )
                train_loss.append(new_loss)
        log_plot("train_lx_loss", train_loss)
        log_plot("eval_lx_loss", eval_loss)
        logger.scalar("pretrain/train_lx_loss_min", np.min(train_loss))
        logger.scalar("pretrain/eval_lx_loss_min", np.min(eval_loss))
        logger.write(step=i)
        print(eval_loss)
        print('logged')
        return lx_mlp, lx_opt
    def evaluate(other_dataset=None, eval_prefix=""):
        agent.eval()
        
        eval_policy = functools.partial(agent, training=False)

        # For Logging (1 episode)
        if config.video_pred_log:
            video_pred = agent._wm.video_pred(next(eval_dataset))
            logger.video("eval_recon/openl_agent", to_np(video_pred))

            if other_dataset:
                video_pred = agent._wm.video_pred(next(other_dataset))
                logger.video("train_recon/openl_agent", to_np(video_pred))

        
        logger.write(step=logger.step)
        recon_eval = eval_obs_recon()  # testing observation reconstruction

        agent.train()
        return recon_eval, recon_eval
    # ==================== Actor Pretrain ====================
    total_pretrain_steps = config.pretrain_steps 
    print(total_pretrain_steps)
    if total_pretrain_steps > 0:
        
        cprint(
            f"Pretraining for {config.pretrain_steps=}",
            color="cyan",
            attrs=["bold"],
        )
        ckpt_name = (  # noqa: E731
            lambda step: "pretrain_joint"
            if step < config.pretrain_steps
            else "pretrain_actor"
        )
        best_pretrain_success = float("inf")
        best_pretrain_success = float("inf")
        for step in trange(
            total_pretrain_steps,
            desc="Encoder + Actor pretraining",
            ncols=0,
            leave=False,
        ):
            if (
                ((step + 1) % config.eval_every) == 0
                or step == 1
                # and step > 0
            ):
               
                print('eval')
                score, success = evaluate(
                    other_dataset=expert_dataset, eval_prefix="pretrain"
                )
                best_pretrain_success = tools.save_checkpoint(
                    ckpt_name, step, success, best_pretrain_success, agent, logdir
                )

    
            exp_data = next(expert_dataset)

            agent.pretrain_model_only(exp_data, step)
    
    print(logdir)
    print(logdir / "pretrain_joint.pt")
    checkpoint = torch.load(logdir / "pretrain_joint.pt")
    agent.load_state_dict(checkpoint["agent_state_dict"])
    print('training l(x)')
    lx_mlp, lx_opt = train_lx('classifier', logdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()

    yaml = yaml.YAML(typ="safe", pure=True)
    configs = yaml.load(
        (pathlib.Path(sys.argv[0]).parent / "../configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
