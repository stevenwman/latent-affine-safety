"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

This experiment runs double deep Q-network with the discounted reach-avoid
Bellman equation (DRABE) proposed in [RSS21] on a 3-dimensional Dubins car
problem. We use this script to generate Fig. 5 in the paper.

Examples:
    RA: python3 sim_car_one.py -sf -of scratch -w -wi 5000 -g 0.9999 -n 9999
    test: python3 sim_car_one.py -sf -of scratch -w -wi 50 -mu 1000 -cp 400
        -n tmp
"""

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer_dir)
print(sys.path)
import argparse
import time
from warnings import simplefilter
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import ruamel.yaml as yaml

from RARL.DDQNSingleAvoid import DDQNSingleAvoid as DDQNSingle
from RARL.config import dqnConfig
from RARL.utils import save_obj
from gym_reachability import gym_reachability  # Custom Gym env.
import models 
import tools

import argparse
import collections
import copy
import warnings
import functools
import time
import pathlib
import sys
from datetime import datetime
from pathlib import Path
from termcolor import cprint


matplotlib.use('Agg')
simplefilter(action='ignore', category=FutureWarning)
timestr = time.strftime("%Y-%m-%d-%H_%M")

def RARL(config):

  # == CONFIGURATION ==
  env_name = "dubins_car_latent_avoid-v1"
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  maxUpdates = config.maxUpdates
  updateTimes = config.updateTimes
  updatePeriod = int(maxUpdates / updateTimes)
  updatePeriodHalf = int(updatePeriod / 2)
  maxSteps = config.numT

  # == Environment ==
  print("\n== Environment Information ==")
  if config.doneType == 'toEnd':
    sample_inside_obs = True
  elif config.doneType == 'TF' or config.doneType == 'fail':
    sample_inside_obs = False

  print(env_name)
  print(gym_reachability)
  env = gym.make(
      env_name, config=config, device=device, mode=config.mode, doneType=config.doneType,
      sample_inside_obs=sample_inside_obs
  )

  fn = config.name + '-' + config.doneType
  if config.showTime:
    fn = fn + '-' + timestr

  wm = models.WorldModel(env.observation_space, env.action_space, 0, config)
  checkpoint = torch.load('best_pretrain_joint_0_10.pt')
  wm.dynamics.sample = False

  state_dict = {k[14:]:v for k,v in checkpoint['agent_state_dict'].items() if '_wm' in k}
  wm.load_state_dict(state_dict)
  lx_mlp, _ = wm._init_lx_mlp(config, 1)
  lx_ckpt = torch.load('best_classifier_0_01.pt')
  lx_mlp.load_state_dict(lx_ckpt['agent_state_dict'])
  env.car.set_wm(wm, lx_mlp, config)

  outFolder = os.path.join(config.outFolder, 'car-DDQN', fn)
  print(outFolder)
  figureFolder = os.path.join(outFolder, 'figure')
  os.makedirs(figureFolder, exist_ok=True)


  stateDim = env.state.shape[0]
  if config.wm:
    if config.dyn_discrete:
      stateDim = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    else:
      stateDim = config.dyn_stoch + config.dyn_deter
  actionNum = env.action_space.n
  actionList = np.arange(actionNum)
  print(
      "State Dimension: {:d}, ActionSpace Dimension: {:d}".format(
          stateDim, actionNum
      )
  )

  # == Setting in this Environment ==
  env.set_speed(speed=config.speed)
  env.set_constraint(radius=config.obs_r)
  env.set_radius_rotation(R_turn=config.speed/config.turnRate)
  print("Dynamic parameters:")
  print("  CAR", end='\n    ')
  print(
      "Constraint: {:.1f} ".format(env.car.constraint_radius)
      + "Turn: {:.2f} ".format(env.car.R_turn)
      + "Max speed: {:.2f} ".format(env.car.speed)
      + "Max angular speed: {:.3f}".format(env.car.max_turning_rate)
  )
  print("  ENV", end='\n    ')
  print(
      "Constraint: {:.1f} ".format(env.constraint_radius)
      + "Turn: {:.2f} ".format(env.R_turn)
      + "Max speed: {:.2f} ".format(env.speed)
  )
  print(env.car.discrete_controls)
  env.set_seed(config.randomSeed)
  
  # == Get and Plot max{l_x, g_x} ==
  if config.plotFigure or config.storeFigure:
    nx, ny = 51, 51
    
    v = np.zeros((nx, ny))
    g_x = np.zeros((nx, ny))
    xs = np.linspace(env.bounds[0, 0], env.bounds[0, 1], nx)
    ys = np.linspace(env.bounds[1, 0], env.bounds[1, 1], ny)

    it = np.nditer(v, flags=['multi_index'])
    ###
    idxs = []  
    imgs = []
    thetas = []
    it = np.nditer(v, flags=["multi_index"])
    while not it.finished:
      idx = it.multi_index
      x = xs[idx[0]]
      y = ys[idx[1]]
      theta = np.random.random()*2*np.pi
      assert theta > 0 and theta < 2*np.pi
      thetas.append(theta)
      if env.car.use_wm:
        imgs.append(env.capture_image(np.array([x, y, theta])))
        idxs.append(idx)        
      it.iternext()
    idxs = np.array(idxs)
    x_lin = xs[idxs[:,0]]
    y_lin = ys[idxs[:,1]]
    theta_lin = np.array(thetas)
    
      
    g_x, _, _ = env.car.get_latent(x_lin, y_lin, theta_lin, imgs)


###
    v[idxs[:, 0], idxs[:, 1]] = g_x
    g_x = v

    vmax = round(max(np.max(g_x), 0),1)
    vmin = round(min(np.min(g_x), -vmax),1)
    axStyle = env.get_axes()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    ax = axes[0]
    im = ax.imshow(
        g_x.T, interpolation='none', extent=axStyle[0], origin="lower",
        cmap="seismic", vmin=vmin, vmax=vmax, zorder=-1
    )
    cbar = fig.colorbar(
        im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
    )
    cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
    ax.set_title(r'$g(x)$', fontsize=18)

    ax = axes[1]
    im = ax.imshow(
        v.T > 0, interpolation='none', extent=axStyle[0], origin="lower",
        cmap="seismic", vmin=-1, vmax=1, zorder=-1
    )
    cbar = fig.colorbar(
        im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
    )
    cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
    ax.set_title(r'$v(x)$', fontsize=18)

    for ax in axes:
      env.plot_target_failure_set(ax=ax)
      env.plot_formatting(ax=ax)

    fig.tight_layout()
    if config.storeFigure:
      figurePath = os.path.join(figureFolder, 'env.png')
      fig.savefig(figurePath)
    if config.plotFigure:
      plt.show()
      plt.pause(0.001)
    plt.close()
  
  # == Agent CONFIG ==
  print("\n== Agent Information ==")
  if config.annealing:
    GAMMA_END = 0.9999
    EPS_PERIOD = int(updatePeriod / 10)
    EPS_RESET_PERIOD = updatePeriod
  else:
    GAMMA_END = config.gamma
    EPS_PERIOD = updatePeriod
    EPS_RESET_PERIOD = maxUpdates

  CONFIG = dqnConfig(
      DEVICE=device, ENV_NAME=env_name, SEED=config.randomSeed,
      MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps, BATCH_SIZE=64,
      MEMORY_CAPACITY=config.memoryCapacity, ARCHITECTURE=config.architecture,
      ACTIVATION=config.actType, GAMMA=config.gamma, GAMMA_PERIOD=updatePeriod,
      GAMMA_END=GAMMA_END, EPS_PERIOD=EPS_PERIOD, EPS_DECAY=0.7,
      EPS_RESET_PERIOD=EPS_RESET_PERIOD, LR_C=config.learningRate,
      LR_C_PERIOD=updatePeriod, LR_C_DECAY=0.8, MAX_MODEL=50
  )

  # == AGENT ==
  dimList = [stateDim] + list(CONFIG.ARCHITECTURE) + [actionNum]
  
  agent = DDQNSingle(
      CONFIG, actionNum, actionList, dimList=dimList, mode=config.mode,
      terminalType=config.terminalType
  )
  print("We want to use: {}, and Agent uses: {}".format(device, agent.device))
  print("Critic is using cuda: ", next(agent.Q_network.parameters()).is_cuda)

  vmin = -1
  vmax = 1
  if config.warmup:
    print("\n== Warmup Q ==")
    lossList = agent.initQ(
        env, config.warmupIter, outFolder, num_warmup_samples=200, vmin=vmin,
        vmax=vmax, plotFigure=config.plotFigure, storeFigure=config.storeFigure
    )

    if config.plotFigure or config.storeFigure:
      fig, ax = plt.subplots(1, 1, figsize=(4, 4))
      tmp = np.arange(25, config.warmupIter)
      #tmp = np.arange(config.warmupIter)
      ax.plot(tmp, lossList[tmp], 'b-')
      ax.set_xlabel('Iteration', fontsize=18)
      ax.set_ylabel('Loss', fontsize=18)
      plt.tight_layout()

      if config.storeFigure:
        figurePath = os.path.join(figureFolder, 'initQ_Loss.png')
        fig.savefig(figurePath)
      if config.plotFigure:
        plt.show()
        plt.pause(0.001)
      plt.close()

  print("\n== Training Information ==")
  vmin = -1
  vmax = 1
  trainRecords, trainProgress = agent.learn(
      env, MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps, warmupQ=False,
      doneTerminate=True, vmin=vmin, vmax=vmax, showBool=True,
      checkPeriod=config.checkPeriod, outFolder=outFolder,
      plotFigure=config.plotFigure, storeFigure=config.storeFigure
  )

  trainDict = {}
  trainDict['trainRecords'] = trainRecords
  trainDict['trainProgress'] = trainProgress
  filePath = os.path.join(outFolder, 'train')

  save_obj(trainDict, filePath)


def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value
if __name__ == "__main__":

    # == ARGS ==
    parser = argparse.ArgumentParser()

    # environment parameters
    parser.add_argument(
        "-dt", "--doneType", help="when to raise done flag", default='toEnd',
        type=str
    )
    parser.add_argument(
        "-ct", "--costType", help="cost type", default='sparse', type=str
    )
    parser.add_argument(
        "-rnd", "--randomSeed", help="random seed", default=0, type=int
    )

    # car dynamics
    parser.add_argument(
        "-cr", "--consRadius", help="constraint radius", default=0.5, type=float
    )

    parser.add_argument(
        "-turn", "--turnRate", help="turning rate", default=1.25, type=float
    )
    parser.add_argument(
        "--dt", help="timestep", default=0.05, type=float
    )
    parser.add_argument("-s", "--speed", help="speed", default=1., type=float)

    # training scheme
    parser.add_argument(
        "-w", "--warmup", help="warmup Q-network", action="store_true"
    )
    parser.add_argument(
        "-wi", "--warmupIter", help="warmup iteration", default=10000, type=int
    )
    parser.add_argument(
        "-mu", "--maxUpdates", help="maximal #gradient updates", default=400000,
        type=int
    )
    parser.add_argument(
        "-ut", "--updateTimes", help="#hyper-param. steps", default=20, type=int
    )
    parser.add_argument(
        "-mc", "--memoryCapacity", help="memoryCapacity", default=10000, type=int
    )
    parser.add_argument(
        "-cp", "--checkPeriod", help="check period", default=10000, type=int
    )

    # hyper-parameters
    parser.add_argument(
        "-a", "--annealing", help="gamma annealing", action="store_true"
    )
    parser.add_argument(
        "-arc", "--architecture", help="NN architecture", default=[100, 100],
        nargs="*", type=int
    )
    parser.add_argument(
        "-lr", "--learningRate", help="learning rate", default=1e-3, type=float
    )
    parser.add_argument(
        "-g", "--gamma", help="contraction coeff.", default=0.9999, type=float
    )
    parser.add_argument(
        "-act", "--actType", help="activation type", default='Tanh', type=str
    )

    # RL type
    parser.add_argument("-m", "--mode", help="mode", default='RA', type=str)
    parser.add_argument(
        "-tt", "--terminalType", help="terminal value", default='g', type=str
    )

    # file
    parser.add_argument(
        "-st", "--showTime", help="show timestr", action="store_true"
    )
    parser.add_argument("-n", "--name", help="extra name", default='', type=str)
    parser.add_argument(
        "-of", "--outFolder", help="output file", default='experiments', type=str
    )
    parser.add_argument(
        "-pf", "--plotFigure", help="plot figures", action="store_true"
    )
    parser.add_argument(
        "-sf", "--storeFigure", help="store figures", action="store_true"
    )

    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--expt_name", type=str, default=None)
    parser.add_argument("--resume_run", type=bool, default=False)
    # environment parameters
    config, remaining = parser.parse_known_args()

    if not config.resume_run:
        curr_time = datetime.now().strftime("%m%d/%H%M%S")
        config.expt_name = (
            f"{curr_time}_{config.expt_name}" if config.expt_name else curr_time
        )
    else:
        assert config.expt_name, "Need to provide experiment name to resume run."

    yaml = yaml.YAML(typ="safe", pure=True)
    configs = yaml.load(
        (pathlib.Path(sys.argv[0]).parent / "../dreamerv3-torch/configs.yaml").read_text()
    )

    name_list = ["defaults", *config.configs] if config.configs else ["defaults"]

    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    print(defaults.keys())
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    final_config = parser.parse_args(remaining)

    final_config.logdir = f"{final_config.logdir}/{config.expt_name}"
    #final_config.time_limit = HORIZONS[final_config.task.split("_")[-1]]

    print("---------------------")
    cprint(f"Experiment name: {config.expt_name}", "red", attrs=["bold"])
    cprint(f"Task: {final_config.task}", "cyan", attrs=["bold"])
    cprint(f"Logging to: {final_config.logdir}", "cyan", attrs=["bold"])
    print("---------------------")

    final_config.name = 'latent'

    RARL(final_config)
  
