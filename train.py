import numpy as np
import gym
import os, sys
from arguments import get_args
from mpi4py import MPI
from rl_modules.ddpg_agent import ddpg_agent
import random
import torch





"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""

# gy = True: gym environment, False:AntReacher
# log = True: wandb
# her = True: HER Strategy   False: DDPG
#test = True: 评估模型，不训练
#arguments
# PS: wandb记录的表名需要手动改。在wandb.init和ddpg_agent.py里。savemodel
import design_env
gy = False
log = False
her = True
test = False

if(log == True):
    import wandb
    os.environ["WANDB_API_KEY"] = "7345a4ba788b2d78ab6a78d185784b2ea818317e"
    wandb.login()
    wandb.init(
        project="GCRLbaselines", name="HER_AntReacher_seed1",group="HER_AntReacher"
    )
    os.environ["WANDB_MODE"] = "offline"

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

def launch(args):
    # create the ddpg_agent
    if (gy == True):
        env = gym.make(args.env_name)
        # set random seeds for reproduce
        env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
        random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
        np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
        torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
        if args.cuda:
            torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
        # get the environment parameters
        env_params = get_env_params(env)
    else:
        env = design_env.design_env(args.env_name)
        end_goal = env.get_next_goal()
        observation = env.reset_sim(end_goal)

        env_params = {'obs':observation.shape[0],
                    'goal':end_goal.shape[0],
                    'action':env.action_dim,
                    'action_max':env.action_bounds_high,
                    'max_timesteps':env.max_actions}
    # create the ddpg agent to interact with the environment 
    ddpg_trainer = ddpg_agent(args, env, env_params, gy, her,test)
    ddpg_trainer.learn(log)

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
