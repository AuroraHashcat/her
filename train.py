import numpy as np
import gym
import os, sys
from arguments import get_args
from mpi4py import MPI
from rl_modules.ddpg_agent import ddpg_agent
import random
import torch


# gy = True: gym environment, False:AntReacher
# log = True: wandb
#test = True: 评估模型，不训练
#show: mujoco_py render
#arguments
# PS: wandb记录的表名需要手动改。在wandb.init和ddpg_agent.py里。savemodel+seed
import design_env
gy = False
log = True
test = False
show = False
her = True

# ant_reacher: python train.py --cuda cuda:1 --seed 1
# ant_four_rooms: python train.py --cuda cuda:1 --env-name ant_four_rooms --seed 1


if(log == True and MPI.COMM_WORLD.Get_rank() == 0):
    import wandb
    # cx's key
    # os.environ["WANDB_API_KEY"] = "7345a4ba788b2d78ab6a78d185784b2ea818317e"
    # jy's key
    os.environ["WANDB_API_KEY"] = "9d317e91d5b56a3aa6f1fe7463d10fa81824ed45"
    wandb.login()
    wandb.init(
        project="HER", name="apo_ant_4rooms_seed_4",group="apo_ant_4rooms"
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
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    if gy == True:
        env = gym.make(args.env_name)
        # set random seeds for reproduce
        env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
        # get the environment parameters
        env_params = get_env_params(env)
    else:
        env = design_env.design_env(args.env_name,show)
        end_goal = env.get_next_goal(test)
        end_goal = env.project_state_to_end_goal(env.sim, end_goal)
        observation = env.reset_sim(end_goal)
        env_params = {'obs':observation.shape[0],
                    'goal':end_goal.shape[0],
                    'action':env.action_dim,
                    'action_max':env.action_bounds_high,
                    'max_timesteps':env.max_actions}
    # create the ddpg agent to interact with the environment 
    ddpg_trainer = ddpg_agent(args, env, env_params, gy, test, her)
    ddpg_trainer.learn(log,show)

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
