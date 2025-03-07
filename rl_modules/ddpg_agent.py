import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import replay_buffer
from rl_modules.models import actor, critic
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
import wandb
import time
from llama.llama3_test import generator_four, check_quality
import json

"""
ddpg with HER (MPI-version)

"""

class ddpg_agent:
    def __init__(self, args, env, env_params,gy,test,her):
        self.args = args
        self.env = env
        self.env_params = env_params
        self.gy = gy
        self.her = her
        self.test = test
        # create the network
        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        self.actor_target_network = actor(env_params)
        self.critic_target_network = critic(env_params)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # her sampler
        if (self.gy == True):
            self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        else:
            self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.sparse_reward)
        
        # create the replay buffer
        #self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions,self.gy,self.her)
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_subgoal_transitions,self.gy,self.her)
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

        # if (self.test):
        # saved_data = torch.load('/home/wuchenxi/projects/hindsight-experience-replay/saved_models/ant_reacher/model_seed1.pt')
        # o_mean, o_std, g_mean, g_std, actor_state_dict = saved_data

        # self.o_norm.mean = o_mean
        # self.o_norm.std = o_std
        # self.g_norm.mean = g_mean
        # self.g_norm.std = g_std

        # self.actor_network.load_state_dict(actor_state_dict)

    def llm_choose_subgoal(self, position):

        # reset subgoal_num
        subgoal_num = None

        """use llama-3.1-8B-Instruct to select subgoals"""
        while True:
            start_time = time.time()
            try:
                messages, llm_output = generator_four(position)
                epsilon = np.random.uniform(0, 1)
                if (epsilon < 0.3):
                    llm_output = check_quality(messages, llm_output)
                end_time = time.time()
                llm_generated_time = end_time - start_time
                llm_output = json.loads(llm_output)
                if isinstance(llm_output, list) and all(
                        isinstance(coord, list) and len(coord) == 2 for coord in llm_output) and len(
                        llm_output) <= 10:
                    subgoal_num = len(llm_output)
                    break  # 如果所有检查都通过，退出循环
            except (json.JSONDecodeError, AssertionError) as e:
                print(f"Output '{self.llm_output}' does not meet the criteria. Regenerating...")

        print("llm output:", llm_output)
        print("the number of subgoals:", subgoal_num)
        return llm_output, llm_generated_time


    def learn(self,log,show):
        # start to collect samples
        for epoch in range(self.args.n_epochs):    #1000
            if (self.test == False):
                for _ in range(self.args.n_cycles):      #100    episode
                    mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                    for _ in range(self.args.num_rollouts_per_mpi):  #1
                        # reset the rollouts
                        ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                        # reset the environment
                        if (self.gy == True):
                            observation = self.env.reset()
                            obs = observation['observation']
                            ag = observation['achieved_goal']
                            g = observation['desired_goal']
                            # start to collect samples
                            for t in range(self.env_params['max_timesteps']):    #800
                                with torch.no_grad():
                                    input_tensor = self._preproc_inputs(obs, g)
                                    pi = self.actor_network(input_tensor)
                                    action = self._select_actions(pi)
                                # feed the actions into the environment
                                observation_new, _, _, info = self.env.step(action)
                                obs_new = observation_new['observation']
                                ag_new = observation_new['achieved_goal']
                                ep_obs.append(obs.copy())
                                ep_ag.append(ag.copy())
                                ep_g.append(g.copy())
                                ep_actions.append(action.copy())
                                # re-assign the observation
                                obs = obs_new
                                ag = ag_new

                            ep_obs.append(obs.copy())
                            ep_ag.append(ag.copy())
                        else:
                            g = self.env.get_next_goal(self.test)
                            g = self.env.project_state_to_end_goal(self.env.sim, g)
                            obs = self.env.reset_sim(g)
                            ag = self.env.project_state_to_end_goal(self.env.sim,obs)
                            # subgoals = [[1,1],[2,2]]
                            llm_input = "start point" + str(np.round(ag)) + " " + "end point" + str(np.round(g))
                            subgoals, time = self.llm_choose_subgoal(llm_input)
                            for t in range(self.env_params['max_timesteps']):
                                action = self.env.action_space.sample()
                                obs_new = self.env.execute_action(action)
                                ag_new = self.env.project_state_to_end_goal(self.env.sim,obs_new)
                                
                                ep_obs.append(obs.copy())
                                ep_ag.append(ag.copy())
                                ep_g.append(g.copy())
                                ep_actions.append(action.copy())
                                # re-assign the observation
                                obs = obs_new
                                ag = ag_new
                            ep_obs.append(obs.copy())
                            ep_ag.append(ag.copy())

                        mb_obs.append(ep_obs)
                        mb_ag.append(ep_ag)
                        mb_g.append(ep_g)
                        mb_actions.append(ep_actions)
                    # convert them into arrays
                    mb_obs = np.array(mb_obs)
                    mb_ag = np.array(mb_ag)
                    mb_g = np.array(mb_g)
                    mb_actions = np.array(mb_actions)
                    # store the episodes
                    self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions,subgoals])
                    #relabel
                    self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions,subgoals])
                    for _ in range(self.args.n_batches):
                        # train the network
                        self._update_network(log,subgoals)
                    # soft update
                    self._soft_update_target_network(self.actor_target_network, self.actor_network)
                    self._soft_update_target_network(self.critic_target_network, self.critic_network)
            # start to do the evaluation
            success_rate = self._eval_agent(show)
            
            
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, success_rate))
                if (log == True):
                    wandb.log({"AntWShape/success rate": success_rate})
                if (not self.test):
                    torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
                            self.model_path + '/apo_sparse_model_seed4.pt')

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs
    
    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], 
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions,subgoals = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       'subgoals':subgoals
                       }
        # transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions,self.gy,self.her)
        transitions = self.her_module.sample_subgoal_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self,log,subgoals):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size,subgoals)   #256
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm],axis = 1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm],axis = 1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_np_array = np.array(transitions['actions'])
 
        # 然后，将这个 numpy.ndarray 转换为 PyTorch 张量
        actions_tensor = torch.tensor(actions_np_array, dtype=torch.float32)

        # actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        if (log == True):
            wandb.log({"AntReacher/actor_loss": actor_loss,
                        "AntReacher/critic_loss": critic_loss})
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

    # do the evaluation
    def _eval_agent(self,show):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):  #100
            per_success_rate = []
            if (self.gy == True):
                observation = self.env.reset()
                obs = observation['observation']
                g = observation['desired_goal']
                for _ in range(self.env_params['max_timesteps']):
                    with torch.no_grad():
                        input_tensor = self._preproc_inputs(obs, g)
                        pi = self.actor_network(input_tensor)
                        # convert the actions
                        actions = pi.detach().cpu().numpy().squeeze()
                    observation_new, _, _, info = self.env.step(actions)
                    obs = observation_new['observation']
                    g = observation_new['desired_goal']
                    if (show):
                        if(info['is_success']):
                            break
                    else:
                        per_success_rate.append(info['is_success'])
                
            else:
                g = self.env.get_next_goal(self.test)
                g = self.env.project_state_to_end_goal(self.env.sim, g)
                obs = self.env.reset_sim(g)
                for t in range(self.env_params['max_timesteps']):
                    with torch.no_grad():
                        input_tensor = self._preproc_inputs(obs, g)
                        pi = self.actor_network(input_tensor)
                        # convert the actions
                        action = pi.detach().cpu().numpy().squeeze()
                    obs = self.env.execute_action(action)
                    if (show):
                        if(self.env.success(obs,g)):
                            break
                    else:
                        per_success_rate.append(self.env.success(obs,g))


                
            total_success_rate.append(per_success_rate)

        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()
