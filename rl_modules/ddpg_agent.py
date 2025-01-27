import copy
import random

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
from llama.llama3_test import generator_four, check_quality, generator_reacher
import json

# reward model
from reward_model.train_reward_model import RewardModel

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

        # self.actor_network.load_state_dict(actor_state_dict)
        state_dim = self.env.initial_state_space.shape[0]
        goal_dim = self.env.end_goal_dim - 1  # 只保留目标的两位坐标(x,y)
        action_dim = self.env.action_dim
        self.reward_model = RewardModel(state_dim, action_dim, goal_dim, args)

        # her sampler
        if (self.gy == True):
            self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k ,self.args,self.env,self.env.compute_reward)
        else:
            self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.args , self.env,self.reward_model)
        
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions,self.gy,self.her)
        # self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_subgoal_transitions,self.gy,self.her)
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

        self.llm_input = []
        self.llm_output = []

        self.goal_array = []
        self.relabel_temp = {}
        self.transition = []
        self.new_transition = []

        self.eff_traj_num = 0
        self.train_traj_num = 0 # decide when to update reward model
        self.test_traj_num = 0
        self.test_reward = 0

        self.test_data_for_rm = []

        self.step = 0

    def llm_choose_subgoal(self, position):

        # reset subgoal_num
        self.subgoal_num = None

        """use llama-3.1-8B-Instruct to select subgoals"""
        while True:
            start_time = time.time()
            try:
                messages, self.llm_output = generator_reacher(position)
                epsilon = np.random.uniform(0, 1)
                if (epsilon < 0.3):
                    self.llm_output = check_quality(messages, self.llm_output)
                end_time = time.time()
                llm_generated_time = end_time - start_time
                self.llm_output = json.loads(self.llm_output)
                if isinstance(self.llm_output, list) and all(
                        isinstance(coord, list) and len(coord) == 2 for coord in self.llm_output) and len(
                    self.llm_output) <= 10:
                    self.goal_array.extend(self.llm_output)
                    self.subgoal_num = len(self.llm_output)
                    break  # 如果所有检查都通过，退出循环
            except (json.JSONDecodeError, AssertionError) as e:
                print(f"Output '{self.llm_output}' does not meet the criteria. Regenerating...")

        print("llm output:", self.llm_output)
        print("the number of subgoals:", self.subgoal_num)
        print("goal_array:", self.goal_array)
        return self.goal_array, llm_generated_time

    def sparse_reward_subgoal(self, env, state,t):

        goal_for_transition = None
        goal_status = [False for i in range(len(self.goal_array))]
        reward = -1

        # Project next_state onto end goal spaces, the self.current_state is already replaced by the next_state!!!
        proj_end_goal = env.project_state_to_end_goal(env.sim, state) # 取前2位

        for i in range(len(self.goal_array)):

            goal_achieved = True

            # If the difference in two dimension is greater than threshold, goal not achieved
            for j in range(len(proj_end_goal)): # len(proj_end_goal) = 2
                if np.absolute(self.goal_array[i][j] - proj_end_goal[j]) > env.end_goal_thresholds[j]:
                    goal_achieved = False
                    break

            # If projected state within threshold of goal, mark as achieved
            if goal_achieved:
                goal_status[i] = True
                if i == 0:
                    reward = 0
                    print(f"steps {t}, end point is achieved, mission success")
                    goal_for_transition = self.goal_array[0]
                    del self.goal_array[i]
                    print("goal_array_still:", self.goal_array)
                    return goal_status, reward, goal_for_transition
                # subgoal achieved, delete subgoal from goal_array
                else:
                    reward = 0
                    # print("reward", reward)
                    print(f"steps {t}, one subgoal is achieved")
                    goal_for_transition = self.goal_array[i]
                    del self.goal_array[i]
                    print("goal_array_still:", self.goal_array)
                    return goal_status, reward, goal_for_transition
            else:
                goal_status[i] = False
                reward = -1
                goal_for_transition = self.goal_array[0]

        return goal_status, reward, goal_for_transition

    def learn_reward_model(self, reward_model_path):
        # 1) construct preference dataset
        preference_pair = self.reward_model.construct_pbrl_data()

        # 2) train reward model
        print("the reward model is start to update!")
        if self.args.bceloss:
            self.reward_model.train_reward_model_bceloss(preference_pair, reward_model_path)
        elif self.args.celoss:
            self.reward_model.train_reward_model_celoss(preference_pair)
        self.reward_model.save(reward_model_path)

    def test_reward_model(self):

        test_set = self.test_data_for_rm
        print(f"test reward model, and the test_set is {len(test_set)}")

        sampled_trajectories = random.sample(test_set, 1)
        total_reward = 0
        i = 0

        for trajectory in sampled_trajectories:
            reward_sum = 0
            for eff_transitions in trajectory:
                # eff_transitions = [lst[:3] for lst in transition]
                state = np.array(eff_transitions[0])
                # print(state)
                action = np.array(eff_transitions[1])
                # print(action)
                goal = np.array(eff_transitions[2])
                # print(goal)
                sga_t = np.concatenate([state, goal, action], axis=-1)
                sga_t = np.array(sga_t)
                reward_hat = self.reward_model.r_hat(sga_t)
                # print(reward_hat)
                reward_sum += reward_hat
                # print(f"Transition: {eff_transitions[i]}, Reward Hat: {reward_hat}")
                # print(f"Reward Hat: {reward_hat}")
                # wandb.log({"reward_hat": reward_hat}

            average_reward = reward_sum / len(trajectory)
            # print(f"第{i}条trajectory的reward_sum_average", average_reward)
            i += 1
            total_reward += average_reward
        total_average_reward = total_reward / 1
        print(f"{len(sampled_trajectories)}条sampled_trajectories的average_reward", total_average_reward)

        return total_average_reward

    def learn(self,log,show):
        # start to collect samples
        for epoch in range(self.args.n_epochs):    #1000
            if (self.test == False):
                for cycle in range(self.args.n_cycles):      #100    episode
                    mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                    for _ in range(self.args.num_rollouts_per_mpi):  #1

                        # update reward model
                        if self.args.reward_model:
                            if cycle % self.args.rm_update_freq == 0 and self.train_traj_num >= self.args.rm_update_traj_num:  # 10
                                print(
                                    f"we have {self.train_traj_num} train_set and {self.test_traj_num} test_set so far.")

                                # learn reward model
                                print("\nBatch:", epoch, "Episode:", cycle, "start update reward model network")
                                rm_path = f"{self.model_path}/rm_model_seed_{self.args.seed}.pt"
                                self.learn_reward_model(rm_path)

                                if self.test_traj_num >= 1:
                                    # test reward model
                                    self.test_reward = self.test_reward_model()

                                self.args.reward_model_relabel = True

                        # reset the rollouts
                        ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                        self.goal_array = []

                        # reset the environment
                        print("\nBatch:", epoch, "Episode:", cycle)
                        achieved_subgoal_num = 0
                        relabel_clip = False
                        done = False
                        self.relabel_temp = {}
                        self.transition = []

                        g = self.env.get_next_goal(self.test)
                        g = self.env.project_state_to_end_goal(self.env.sim, g)
                        self.goal_array.append(g)
                        print("Next End Goal: ", self.goal_array[0])
                        obs = self.env.reset_sim(g)
                        print("Initial Ant Position: ", obs[:3])
                        ag = self.env.project_state_to_end_goal(self.env.sim,obs)


                        self.llm_input = "start point" + str(np.round(ag)) + " " + "end point" + str(np.round(g))
                        self.goal_array, time = self.llm_choose_subgoal(self.llm_input)

                        for t in range(self.env_params['max_timesteps']):
                            action = self.env.action_space.sample()
                            obs_new = self.env.execute_action(action)
                            ag_new = self.env.project_state_to_end_goal(self.env.sim,obs_new)

                            goal_status, reward_relabelled, goal_relabelled = self.sparse_reward_subgoal(self.env,
                                                                                                         obs_new,t)

                            if goal_status[0]:
                                done = True
                            if t + 1 == self.env_params['max_timesteps']:
                                done = True
                                print("Out of actions (Steps: %d)" % t)
                            # calculate reward
                            done = float(done)
                            done_no_max = 0 if t + 1 == self.env.max_actions else done

                            if  reward_relabelled == 0:
                                achieved_subgoal_num += 1
                                relabel_clip = True
                                # goal
                                relabel_goal = goal_relabelled
                                # index
                                relabel_goal_index = t
                                self.relabel_temp[f'subgoal_{achieved_subgoal_num}'] = (
                                achieved_subgoal_num, relabel_goal, relabel_goal_index)

                            self.transition.append(
                                [obs, action, g, reward_relabelled, obs_new, done, done_no_max])


                            ep_obs.append(obs.copy())
                            ep_ag.append(ag.copy())
                            ep_g.append(g.copy())
                            ep_actions.append(action.copy())
                            # re-assign the observation
                            obs = obs_new
                            ag = ag_new
                            self.step += 1
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())


                        print("relabel temp dict: ", self.relabel_temp)
                        """
                        1) if done_no_max is True, there is no need for relabel.
                        2) if done_no_max is False and no sub-goal is achieved, there is no need for relabel either. ->  add her future relabel?
                        3) if done_no_max is False and sub-goal is True, run relabel clip: -> multiple trajectories
                        """
                        if done_no_max:  # -> 1) add her future relabel  2) gcsl relabel (ablation study)
                            self.eff_traj_num += 1
                            self.train_traj_num += 1

                            for transition in self.transition:
                                state, action, goal, reward, next_state, done, done_no_max = transition
                                obs = np.concatenate((state, goal), axis=0)
                                next_obs = np.concatenate((next_state, goal), axis=0)
                                # 1) add RM buffer (s, a, g, r, done_no_max)
                                self.reward_model.add_eff_data_sga(state, action, goal, reward, done_no_max)

                        elif relabel_clip:
                            for key, (subgoal_num, goal, goal_index) in self.relabel_temp.items():
                                print(key, subgoal_num, goal, goal_index)
                                if goal_index == 0:  # if the step is 0 and the sub goal is achieved.
                                    print("goal_index is 0, breaking out of the loop, no need for relabel.")
                                    continue
                                # 1) clip transition and relabel done
                                self.new_transition = copy.deepcopy(
                                    self.transition[:(goal_index + 1)])  # deepcopy创建全新的切片
                                self.new_transition[-1][5] = True
                                # done_no_max
                                self.new_transition[-1][6] = float(self.new_transition[-1][5])
                                # 2) relabel goal
                                for i in range(len(self.new_transition)):
                                    self.new_transition[i][2] = goal  # 更新goal_relabelled为新的goal
                                # 3） recompute reward
                                for i in range(len(self.new_transition)):
                                    # 此处应该测next_state和goal之间的距离阈值
                                    if (np.absolute(self.new_transition[i][4][0] - goal[0]) <
                                        self.env.end_goal_thresholds[
                                            0]) and \
                                            (np.absolute(self.new_transition[i][4][1] - goal[1]) <
                                             self.env.end_goal_thresholds[
                                                 1]):
                                        self.new_transition[i][3] = 1
                                    else:
                                        self.new_transition[i][3] = -1

                                self.eff_traj_num += 1

                                # store data into train/test set
                                # 1) test_set
                                if self.eff_traj_num % 10 == 9:  # train_set:test_set = 9:1
                                    self.test_traj_num += 1
                                    # add test data for reward model
                                    self.test_data_for_rm.append(self.new_transition)

                                # 2) train_set
                                else:
                                    self.train_traj_num += 1
                                    for transition in self.new_transition:
                                        state, action, goal, reward, next_state, done, done_no_max = transition
                                        obs = np.concatenate((state, goal), axis=0)
                                        next_obs = np.concatenate((next_state, goal), axis=0)
                                        # 1) add train_set data into RM buffer (s, a, g, r, done_no_max)
                                        self.reward_model.add_eff_data_sga(state, action, goal, reward, done_no_max)

                        else:
                            for j in range(len(self.transition)):
                                state, action, goal, reward, next_state, done, done_no_max = self.transition[j]
                                # 1) add RM buffer (s, a, g, r, done)
                                self.reward_model.add_nor_data_sga(state, action, goal, reward, done)

                        print(
                            f"we have {self.eff_traj_num} eff_traj, {self.train_traj_num} train_set and {self.test_traj_num} test_set so far.")
                        print()

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
                    self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                    #relabel
                    self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                    for _ in range(self.args.n_batches):
                        # train the network
                        self._update_network(log)
                    # soft update
                    self._soft_update_target_network(self.actor_target_network, self.actor_network)
                    self._soft_update_target_network(self.critic_target_network, self.critic_network)
            # start to do the evaluation
            success_rate = self._eval_agent(show)
            
            
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, success_rate))
                if (log == True):
                    wandb.log({"AntReacher/success rate": success_rate},step=self.step)
                if (not self.test):
                    torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
                             f"{self.model_path}/apo_sparse_model_seed_{self.args.seed}.pt")

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
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
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
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions,self.gy,self.her)
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
    def _update_network(self,log):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)   #256
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
                        "AntReacher/critic_loss": critic_loss},step=self.step)
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
                    if show:
                        if(self.env.success(obs,g)):
                            break
                    else:
                        per_success_rate.append(self.env.success(obs,g))


                
            total_success_rate.append(per_success_rate)

        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()
