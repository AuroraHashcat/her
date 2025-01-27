import numpy as np
import random


class her_sampler:
    def __init__(self, replay_strategy, replay_k, args,env, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        self.args = args
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func
        self.env = env


    def sample_her_transitions(self, episode_batch, batch_size_in_transitions,gy,her):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # to get the params to re-compute reward

        if self.args.reward_model_relabel:
            # print(transitions['obs'].shape)  # 应该是 (N, 29)
            # print(transitions['g'].shape)  # 应该是 (N, 2)
            # print(transitions['actions'].shape)  # 应该是 (N, 8)
            # combined = np.concatenate([transitions['obs'], transitions['g'], transitions['actions']], axis=-1)
            # print(combined.shape)  # 应该是 (N, 29 + 2 + 8) = (N, 39)
            transitions['r'] = np.expand_dims(self.env.sparse_reward(transitions['ag_next'], transitions['g']), 1)
            # print(f"her: {transitions['r']}")
            for idx in range(transitions['r'].shape[0]):
                if transitions['r'][idx] != 0:
                    transitions['r'][idx] = np.expand_dims(self.reward_func.r_hat_batch(np.concatenate([transitions['obs'][idx], transitions['g'][idx],transitions['actions'][idx]], axis=-1)), 1)
            # print(f"rm: {transitions['r']}")
        else:
            transitions['r'] = np.expand_dims(self.env.sparse_reward(transitions['ag_next'], transitions['g']), 1)
            # print(transitions['r'].shape)
            # reward_mean = np.mean(transitions['r'])
            # print(f"Mean reward her: {reward_mean}")
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
        return transitions

    def sample_subgoal_transitions(self, episode_batch, batch_size_in_transitions):  #先relabel，把成功的拿出来（或者加个概率），剩下的在失败里随机选
        episode = {key:episode_batch[key][0] for key in episode_batch.keys() if key != 'subgoals'} #没有把mb和ep合成一个，如果去掉了rollouts则不需要这一句
        episode['subgoals'] = episode_batch['subgoals']
        episode['r'] = []
        episodes = []
        #episode是一个episode完整的traj，是字典，有'obs','ag','g','actions','obs_next','ag_next','subgoals'
        transitions = {key:[] for key in episode.keys()}
        success_num = 0
        for subgoal in episode['subgoals']:  
            for T in range(batch_size_in_transitions):
                # relabel出来subgoals条新的traj
                episode['g'][T] = subgoal
                # 检查subgoals是否到达
                reward, achieved = self.reward_func(episode['obs'][T], episode['g'][T])
                episode['r'].append(reward)
                if (achieved):
                    for key in episode.keys():
                        if (key != 'subgoals'):
                            transitions[key].append(episode[key][T])
                    success_num += 1
                    break
            truncated_episode = {key:episode[key][:T+1] for key in episode.keys()}
            episodes.append(truncated_episode)
            #得到了处理完的新traj: truncated_episode
        #得到了新的subgoals个traj：episodes list 存 episode dict, 要选episode-timesteps的transition共batch_size个组成transitions返回
        # transitions是一个字典，值是只有某键的随机序列，如obs随机序列，但transitions[key][i]是一个transition里拿出来的
        for i in range(batch_size_in_transitions - success_num):
            episode_idx = random.randint(0,len(episodes)-1)
            time_idx = random.randint(0,episodes[episode_idx]['actions'].shape[0]-1)
            #print(episode_idx, time_idx)
            for key in episode.keys():
                #print(episodes[episode_idx][key].shape)
                if (key != 'subgoals'):
                    transitions[key].append(episodes[episode_idx][key][time_idx])
        return transitions
