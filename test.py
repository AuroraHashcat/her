import design_env
import numpy as np
import gym

def sparse_reward(env, end_goal):
    goal_achieved = True

    proj_end_goal = env.project_state_to_end_goal(env.sim, env.get_state())

    for j in range(len(proj_end_goal) - 1):
        if np.absolute(end_goal[j] - proj_end_goal[j]) > env.end_goal_thresholds[j]:
            goal_achieved = False
            break

    if goal_achieved:
        reward = 10
        print(f"steps {self.step}, end point is achieved, mission success")
    else:
        reward = -1

    return goal_achieved, reward





env = design_env.design_env()


end_goal = env.get_next_goal(True)
print("endgoal",end_goal)

state = env.reset_sim(end_goal)
print("state",state)

obs = np.concatenate((state, end_goal))
action = env.action_space.sample()

next_state = env.execute_action(action)
print("nextstate",next_state)



next_obs = np.concatenate((next_state, end_goal))
done, reward = sparse_reward(env, end_goal)
print("done",done)
print("reward",reward)


'''
env = gym.make("FetchReach-v1")
observation = env.reset()
print(observation)
'''
