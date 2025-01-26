"""
This file provides the template for designing the agent and environment.  The below hyperparameters must be assigned to a value for the algorithm to work properly.
"""

import numpy as np
from environment import Environment
import gym



def design_env(model_name,show):


    #model_name: ant_reacher, pendulum, ur5, ant_four_rooms
    model_name = model_name + ".xml"
        
    max_actions = 800
    timesteps_per_action = 15 

    initial_joint_pos = np.array([0, 0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
    initial_joint_pos = np.reshape(initial_joint_pos,(len(initial_joint_pos),1))
    initial_joint_ranges = np.concatenate((initial_joint_pos,initial_joint_pos),1)
    initial_joint_ranges[0] = np.array([-6,6])
    initial_joint_ranges[1] = np.array([-6,6])
    initial_state_space = np.concatenate((initial_joint_ranges,np.zeros((len(initial_joint_ranges)-1,2))),0)

    max_range = 6
    goal_space_train = [[-max_range,max_range],[-max_range,max_range],[0.45,0.55]]
    goal_space_test = [[-max_range,max_range],[-max_range,max_range],[0.45,0.55]]

    project_state_to_end_goal = lambda sim,state: state[:2]

    len_threshold = 0.4
    height_threshold = 0.2
    end_goal_thresholds = np.array([len_threshold, len_threshold, height_threshold])


    env = Environment(model_name, goal_space_train, goal_space_test, project_state_to_end_goal, end_goal_thresholds, 
    initial_state_space, max_actions, timesteps_per_action, show)

    return  env