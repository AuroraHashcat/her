"""
This file provides the template for designing the agent and environment.  The below hyperparameters must be assigned to a value for the algorithm to work properly.
"""

import numpy as np
from environment import Environment
import gym



def design_env(model_name,show):


    #model_name
    #model_name = "ant_reacher.xml"
    model_name = model_name + ".xml"
    #model_name = 'pendulum.xml'


    
    if (model_name == "ant_reacher.xml"):
        #max_actions and timesteps_per_action
        max_actions = 800
        timesteps_per_action = 15 

        #initial_state_space
        initial_joint_pos = np.array([0, 0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
        initial_joint_pos = np.reshape(initial_joint_pos, (len(initial_joint_pos), 1))
        initial_joint_ranges = np.concatenate((initial_joint_pos, initial_joint_pos), 1)
        initial_joint_ranges[0] = np.array([-9.5, 9.5])
        initial_joint_ranges[1] = np.array([-9.5, 9.5])
        initial_state_space = np.concatenate((initial_joint_ranges, np.zeros((len(initial_joint_ranges) - 1, 2))), 0)

        #goal_space_train and goal_space_test
        max_range = 9.5
        goal_space_train = [[-max_range, max_range], [-max_range, max_range], [0.45, 0.55]]
        goal_space_test = [[-max_range, max_range], [-max_range, max_range], [0.45, 0.55]]

        #state->end_goal
        project_state_to_end_goal = lambda sim, state: state[:3]

        #end_goal_thresholds
        len_threshold = 0.5
        height_threshold = 0.2
        end_goal_thresholds = np.array([len_threshold, len_threshold, height_threshold])

        # Instantiate and return agent and environment
        

    elif (model_name == "pendulum.xml"):
        #max_actions and timesteps_per_action
        max_actions = 1000
        timesteps_per_action = 10

        #initial_state_space
        initial_state_space = np.array([[np.pi/4, 7*np.pi/4],[-0.05,0.05]])

        #goal_space_train and goal_space_test
        goal_space_train = [[np.deg2rad(-16),np.deg2rad(16)],[-0.6,0.6]]
        goal_space_test = [[0,0],[0,0]]

        #state->end_goal
        def bound_angle(angle):
            bounded_angle = angle % (2*np.pi)
            if np.absolute(bounded_angle) > np.pi:
                bounded_angle = -(np.pi - bounded_angle % np.pi)
            return bounded_angle
        project_state_to_end_goal = lambda sim, state: np.array([bound_angle(sim.data.qpos[0]), 15 if state[2] > 15 else -15 if state[2] < -15 else state[2]])

        #end_goal_thresholds
        end_goal_thresholds = np.array([np.deg2rad(9.5), 0.6])
       

    else:
        #max_actions and timesteps_per_action
        max_actions = 600
        timesteps_per_action = 15 

        #initial_state_space
        initial_joint_pos = np.array([  5.96625837e-03,   3.22757851e-03,  -1.27944547e-01])
        initial_joint_pos = np.reshape(initial_joint_pos,(len(initial_joint_pos),1))
        initial_joint_ranges = np.concatenate((initial_joint_pos,initial_joint_pos),1)
        initial_joint_ranges[0] = np.array([-np.pi/8,np.pi/8])
        # initial_joint_ranges[1] = np.array([-np.pi/4,0])
        initial_state_space = np.concatenate((initial_joint_ranges,np.zeros((len(initial_joint_ranges),2))),0)

        #goal_space_train and goal_space_test
        goal_space_train = [[-np.pi,np.pi],[-np.pi/4,0],[-np.pi/4,np.pi/4]]
        goal_space_test = [[-np.pi,np.pi],[-np.pi/4,0],[-np.pi/4,np.pi/4]]

        #state->end_goal
        def bound_angle(angle):
            bounded_angle = np.absolute(angle) % (2*np.pi)
            if angle < 0:
                bounded_angle = -bounded_angle
            return bounded_angle
        project_state_to_end_goal = lambda sim, state: np.array([bound_angle(sim.data.qpos[i]) for i in range(len(sim.data.qpos))])

        #end_goal_thresholds
        angle_threshold = np.deg2rad(10)
        end_goal_thresholds = np.array([angle_threshold, angle_threshold, angle_threshold])


    env = Environment(model_name, goal_space_train, goal_space_test, project_state_to_end_goal, end_goal_thresholds, 
    initial_state_space, max_actions, timesteps_per_action, show)

    return  env