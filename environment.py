from tkinter import *
from tkinter import ttk
import time
import numpy as np
#from PyQt5 import QtCore #.QProcess.state
from mujoco_py import load_model_from_path, MjSim, MjViewer
import gym

class Environment:

    def __init__(self, model_name, goal_space_train, goal_space_test, project_state_to_end_goal, end_goal_thresholds, initial_state_space, subgoal_bounds, subgoal_thresholds, max_actions = 1200, num_frames_skip = 10, show = False):

        self.name = model_name

        # Create Mujoco Simulation
        self.model = load_model_from_path("/home/wuchenxi/project/mujoco_files/"+ model_name)
        self.sim = MjSim(self.model)

        # Set dimensions and ranges of states, actions, and goals in order to configure actor/critic networks
        self.end_goal_dim = len(goal_space_test)
        if model_name == "pendulum.xml":
            self.state_dim = 2*len(self.sim.data.qpos) + len(self.sim.data.qvel) #+ self.end_goal_dim
        else:
            self.state_dim = len(self.sim.data.qpos) + len(self.sim.data.qvel) #+  self.end_goal_dim # State will include (i) joint angles and (ii) joint velocities
        self.action_dim = len(self.sim.model.actuator_ctrlrange) # low-level action dim
        self.action_bounds_low = self.sim.model.actuator_ctrlrange[0][0]
       
        self.action_bounds_high = self.sim.model.actuator_ctrlrange[0][1] # low-level action bounds
        self.action_space = gym.spaces.Box(low=self.action_bounds_low, high=self.action_bounds_high, shape=(self.action_dim,), dtype=np.float32)
        #self.action_offset = np.zeros((len(self.action_bounds))) # Assumes symmetric low-level action ranges

        #self.subgoal_dim = len(subgoal_bounds)
        #self.subgoal_bounds = subgoal_bounds

        # Projection functions
        self.project_state_to_end_goal = project_state_to_end_goal
        #self.project_state_to_subgoal = project_state_to_subgoal

        # Convert subgoal bounds to symmetric bounds and offset.  Need these to properly configure subgoal actor networks
        #self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        #self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

        #for i in range(len(self.subgoal_bounds)):
        #    self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2
        #    self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]


        # End goal/subgoal thresholds
        self.end_goal_thresholds = end_goal_thresholds
        #self.subgoal_thresholds = subgoal_thresholds

        # Set inital state and goal state spaces
        self.initial_state_space = initial_state_space
        self.goal_space_train = goal_space_train
        self.goal_space_test = goal_space_test
        #self.subgoal_colors = ["Magenta","Green","Red","Blue","Cyan","Orange","Maroon","Gray","White","Black"]

        self.max_actions = max_actions

        # Implement visualization if necessary
        self.visualize = False  # Visualization boolean
        if self.visualize:
            self.viewer = MjViewer(self.sim)
        self.num_frames_skip = num_frames_skip


    def sparse_reward(self, states, end_goals):
        """
        计算批量奖励函数。
        
        参数:
            states: np.ndarray，形状为 (N, D)，表示多个 achieved goals。
            end_goals: np.ndarray，形状为 (N, D)，表示多个 desired goals。
            
        返回:
            np.ndarray，形状为 (N,)，每个 achieved goal 与对应的 desired goal 的奖励值。
        """
        rewards = np.zeros(states.shape[0])  # 初始化奖励数组
        project_state_to_end_goal = lambda state: state[:3]

        for i in range(states.shape[0]):  # 遍历每对 (state, end_goal)
            state = states[i]
            end_goal = end_goals[i]
            goal_achieved = True

            proj_end_goal = project_state_to_end_goal(state)
            for j in range(len(state)):  # 检查每个维度是否满足阈值
                if np.abs(end_goal[j] - proj_end_goal[j]) > self.end_goal_thresholds[j]:
                    goal_achieved = False
                    break

            if goal_achieved:
                rewards[i] = 10
                #print(f"Step {i}: End point is achieved, mission success")
            else:
                rewards[i] = -1

        return rewards

    def success(self, state, end_goal):
        goal_achieved = 1

        project_state_to_end_goal = lambda state: state[:3]
        proj_end_goal = project_state_to_end_goal(state)

        for j in range(len(proj_end_goal)):
            if (np.absolute(end_goal[j] - proj_end_goal[j]) > self.end_goal_thresholds[j]):
                goal_achieved = 0
                break

        return goal_achieved

    
    # Get state, which concatenates joint positions and velocities
    def get_state(self):

        if self.name == "pendulum.xml":
            return np.concatenate([np.cos(self.sim.data.qpos),np.sin(self.sim.data.qpos),
                               self.sim.data.qvel])
        else:
            return np.concatenate((self.sim.data.qpos, self.sim.data.qvel))

    # Reset simulation to state within initial state specified by user
    def reset_sim(self, next_goal):

        # Reset controls
        self.sim.data.ctrl[:] = 0

        if self.name == "ant_reacher.xml":
            while True:
                # Reset joint positions and velocities
                for i in range(len(self.sim.data.qpos)):
                    self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])

                for i in range(len(self.sim.data.qvel)):
                    self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])

                # Ensure initial ant position is more than min_dist away from goal
                min_dist = 8
                if np.linalg.norm(next_goal[:2] - self.sim.data.qpos[:2]) > min_dist:
                    break

        else:

            # Reset joint positions and velocities
            for i in range(len(self.sim.data.qpos)):
                self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])

            for i in range(len(self.sim.data.qvel)):
                self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])

        self.sim.step()

        # Return state
        return self.get_state()

    # Execute low-level action for number of frames specified by num_frames_skip
    def execute_action(self, action):

        self.sim.data.ctrl[:] = action
        for _ in range(self.num_frames_skip):
            self.sim.step()
            if self.visualize:
                self.viewer.render()

        return self.get_state()


    # Visualize end goal.  This function may need to be adjusted for new environments.
    def display_end_goal(self,end_goal):

        # Goal can be visualized by changing the location of the relevant site object.
        if self.name == "pendulum.xml":
            self.sim.data.mocap_pos[0] = np.array([0.5*np.sin(end_goal[0]),0,0.5*np.cos(end_goal[0])+0.6])
        elif self.name == "ur5.xml":

            theta_1 = end_goal[0]
            theta_2 = end_goal[1]
            theta_3 = end_goal[2]

            # shoulder_pos_1 = np.array([0,0,0,1])
            upper_arm_pos_2 = np.array([0,0.13585,0,1])
            forearm_pos_3 = np.array([0.425,0,0,1])
            wrist_1_pos_4 = np.array([0.39225,-0.1197,0,1])


            # Transformation matrix from shoulder to base reference frame
            T_1_0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.089159],[0,0,0,1]])

            # Transformation matrix from upper arm to shoulder reference frame
            T_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0],[np.sin(theta_1), np.cos(theta_1), 0, 0],[0,0,1,0],[0,0,0,1]])

            # Transformation matrix from forearm to upper arm reference frame
            T_3_2 = np.array([[np.cos(theta_2),0,np.sin(theta_2),0],[0,1,0,0.13585],[-np.sin(theta_2),0,np.cos(theta_2),0],[0,0,0,1]])

            # Transformation matrix from wrist 1 to forearm reference frame
            T_4_3 = np.array([[np.cos(theta_3),0,np.sin(theta_3),0.425],[0,1,0,0],[-np.sin(theta_3),0,np.cos(theta_3),0],[0,0,0,1]])

            # Determine joint position relative to original reference frame
            # shoulder_pos = T_1_0.dot(shoulder_pos_1)
            upper_arm_pos = T_1_0.dot(T_2_1).dot(upper_arm_pos_2)[:3]
            forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3]
            wrist_1_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(wrist_1_pos_4)[:3]

            joint_pos = [upper_arm_pos, forearm_pos, wrist_1_pos]

            """
            print("\nEnd Goal Joint Pos: ")
            print("Upper Arm Pos: ", joint_pos[0])
            print("Forearm Pos: ", joint_pos[1])
            print("Wrist Pos: ", joint_pos[2])
            """

            for i in range(3):
                self.sim.data.mocap_pos[i] = joint_pos[i]

        elif self.name == "ant_reacher.xml":
            self.sim.data.mocap_pos[0][:2] = np.copy(end_goal[:2])

        else:
            assert False, "Provide display end goal function in environment.py file"

    #Function returns an end goal
    def get_next_goal(self):

        end_goal = np.zeros((len(self.goal_space_test)))

        if self.name == "ur5.xml":

            goal_possible = False
            while not goal_possible:
                end_goal = np.zeros(shape=(self.end_goal_dim,))
                end_goal[0] = np.random.uniform(self.goal_space_test[0][0],self.goal_space_test[0][1])

                end_goal[1] = np.random.uniform(self.goal_space_test[1][0],self.goal_space_test[1][1])
                end_goal[2] = np.random.uniform(self.goal_space_test[2][0],self.goal_space_test[2][1])

                # Next need to ensure chosen joint angles result in achievable task (i.e., desired end effector position is above ground)

                theta_1 = end_goal[0]
                theta_2 = end_goal[1]
                theta_3 = end_goal[2]

                # shoulder_pos_1 = np.array([0,0,0,1])
                upper_arm_pos_2 = np.array([0,0.13585,0,1])
                forearm_pos_3 = np.array([0.425,0,0,1])
                wrist_1_pos_4 = np.array([0.39225,-0.1197,0,1])

                # Transformation matrix from shoulder to base reference frame
                T_1_0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.089159],[0,0,0,1]])

                # Transformation matrix from upper arm to shoulder reference frame
                T_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0],[np.sin(theta_1), np.cos(theta_1), 0, 0],[0,0,1,0],[0,0,0,1]])

                # Transformation matrix from forearm to upper arm reference frame
                T_3_2 = np.array([[np.cos(theta_2),0,np.sin(theta_2),0],[0,1,0,0.13585],[-np.sin(theta_2),0,np.cos(theta_2),0],[0,0,0,1]])

                # Transformation matrix from wrist 1 to forearm reference frame
                T_4_3 = np.array([[np.cos(theta_3),0,np.sin(theta_3),0.425],[0,1,0,0],[-np.sin(theta_3),0,np.cos(theta_3),0],[0,0,0,1]])

                forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3]
                wrist_1_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(wrist_1_pos_4)[:3]

                # Make sure wrist 1 pos is above ground so can actually be reached
                if np.absolute(end_goal[0]) > np.pi/4 and forearm_pos[2] > 0.05 and wrist_1_pos[2] > 0.15:
                    goal_possible = True



        # elif not test and self.goal_space_train is not None:
        #     for i in range(len(self.goal_space_train)):
        #         end_goal[i] = np.random.uniform(self.goal_space_train[i][0],self.goal_space_train[i][1])
        else:
            assert self.goal_space_test is not None, "Need goal space for testing. Set goal_space_test variable in \"design_env.py\" file"

            for i in range(len(self.goal_space_test)):
                end_goal[i] = np.random.uniform(self.goal_space_test[i][0],self.goal_space_test[i][1])

        return end_goal


'''
    # Visualize all subgoals
    def display_subgoals(self, subgoals):

        # Display up to 10 subgoals and end goal
        if len(subgoals) <= 11:
            subgoal_ind = 1
        else:
            subgoal_ind = len(subgoals) - 11

        for i in range(1, min(len(subgoals),11)):

            if self.name == "pendulum.xml":
                self.sim.data.mocap_pos[i] = np.array([0.5*np.sin(subgoals[subgoal_ind][0]),0,0.5*np.cos(subgoals[subgoal_ind][0])+0.6])
                # Visualize subgoal
                self.sim.model.site_rgba[i][3] = 1
                subgoal_ind += 1

            elif self.name == "ur5.xml":

                theta_1 = subgoals[subgoal_ind][0]
                theta_2 = subgoals[subgoal_ind][1]
                theta_3 = subgoals[subgoal_ind][2]

                # shoulder_pos_1 = np.array([0,0,0,1])
                upper_arm_pos_2 = np.array([0,0.13585,0,1])
                forearm_pos_3 = np.array([0.425,0,0,1])
                wrist_1_pos_4 = np.array([0.39225,-0.1197,0,1])

                # Transformation matrix from shoulder to base reference frame
                T_1_0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.089159],[0,0,0,1]])

                # Transformation matrix from upper arm to shoulder reference frame
                T_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0],[np.sin(theta_1), np.cos(theta_1), 0, 0],[0,0,1,0],[0,0,0,1]])

                # Transformation matrix from forearm to upper arm reference frame
                T_3_2 = np.array([[np.cos(theta_2),0,np.sin(theta_2),0],[0,1,0,0.13585],[-np.sin(theta_2),0,np.cos(theta_2),0],[0,0,0,1]])

                # Transformation matrix from wrist 1 to forearm reference frame
                T_4_3 = np.array([[np.cos(theta_3),0,np.sin(theta_3),0.425],[0,1,0,0],[-np.sin(theta_3),0,np.cos(theta_3),0],[0,0,0,1]])

                # Determine joint position relative to original reference frame
                # shoulder_pos = T_1_0.dot(shoulder_pos_1)
                upper_arm_pos = T_1_0.dot(T_2_1).dot(upper_arm_pos_2)[:3]
                forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3]
                wrist_1_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(wrist_1_pos_4)[:3]

                joint_pos = [upper_arm_pos, forearm_pos, wrist_1_pos]

                """
                print("\nSubgoal %d Joint Pos: " % i)
                print("Upper Arm Pos: ", joint_pos[0])
                print("Forearm Pos: ", joint_pos[1])
                print("Wrist Pos: ", joint_pos[2])
                """

                # Designate site position for upper arm, forearm and wrist
                for j in range(3):
                    self.sim.data.mocap_pos[3 + 3*(i-1) + j] = np.copy(joint_pos[j])
                    self.sim.model.site_rgba[3 + 3*(i-1) + j][3] = 1

                # print("\nLayer %d Predicted Pos: " % i, wrist_1_pos[:3])

                subgoal_ind += 1

            elif self.name == "ant_reacher.xml":
                self.sim.data.mocap_pos[i][:2] = np.copy(subgoals[subgoal_ind][:2])
                self.sim.model.site_rgba[i][3] = 1

                subgoal_ind += 1

            else:
                # Visualize desired gripper position, which is elements 18-21 in subgoal vector
                self.sim.data.mocap_pos[i] = subgoals[subgoal_ind]
                # Visualize subgoal
                self.sim.model.site_rgba[i][3] = 1
                subgoal_ind += 1
'''
