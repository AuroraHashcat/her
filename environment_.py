from tkinter import *
from tkinter import ttk
import time
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import gym

class Environment():

    def __init__(self, model_name, goal_space_train, goal_space_test, project_state_to_end_goal, end_goal_thresholds, initial_state_space, max_actions = 1200, num_frames_skip = 10, show = False):

        self.name = model_name

        # Create Mujoco Simulation
        self.model = load_model_from_path("/home/wuchenxi/projects/her/"+ model_name)
        self.sim = MjSim(self.model)

        # Set dimensions and ranges of states, actions, and goals in order to configure actor/critic networks
        self.state_dim = len(self.sim.data.qpos) + len(self.sim.data.qvel) # State will include (i) joint angles and (ii) joint velocities
        self.action_dim = len(self.sim.model.actuator_ctrlrange) # low-level action dim
        self.action_bounds_low = self.sim.model.actuator_ctrlrange[0][0]
        self.action_bounds_high = self.sim.model.actuator_ctrlrange[0][1] # low-level action bounds
        self.action_space = gym.spaces.Box(low=self.action_bounds_low, high=self.action_bounds_high, shape=(self.action_dim,), dtype=np.float32)
        self.end_goal_dim = len(goal_space_test)
 
        # Projection functions
        self.project_state_to_end_goal = project_state_to_end_goal


        # End goal/subgoal thresholds
        self.end_goal_thresholds = end_goal_thresholds

        # Set inital state and goal state spaces
        self.initial_state_space = initial_state_space
        self.goal_space_train = goal_space_train
        self.goal_space_test = goal_space_test

        self.max_actions = max_actions

        # Implement visualization if necessary
        self.visualize = show  # Visualization boolean
        if self.visualize:
            self.viewer = MjViewer(self.sim)
        self.num_frames_skip = num_frames_skip

    #her
    def sparse_reward(self, states, end_goals):

        rewards = np.zeros(states.shape[0])  # 初始化奖励数组
        # project_state_to_end_goal = self.project_state_to_end_goal

        for i in range(states.shape[0]):  # 遍历每对 (state, end_goal)
            state = states[i]
            end_goal = end_goals[i]
            goal_achieved = True

            # proj_end_goal = project_state_to_end_goal(self.sim,state)
            for j in range(len(state)):  # 检查每个维度是否满足阈值
                if np.abs(end_goal[j] - state[j]) > self.end_goal_thresholds[j]:
                    goal_achieved = False
                    break

            if goal_achieved:
                rewards[i] = 1
                #print(f"Step {i}: End point is achieved, mission success")
            else:
                rewards[i] = -1

        return rewards



    def dense_reward(self, states, end_goals):

        rewards = np.zeros(states.shape[0])
        if self.name == 'ant_obstacle_1.xml':
            wall_names = ["obstacle1","obstacle2","obstacle3","obstacle4"]
        elif self.name == 'ant_obstacle_2.xml':
            wall_names = ["obstacle1","obstacle2","obstacle3","obstacle4","obstacle5"]
        elif self.name == 'ant_s_shape.xml':
            wall_names = ["east_stick_in","west_stick_in"]
        elif self.name == 'ant_w_shape.xml':
            wall_names = ["east_stick_in","west_stick_in","north_stick_in","single_stick_in"]
        wall_geom_ids = [self.sim.model.geom_name2id(wall_name) for wall_name in wall_names]
        ant_geom_id = self.sim.model.geom_name2id('torso_geom')

        for i in range(states.shape[0]):
        # 权重和参数
            w_efficiency = 0.01     # 行为效率奖励的权重
            collision_penalty = -1 # 每次碰撞的固定惩罚

            state = states[i][:2]
            goal = end_goals[i]
            distance_to_goal = np.linalg.norm(state - goal)
            
            goal_reward = -distance_to_goal  # 距离越小，奖励越高

            collision_reward = 0
            for i in range(self.sim.data.ncon):  # 遍历所有接触点
                contact = self.sim.data.contact[i]
                if contact.geom1 == ant_geom_id and contact.geom2 in wall_geom_ids or \
                contact.geom2 == ant_geom_id and contact.geom1 in wall_geom_ids:
                    # 如果 Ant 与指定的墙体发生碰撞
                    collision_reward += collision_penalty
                    break  # 一次碰撞就足够，避免重复惩罚

            # (5) 行为效率奖励: 惩罚动作幅度过大的行为
            action_magnitude = np.linalg.norm(self.sim.data.ctrl)  # 控制输入的幅度
            efficiency_reward = -w_efficiency * action_magnitude

            # (6) 总奖励
            rewards[i] = goal_reward + collision_reward + efficiency_reward
        

        return rewards

    def success(self, state, end_goal):
        goal_achieved = 1

        project_state_to_end_goal = self.project_state_to_end_goal
        proj_end_goal = project_state_to_end_goal(self.sim,state)

        for j in range(len(proj_end_goal)):
            if (np.absolute(end_goal[j] - proj_end_goal[j]) > self.end_goal_thresholds[j]):
                goal_achieved = 0
                break

        return goal_achieved
        
    # Get state, which concatenates joint positions and velocities
    def get_state(self):
        return np.concatenate((self.sim.data.qpos, self.sim.data.qvel))

    def is_position_in_obstacle(self, position, obstacles):
        x, y = position
        for obstacle in obstacles:
            obstacle_x, obstacle_y, _ = obstacle['position']
            obstacle_width, obstacle_height, _ = obstacle['dimensions']
            
            # 计算障碍物的边界
            obstacle_x_min = int(obstacle_x)
            obstacle_x_max = int(obstacle_x + obstacle_width)
            obstacle_y_min = int(obstacle_y)
            obstacle_y_max = int(obstacle_y + obstacle_height)
            
            # 检查位置是否在障碍物内
            if (obstacle_x_min <= x < obstacle_x_max) and (obstacle_y_min <= y < obstacle_y_max):
                return True
        
        # 如果位置不在任何障碍物内，则返回False
        return False

    # Reset simulation to state within initial state specified by user
    def reset_sim(self, next_goal = None):

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

        elif self.name == "ant_four_rooms.xml":

            # Choose initial start state to be different than room containing the end goal

            # Determine which of four rooms contains goal
            goal_room = 0

            if next_goal[0] < 0 and next_goal[1] > 0:
                goal_room = 1
            elif next_goal[0] < 0 and next_goal[1] < 0:
                goal_room = 2
            elif next_goal[0] > 0 and next_goal[1] < 0:
                goal_room = 3


            # Place ant in room different than room containing goal
            # initial_room = (goal_room + 2) % 4


            initial_room = np.random.randint(0,4)
            while initial_room == goal_room:
                initial_room = np.random.randint(0,4)


            # Set initial joint positions and velocities
            for i in range(len(self.sim.data.qpos)):
                self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])

            for i in range(len(self.sim.data.qvel)):
                self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])

            # Move ant to correct room
            self.sim.data.qpos[0] = np.random.uniform(3,6.5)
            self.sim.data.qpos[1] = np.random.uniform(3,6.5)

            # If goal should be in top left quadrant
            if initial_room == 1:
                self.sim.data.qpos[0] *= -1

            # Else if goal should be in bottom left quadrant
            elif initial_room == 2:
                self.sim.data.qpos[0] *= -1
                self.sim.data.qpos[1] *= -1

            # Else if goal should be in bottom right quadrant
            elif initial_room == 3:
                self.sim.data.qpos[1] *= -1

            # print("Goal Room: %d" % goal_room)
            # print("Initial Ant Room: %d" % initial_room)

        elif self.name == "ant_s_shape.xml":

            self.sim.data.qpos[0] = 6
            self.sim.data.qpos[1] = 6
            self.sim.data.qpos[2] = np.random.uniform(self.initial_state_space[2][0],self.initial_state_space[2][1])

            for i in range(len(self.sim.data.qvel)):
                self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])

        elif self.name == "ant_w_shape.xml":
            self.sim.data.qpos[0] = -6
            self.sim.data.qpos[1] = 6
            self.sim.data.qpos[2] = np.random.uniform(self.initial_state_space[2][0],self.initial_state_space[2][1])

            for i in range(len(self.sim.data.qvel)):
                self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])

        elif self.name == "ant_obstacle_1.xml":
            while True:
                # Reset joint positions and velocities
                for i in range(len(self.sim.data.qpos)):
                    self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])
                position = [self.sim.data.qpos[0],self.sim.data.qpos[1]]
                obstacles = [
                        {'position': (2, 5, 1), 'dimensions': (1, 0.5, 1)},
                        {'position': (4, -5, 1), 'dimensions': (1, 0.5, 1)},
                        {'position': (-4, 2, 1), 'dimensions': (0.5, 1.5, 1)},
                        {'position': (1, 0, 1), 'dimensions': (0.5, 1, 1)}
                    ]
                min_dist = 6
                if np.linalg.norm(next_goal[:2] - self.sim.data.qpos[:2]) > min_dist and self.is_position_in_obstacle(position, obstacles) == False:
                    break
            
            for i in range(len(self.sim.data.qvel)):
                self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])

        elif self.name == "ant_obstacle_2.xml":
            while True:
                # Reset joint positions and velocities
                for i in range(len(self.sim.data.qpos)):
                    self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])
                position = [self.sim.data.qpos[0],self.sim.data.qpos[1]]
                obstacles = [
                        {'position': (2,4,1), 'dimensions': (0.5, 1, 1)},
                        {'position': (3, -4, 1), 'dimensions': (1, 0.5, 1)},
                        {'position': (-6, 1, 1), 'dimensions': (1, 1, 1)},
                        {'position': (4, -4, 1), 'dimensions': (0.5, 1, 1)},
                        {'position': (1, 0, 1), 'dimensions': (1, 0.5, 1)}
                    ]
                min_dist = 6
                if np.linalg.norm(next_goal[:2] - self.sim.data.qpos[:2]) > min_dist and self.is_position_in_obstacle(position, obstacles) == False:
                    break
            
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

        elif self.name == "ant_reacher.xml" or self.name == "ant_four_rooms.xml":
            self.sim.data.mocap_pos[0][:3] = np.copy(end_goal[:3])

        else:
            assert False, "Provide display end goal function in environment.py file"


    # Function returns an end goal
    def get_next_goal(self,test):

        end_goal = np.zeros((len(self.goal_space_test)))


        if self.name == "ant_four_rooms.xml":

            # Randomly select one of the four rooms in which the goal will be located
            room_num = np.random.randint(0,4)

            # Pick exact goal location
            end_goal[0] = np.random.uniform(3,6.5)
            end_goal[1] = np.random.uniform(3,6.5)
            end_goal[2] = np.random.uniform(0.45,0.55)

            # If goal should be in top left quadrant
            if room_num == 1:
                end_goal[0] *= -1

            # Else if goal should be in bottom left quadrant
            elif room_num == 2:
                end_goal[0] *= -1
                end_goal[1] *= -1

            # Else if goal should be in bottom right quadrant
            elif room_num == 3:
                end_goal[1] *= -1

        elif self.name == "ant_reacher.xml":
            for i in range(len(self.goal_space_train)):
                end_goal[i] = np.random.uniform(self.goal_space_train[i][0],self.goal_space_train[i][1])
                
        elif self.name == "ant_w_shape.xml" or self.name == "ant_s_shape.xml": 
            end_goal[0] = -6
            end_goal[1] = -6

        elif self.name == "ant_obstacle_1.xml":
            end_goal[0] = 5
            end_goal[1] = 5
        
        elif self.name == "ant_obstacle_1.xml":
            end_goal[0] = 6
            end_goal[1] = 6
        # Visualize End Goal
        # self.display_end_goal(end_goal)

        return end_goal