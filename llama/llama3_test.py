from llama.customized_call import ChatWithOurServer
import time
import re
import json
#from env.design_env import design_env
import numpy as np
# new
# client = ChatWithOurServer(base_url="http://172.18.36.112:65431/v1", model='Meta-Llama-3-8B-Instruct')
# jiangyue
client = ChatWithOurServer(base_url="http://172.18.36.112:65433/v1", model='Llama-3.1-8B-Instruct')

# chenxi
# client = ChatWithOurServer(base_url = "http://0.0.0.0:65433/v1", model='Llama-3.1-8B-Instruct')


def generator_four(obs):

    xml_content = """
     <body name="east_wall" pos="8.25 0 1">
      <geom type="box" size="0.25 8.25 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>
    <body name="north_wall" pos="0 8.25 1">
      <geom type="box" size="8.5 0.25 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>
    <body name="west_wall" pos="-8.25 0 1">
      <geom type="box" size="0.25 8.25 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>
    <body name="south_wall" pos="0 -8.25 1">
      <geom type="box" size="8.5 0.25 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>

    <body name="center_vertical" pos="0 0 1">
      <geom type="box" size="0.5 3.5 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>
    <body name="center_horizontal" pos="0 0 1">
      <geom type="box" size="3.5 0.5 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>
    <body name="east_stick_out" pos="7.25 0 1">
      <geom type="box" size="0.75 0.5 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>
    <body name="north_stick_out" pos="0 7.25 1">
      <geom type="box" size="0.5 0.75 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>
    <body name="west_stick_out" pos="-7.25 0 1">
      <geom type="box" size="0.75 0.5 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>
    <body name="south_stick_out" pos="0 -7.25 1">
      <geom type="box" size="0.5 0.75 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>

        The task is to select subgoals that can be reached sequentially from the start point to the end point 
        while avoiding the obstacles: four 'stick_out', which the agent can't pass through.
        Ensure that the subgoals do not lead the agent into the 'stick_out' obstacles.
        The subgoals should form a coherent path that guides the agent safely around the obstacles and towards the goal platform.
        """

    background = [{"role": "system",
        "content": """
            You are an expert AI assistant that solve the subgoal-chosen problem in a goal-conditioned task step by step. 
            The agent need get to the end point from the start point.
            The task is to select subgoals that can be reached sequentially according to the distance and obstacle positions.
            Rules for generating subgoals:
            1. The first subgoal should be close enough, making it extremely easy for the agent to reach it.
            2. The number of subgoals should depend on the distance between the starting point and the end point.
            3. The maximum number of subgoals should not exceed 10.
            4. Attention: The response must be in the format of a nested list, like [subgoal_1, subgoal_2, ...], with no other explanation or analysis.
            """
    }]

    task = [{"role":"system",
    "content":"The environment information is descripted as the xml:" + xml_content }]

        

    user_0 = [{"role": "user", "content": "start point:[5. 3.] end point:[-6. 5.]"}]
    user_1 = [{"role": "user", "content": "start point:[-2. 3.] end point:[4. 5.]"}]
    user_2 = [{"role": "user", "content": "start point:[7. -5.] end point:[-3. 7.]"}]

    assistant_0 = [{"role": "assistant", "content": """[[4,3],[1,3],[1,5],[-4,5]]"""}]
    assistant_1 = [{"role": "assistant", "content": """[[-1, 3],[-1, 5],[2, 5]]"""}]
    assistant_2 = [{"role": "assistant", "content": """[[5,-5],[0, -5],[-4, -5],[-5, 0],[-5, 7]]"""}]

    user = [{"role": "user", "content": obs}]
    messages = background + task + user_0 + user_1 + user_2 + assistant_0 + assistant_1 + assistant_2 + user
    chat_completion = client.create(
        messages=messages,
        max_new_tokens=500,
        top_p=0.9,
    )

    return  messages, chat_completion

def generator_reacher(obs):

    xml_content = """
    <body name="east_wall" pos="12 0 0.4">
      <geom type="box" size="0.25 12 0.4" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>
    <body name="north_wall" pos="0 12 0.4">
      <geom type="box" size="12.25 0.25 0.4" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>
    <body name="west_wall" pos="-12 0 0.4">
      <geom type="box" size="0.25 12 0.4" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>
    <body name="south_wall" pos="0 -12 0.4">
      <geom type="box" size="12.25 0.25 0.4" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>"""
    background = [{"role": "system",
    "content": """
    You are an expert AI assistant that solve the subgoal-chosen problem in a goal-conditioned task step by step. 
            The agent need get to the end point from the start point.
            The task is to select subgoals that can be reached sequentially according to the distance.
            Rules for generating subgoals:
            1. The first subgoal should be close enough, making it extremely easy for the agent to reach it.
            2. The number of subgoals should depend on the distance between the starting point and the end point.
            3. The maximum number of subgoals should not exceed 10.
            4. The response should be in the format of a nested list, like [[subgoal_1], [subgoal_2], ...], with no other explanation or analysis.
            """
    }]

    task = [{"role":"system",
    "content":"The environment information is descripted as the xml:" + xml_content }]

    user_0 = [{"role": "user", "content": "start point:[5. 3.] end point:[9. 5.]"}]
    user_1 = [{"role": "user", "content": "start point:[-2. 3.] end point:[11. 5.]"}]
    user_2 = [{"role": "user", "content": "start point:[7. -5.] end point:[-11. 7.]"}]

    assistant_0 = [{"role": "assistant", "content": """[[6, 4],[8, 5]]"""}]
    assistant_1 = [{"role": "assistant", "content": """[[-1, 4],[3, 4],[6, 5],[10, 5]]"""}]
    assistant_2 = [{"role": "assistant", "content": """[[6, -4],[3, -1],[0, 3],[-5, 5],[-10, 7]]"""}]

    user = [{"role": "user", "content": obs}]
    messages = background + task + user_0 + assistant_0 + user_1 + assistant_1 + user_2 + assistant_2 + user
    chat_completion = client.create(
        messages=messages,
        max_new_tokens=500,
        top_p=0.9,
    )

    return  messages, chat_completion

def generator_s(obs):

    xml_content = """
    <body name="east_wall" pos="8.25 0 1">
      <geom type="box" size="0.25 8.25 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>
    <body name="north_wall" pos="0 8.25 1">
      <geom type="box" size="8.5 0.25 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>
    <body name="west_wall" pos="-8.25 0 1">
      <geom type="box" size="0.25 8.25 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>
    <body name="south_wall" pos="0 -8.25 1">
      <geom type="box" size="8.5 0.25 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>
    <body name="east_stick_in" pos="2.75 3 1">
      <geom type="box" size="5.25 0.5 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>
    <body name="west_stick_in" pos="-2.75 -3 1">
      <geom type="box" size="5.25 0.5 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>
    The obstacles to avoid are 'east_stick_in' and 'west_stick_in'. The agent cannot pass through these obstacles."""

    background = [{"role": "system",
    "content": """
            You are an AI assistant specialized in generating subgoal sequences for goal-conditioned tasks.
            Your task is to create a series of subgoals that guide an agent from a starting point to an endpoint,
            while avoiding specific obstacles. The agent must navigate through an environment described by the following XML:
            """ + xml_content + """
            The subgoals should form a coherent path that safely guides the agent around these obstacles and towards the endpoint.
            The subgoals should be:
            1. Sequential and reachable.
            2. Close enough to each other to make them achievable in a reasonable number of steps.
            3. Not more than 10 in total.
            4. Provided in a nested list format, e.g., [[x1, y1], [x2, y2], ...].
            No additional explanation or analysis is needed.
            """
    }]

    user_0 = [{"role": "user", "content": "start point:[6, 6] end point:[-6, -6]"}]
    user_1 = [{"role": "user", "content": "start point:[6, 6] end point:[-6, -6]"}]
    user_2 = [{"role": "user", "content": "start point:[6, 6] end point:[-6, -6]"}]

    assistant_0 = [{"role": "assistant", "content": """[[5, 5],[0, 5],[-4, 2],[0,0],[4,0],[5,-3],[0,-5]]"""}]
    assistant_1 = [{"role": "assistant", "content": """[[4, 5],[-4, 3],[-3, 1],[0,0],[4,-2],[2,-5],[-3,-5]]"""}]
    assistant_2 = [{"role": "assistant", "content": """[[5, 6],[0, 4],[-5, 2],[0, 0],[4, -1],[5,-3][0,-5],[-4,-5]]"""}]

    user = [{"role": "user", "content": obs}]
    messages = background  + user_0 + assistant_0 + user_1 + assistant_1 + user_2 + assistant_2 + user
    chat_completion = client.create(
        messages=messages,
        max_new_tokens=500,
        top_p=0.9,
    )

    return  messages, chat_completion

def generator_w(obs):

    xml_content = """
    walls:
     <body name="east_wall" pos="8.25 0 1">
      <geom type="box" size="0.25 8.25 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>
    <body name="north_wall" pos="0 8.25 1">
      <geom type="box" size="8.5 0.25 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>
    <body name="west_wall" pos="-8.25 0 1">
      <geom type="box" size="0.25 8.25 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>
    <body name="south_wall" pos="0 -8.25 1">
      <geom type="box" size="8.5 0.25 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>

  obstacles:
    <body name="east_stick_in" pos="1 3 1">
      <geom type="box" size="3 0.5 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>

    <body name="west_stick_in" pos="1 -3 1">
      <geom type="box" size="3 0.5 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>

    <body name="north_stick_in" pos="-2.5 0 1">
      <geom type="box" size="0.5 3.5 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>

    <body name="single_stick_in" pos="-5.5 0 1">
      <geom type="box" size="2.5 0.5 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
    </body>
        """

    obstacle_details = (
        "Obstacles to avoid are detailed below:\n"
        "- 'east_stick_in': positioned at (1, 3, 1) with dimensions 3x0.5x1\n"
        "- 'west_stick_in': positioned at (1, -3, 1) with dimensions 3x0.5x1\n"
        "- 'north_stick_in': positioned at (-2.5, 0, 1) with dimensions 0.5x3.5x1\n"
        "- 'single_stick_in': positioned at (-5.5, 0, 1) with dimensions 2.5x0.5x1\n"
    )
 
    background = [{"role": "system",
        "content": """You are an AI assistant specialized in generating subgoal sequences for goal-conditioned tasks.
            Your task is to create a series of subgoals that guide an agent from a starting point to an endpoint,
            while avoiding the specified obstacles. The agent must navigate through an environment described by the following XML:
            """ + xml_content +  """
            The subgoals should form a coherent path that safely guides the agent around these obstacles and towards the endpoint.
            **Critical**: The generated subgoals must not intersect with any of the listed obstacles:"""
            + obstacle_details + """
            The subgoals should be:
            1. Sequential and reachable.
            2. Close enough to each other to make them achievable in a reasonable number of steps.
            3. Not more than 10 in total.
            4. Provided in a nested list format, e.g., [[x1, y1], [x2, y2], ...].
            No additional explanation or analysis is needed."""
    }]


    user_0 = [{"role": "user", "content": "start point:[-6,6] end point:[-6,-6]"}]
    user_1 = [{"role": "user", "content": "start point:[-6,6] end point:[-6,-6]"}]
    user_2 = [{"role": "user", "content": "start point:[-6,6] end point:[-6,-6]"}]

    assistant_0 = [{"role": "assistant", "content": """[[-5,5],[-1,5],[4,5],[6,3],[6,0],[6,-3],[2,-6],[-4,-6]]"""}]
    assistant_1 = [{"role": "assistant", "content": """[[-5,4],[-1,4],[4,4],[6,1],[5,-4],[2,5],[-3,-6]]"""}]
    assistant_2 = [{"role": "assistant", "content": """[[-4,5],[1,5],[3,5],[5,4],[5,-1],[4,-4],[-1,-5]]"""}]

    user = [{"role": "user", "content": obs}]
    messages = background + user_0 + assistant_0 + user_1 + assistant_1 + user_2  + assistant_2 + user
    chat_completion = client.create(
        messages=messages,
        max_new_tokens=500,
        top_p=0.9,
    )

    return  messages, chat_completion

def generator_o1(obs):

    xml_content = """
        <body name="east_wall" pos="8.25 0 1">
          <geom type="box" size="0.25 8.25 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
        </body>
        <body name="north_wall" pos="0 8.25 1">
          <geom type="box" size="8.5 0.25 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
        </body>
        <body name="west_wall" pos="-8.25 0 1">
          <geom type="box" size="0.25 8.25 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
        </body>
        <body name="south_wall" pos="0 -8.25 1">
          <geom type="box" size="8.5 0.25 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
        </body>

        <body name="obstacle1" pos="2 5 1">
          <geom type="box" size="1 0.5 1" contype="1" conaffinity="1" rgba="0.2 0.2 0.7 1" />
        </body>
        <body name="obstacle2" pos="4 -5 1">
          <geom type="box" size="1 0.5 1" contype="1" conaffinity="1" rgba="0.2 0.7 0.2 1" />
        </body>
        <body name="obstacle3" pos="-4 2 1">
          <geom type="box" size="0.5 1.5 1" contype="1" conaffinity="1" rgba="0.7 0.7 0.2 1" />
        </body>
        <body name="obstacle4" pos="1 0 1">
          <geom type="box" size="0.5 1 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
        </body>
        """

    obstacle_details = (
        "Obstacles to avoid are detailed below:\n"
        "- 'obstacle1': positioned at (2,5,1) with dimensions 1x0.5x1\n"
        "- 'obstacle2': positioned at (4,-5,1) with dimensions 1x0.5x1\n"
        "- 'obstacle3': positioned at (-4,2,1) with dimensions 0.5x1.5x1\n"
        "- 'obstacle4': positioned at (1, 0, 1) with dimensions 0.5x1x1\n"
    )

    background = [{"role": "system",
        "content": """You are an AI assistant specialized in generating subgoal sequences for goal-conditioned tasks.
            Your task is to create a series of subgoals that guide an agent from a starting point to an endpoint,
            while avoiding the specified obstacles. The agent must navigate through an environment described by the following XML:
            """ + xml_content +  """
            The subgoals should form a coherent path that safely guides the agent around these obstacles and towards the endpoint.
            **Critical**: The generated subgoals must not intersect with any of the listed obstacles:"""
            + obstacle_details + """
            The subgoals should be:
            1. Sequential and reachable.
            2. Close enough to each other to make them achievable in a reasonable number of steps.
            3. Not more than 10 in total.
            4. Provided in a nested list format, e.g., [[x1, y1], [x2, y2], ...].
            No additional explanation or analysis is needed."""
    }]

       

    user_0 = [{"role": "user", "content": "start point:[-6,-6] end point:[5, 5]"}]
    user_1 = [{"role": "user", "content": "start point:[-6,-6] end point:[5, 5]"}]
    user_2 = [{"role": "user", "content": "start point:[-6,-6] end point:[5, 5]"}]

    assistant_0 = [{"role": "assistant", "content": """[[-6,-5],[-6,0],[-5,3],[-2,6],[2,7]]"""}]
    assistant_1 = [{"role": "assistant", "content": """[[-4,-4],[-2,0],[1,2],[4,4]]"""}]
    assistant_2 = [{"role": "assistant", "content": """[[-2,-4],[2,-2],[4,0],[5,4]]"""}]

    user = [{"role": "user", "content": obs}]
    messages = background+ user_0 +  assistant_0 + user_1 + assistant_1 + user_2 + assistant_2 + user
    chat_completion = client.create(
        messages=messages,
        max_new_tokens=500,
        top_p=0.9,
    )

    return  messages, chat_completion

def generator_o2(obs):

    xml_content = """
        <body name="east_wall" pos="8.25 0 1">
          <geom type="box" size="0.25 8.25 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
        </body>
        <body name="north_wall" pos="0 8.25 1">
          <geom type="box" size="8.5 0.25 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
        </body>
        <body name="west_wall" pos="-8.25 0 1">
          <geom type="box" size="0.25 8.25 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
        </body>
        <body name="south_wall" pos="0 -8.25 1">
          <geom type="box" size="8.5 0.25 1" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 1" />
        </body>

        <!-- Obstacles -->
        <body name="obstacle1" pos="2 4 1">
          <geom type="box" size="0.5 1 1" contype="1" conaffinity="1" rgba="0.2 0.7 0.2 1" />
        </body>
        <body name="obstacle2" pos="-3 -4 1">
          <geom type="box" size="1 0.5 1" contype="1" conaffinity="1" rgba="0.7 0.2 0.7 1" />
        </body>
        <body name="obstacle3" pos="-6 1 1">
          <geom type="box" size="1 1 1" contype="1" conaffinity="1" rgba="0.7 0.7 0.2 1" />
        </body>
        <body name="obstacle4" pos="4 -4 1">
          <geom type="box" size="0.5 1 1" contype="1" conaffinity="1" rgba="0.2 0.2 0.7 1" />
        </body>
        <body name="obstacle5" pos="1 0 1">
          <geom type="box" size="1 0.5 1" contype="1" conaffinity="1" rgba="0.2 0.7 0.2 1" />
        </body>
        """

    obstacle_details = (
        "Obstacles to avoid are detailed below:\n"
        "- 'obstacle1': positioned at (2,4,1) with dimensions 0.5x1x1\n"
        "- 'obstacle2': positioned at (3,-4,1) with dimensions 1x0.5x1\n"
        "- 'obstacle3': positioned at (-6,1,1) with dimensions 1x1x1\n"
        "- 'obstacle4': positioned at (4,-4, 1) with dimensions 0.5x1x1\n"
        "- 'obstacle5': positioned at (1,0,1) with dimensions 1x0.5x1\n"
    )

    background = [{"role": "system",
        "content": """You are an AI assistant specialized in generating subgoal sequences for goal-conditioned tasks.
            Your task is to create a series of subgoals that guide an agent from a starting point to an endpoint,
            while avoiding the specified obstacles. The agent must navigate through an environment described by the following XML:
            """ + xml_content +  """
            The subgoals should form a coherent path that safely guides the agent around these obstacles and towards the endpoint.
            **Critical**: The generated subgoals must not intersect with any of the listed obstacles:"""
            + obstacle_details + """
            The subgoals should be:
            1. Sequential and reachable.
            2. Close enough to each other to make them achievable in a reasonable number of steps.
            3. Not more than 10 in total.
            4. Provided in a nested list format, e.g., [[x1, y1], [x2, y2], ...].
            No additional explanation or analysis is needed."""
    }]

        

    user_0 = [{"role": "user", "content": "start point:[-6,-6] end point:[6, 6]"}]
    user_1 = [{"role": "user", "content": "start point:[-6,-6] end point:[5, 5]"}]
    user_2 = [{"role": "user", "content": "start point:[-6,-6] end point:[5, 5]"}]

    assistant_0 = [{"role": "assistant", "content": """[[-6,-5],[-6,-1],[-5,3],[-2,6],[2,7]]"""}]
    assistant_1 = [{"role": "assistant", "content": """[[-4,-3],[-2,0],[1,3],[4,4]]"""}]
    assistant_2 = [{"role": "assistant", "content": """[[-4,-6],[-1,-4],[2,-2],[4,0],[5,4]]"""}]

    user = [{"role": "user", "content": obs}]
    messages = background + user_0 + assistant_0 + user_1 + assistant_1 + user_2 + assistant_2 + user
    chat_completion = client.create(
        messages=messages,
        max_new_tokens=500,
        top_p=0.9,
    )

    return  messages, chat_completion

def check_quality(messages, last_response):
    # 将助手的回复追加到对话中
    messages.append({"role": "assistant", "content": last_response})

    # 添加新的用户输入
    user_input = """There might be an error in the subgoals above because of lack of understanding of the question. 
                    Please correct the error, check the quality of the generated subgoals and modify them if necessary.
                    The number of subgoals still shouldn't exceed 10.
                    JUST FOLLOW THE FORMAT OF THE FIRST GENERATED, WHICH OUTPUT A LIST OF SUBGOAL COORDINATE.
                    For example, output [[0,0],[1,1]]
                    NO OTHER WORDS. JUST LIST OF SUBGOAL COORDINATE"""
    messages.append({"role": "user", "content": user_input})

    # 发送请求
    chat_completion = client.create(
        messages=messages,
        max_new_tokens=500,
        top_p=0.9,
    )


    return chat_completion

if __name__ == '__main__':

    llm_input = "start point:[-6,6] end point:[-6, -6]"
    print(llm_input)

    start_time = time.time()
    messages, subgoal = generator_w(llm_input)
    print("before:",subgoal)

    subgoal = check_quality(messages, subgoal)
    end_time = time.time()
    print("query time: ", end_time - start_time)
    print("after:",subgoal)
    subgoal = json.loads(subgoal)
    print(len(subgoal))
    # 验证解析后的数据是否是嵌套列表格式
    # if isinstance(subgoal, list) and all(isinstance(coord, list) and len(coord) == 2 for coord in subgoal):
    #     print("right")
    # else:
    #     raise ValueError("生成内容不是有效的嵌套列表格式")

