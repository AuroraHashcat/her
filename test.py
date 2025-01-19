import gym
env = gym.make('FetchReach-v1')

obs = env.reset()
done = False

while (not done):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(reward)