import tensorflow as tf
import tqdm
import numpy as np
import gym
import pickle
import os
import tf_util
import load_policy
import os

TASK_LIST = [
    "Ant-v2",
    "HalfCheetah-v2",
    "Hopper-v2",
    "Humanoid-v2",
    "Reacher-v2",
    "Walker2d-v2",
]

def data_gather(learningpara):

    print("Gathering Data")
    policy_fn = load_policy.load_policy("experts/{}.pkl".format(learningpara['env']))
    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(learningpara['env'])
        max_steps = env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(learningpara['rollouts']):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if learningpara['render']:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)
        expert_data = {
        'actions':np.array(actions),
        'observation':np.array(observations),
        'rewards':np.array(totalr)
        }
        try:
            os.mkdir('data/{}'.format(learningpara['env']))
        except FileExistsError:
            pass
        with open(learningpara['path'],'wb') as f:
            pickle.dump(expert_data,f)
        return expert_data

def configuration_load(env_name):
    config = {
    "env": env_name,
    "max": 1000000,
    "rollouts":100,
    "epochs":30,
    "render": False,
    "path": 'data/{}/{}.pkl'.format(env_name,env_name)
    }
    return config

def data_run():

    data = {}
    for name in TASK_LIST:
        config = configuration_load(name)
        expert = data_gather(config)
        data['name'] = expert
    pickle.dump(data,open('data/expert_big.pkl','w'))
if __name__ == '__main__':
    data_run()
