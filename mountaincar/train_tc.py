import random
import gym
import numpy as np
from collections import deque
# import tensorflow as tf
import os

from model_temporal_tc import QNet
from memory import Memory
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

env = gym.make('MountainCar-v0')

state_size = env.observation_space.shape[0]

action_size = env.action_space.n

batch_size = 32
n_episodes = 70000
output_dir = 'model_output/MountainCar'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

from scipy import stats


def CodingState(s):
    cs = torch.ones([state_size])
    cs *= 6

    s[0] *= 30
    s[1] *= 500

    for i in range(2):
        for j in range(20):
            cs[j + i * 20] -= min(stats.norm.pdf(j - 9.5, s[i], 1.4) * 15, 5.)

    # print(s[0]*100+50, s[1]*20+150, s[2]*100+250, s[3]*20+350)
    # cs[int(s[0]*20+4):int(s[0]*20+6)] /= 6
    # cs[int(s[1]*2+14):int(s[1]*2+16)] /= 6
    # cs[int(s[2]*20+24):int(s[2]*20+26)] /= 6
    # cs[int(s[3]*2+34):int(s[3]*2+36)] /= 6
    # print(cs)
    return cs


def update_target_model(online_net, target_net):
    # Target <- Net
    target_net.load_state_dict(online_net.state_dict())


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size

        self.memory = Memory(200000)

        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_decay = .85
        self.epsilon_min = 0.00001

        self.learning_rate = 0.001251
        self.model = QNet(state_size, action_size)
        self.target_model = QNet(state_size, action_size)
        self.model.cuda()
        self.target_model.cuda()

        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def update_target_model(self):
        update_target_model(self.model, self.target_model)

    def remember(self, state, next_state, action, reward, done):
        # state = torch.Tensor(state).cuda()
        self.memory.push(state, next_state, action, reward, done)

    def act(self, state):
        if np.random.rand(1) <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.forward(state).cpu().detach().numpy()
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = self.memory.sample(batch_size)
        # print(minibatch)
        states = []
        targets = []

        for state, next_state, action, reward, done in minibatch:
            target = reward
            if not done:
                # print(self.target_model.forward(next_state)[0])
                V = self.target_model.forward(next_state)
                target = (reward + self.gamma * V[0].max())
            target_f = self.model.forward(state)
            target_f[0][action] = target
            states.append(state[0].unsqueeze(0))
            targets.append(target_f[0].unsqueeze(0))

        # print(torch.cat(states, dim=0))
        pred = self.model(torch.cat(states, dim=0)).squeeze(1)
        #print(pred[0], torch.cat(targets, dim=0).detach()[0], action)
        loss = F.mse_loss(pred, torch.cat(targets, dim=0).detach())


        weightSumTarget = 1.3
        l2Coeff = 0.001
        regCoeff = 100

        temporalLayerWeights = [x.weight for x in self.model.children() if isinstance(x, nn.Linear)]
        weightSumCost = torch.tensor(0.0).cuda()
        weightL2Cost = torch.tensor(0.0).cuda()
        for weight in temporalLayerWeights:
            weightSumCost += ((F.relu(weightSumTarget - weight.sum(1))) ** 2).sum()
            weightL2Cost += l2Coeff * ((weight.sum(1)) ** 2).sum()

        loss += (weightSumCost + weightL2Cost) * regCoeff
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

    def load(self, name):
        self.model = torch.load('./model.pkl')

    def save(self, name):
        torch.save(self.model, './model.pkl')


agent = DQNAgent(state_size, action_size)

done = False
counter = 0
# scores_memory= deque(maxlen=100)
memory = Memory(200000)
steps = 0
scores_memory = deque(maxlen=100)

optimizer = optim.Adam(agent.model.parameters(), lr=0.001251)

for e in range(n_episodes):
    state = env.reset()

    # state= np.reshape(state, [1, state_size])
    # state = CodingState(state)
    state = torch.Tensor(state).to('cuda')
    state = state.unsqueeze(0)

    print('train')
    for time in range(7000):
        steps += 1
        # if e % 50==0:
        #    env.render()
        action = agent.act(state)
        next_state, reward, done, halp = env.step(action)

        # next_state = np.reshape(next_state, [1, state_size])

        # next_state = CodingState(next_state)
        next_state = torch.Tensor(next_state).to('cuda')

        next_state = next_state.unsqueeze(0)

        action_one_hot = np.zeros(3)
        action_one_hot[action] = 1
        agent.remember(state, next_state, action, reward, done)

        if steps > 32:
            agent.replay(batch_size)

            # epsilon -= 0.00005
            # epsilon = max(epsilon, 0.1)

            # batch = memory.sample(batch_size)
            # loss = QNet.train_model(agent.model, agent.target_model, optimizer, batch)

        state = next_state

        if done:
            scores_memory.append(time)
            scores_avg = np.mean(scores_memory) * -1

            print(
                'episode: {}/{}, score: {}, e {:.2}, help: {}, reward: {}, 100score avg: {}'.format(e, n_episodes, time,
                                                                                                    agent.epsilon,
                                                                                                    state, reward,
                                                                                                    scores_avg),
                flush=True)

            break
    agent.update_target_model()

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    if e % 50 == 0:
        agent.save(output_dir + 'weights_final' + '{:04d}'.format(e) + ".hdf5")
