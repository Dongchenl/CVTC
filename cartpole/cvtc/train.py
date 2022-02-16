import gym
import numpy as np

import torch
import torch.optim as optim
from model_dqn import QNet
from memory import Memory
from scipy import stats
from tensorboardX import SummaryWriter

from config import env_name, INITIAL_EXPLORATION, batch_size, \
    update_target, GOAL_SCORE, log_interval, device, replay_memory_capacity, lr


def coding_state(s_in):
    cs = torch.ones([4*20])
    cs *= 6

    s = s_in
    s[0] *= 20
    s[1] *= 2
    s[2] *= 20
    s[3] *= 2

    for i in range(4):
        for j in range(20):
            cs[j+i*20] -= stats.norm.pdf(j-9.5, s[i], 1.4) * 20

    #print(s[0]*100+50, s[1]*20+150, s[2]*100+250, s[3]*20+350)
    #cs[int(s[0]*20+4):int(s[0]*20+6)] /= 6
    #cs[int(s[1]*2+14):int(s[1]*2+16)] /= 6
    #cs[int(s[2]*20+24):int(s[2]*20+26)] /= 6
    #cs[int(s[3]*2+34):int(s[3]*2+36)] /= 6
    #print(cs)
    return cs


def get_action(state, target_net, epsilon, env):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    else:
        return target_net.get_action(state)


def update_target_model(online_net, target_net):
    # Target <- Net
    target_net.load_state_dict(online_net.state_dict())


def main():
    env = gym.make(env_name)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0] * 20
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    # initialize networks
    online_net = QNet(num_inputs, num_actions)
    target_net = QNet(num_inputs, num_actions)
    # Load trained model: online_net.load_state_dict(torch.load('model.pkl'))

    update_target_model(online_net, target_net)

    optimizer = optim.Adam(online_net.parameters(), lr=lr)
    writer = SummaryWriter('logs')

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()

    memory = Memory(replay_memory_capacity)
    running_score = 0
    epsilon = 1.0
    steps = 0
    loss = 0
    log_steps = 1000

    for e in range(30000):
        done = False

        score = 0
        state = env.reset()
        state = coding_state(state)
        state = torch.Tensor(state).to(device)
        state = state.unsqueeze(0)

        while not done:
            steps += 1

            env.render()

            # Get the network output
            action = get_action(state, target_net, epsilon, env)

            # Get and encode the network output
            next_state, reward, done, _ = env.step(action)
            next_state = coding_state(next_state)
            next_state = torch.Tensor(next_state).to(device)
            next_state = next_state.unsqueeze(0)
            action_one_hot = np.zeros(2)
            action_one_hot[action] = 1
            reward = reward if not done else -1  # or score == 499
            mask = 0 if done else 1

            # Write into memory pool
            memory.push(state, next_state, action_one_hot, reward, mask)

            score += reward
            state = next_state

            if steps > INITIAL_EXPLORATION:
                epsilon -= 0.00005
                epsilon = max(epsilon, 0.1)

                batch = memory.sample(batch_size)
                loss = QNet.train_model(online_net, target_net, optimizer, batch)

                if steps % update_target == 0:
                    update_target_model(online_net, target_net)

            if steps % log_steps == 0:
                print('{} episode | step {} |score: {:.2f} | epsilon: {:.2f}'.format(
                    e, steps, running_score, epsilon), flush=True)

        # score = score if score == 500.0 else score + 1
        running_score = 0.99 * running_score + 0.01 * score

        if e % log_interval == 0:
            writer.add_scalar('log/score', float(running_score), e)
            writer.add_scalar('log/loss', float(loss), e)

        if running_score > GOAL_SCORE:
            break
        if running_score > 180:
            torch.save(online_net.state_dict(), 'model' + str(e) + '.pkl')


if __name__ == "__main__":
    main()
