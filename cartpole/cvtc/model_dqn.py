import torch
import torch.nn as nn
import torch.nn.functional as F

from temporal_layer import TemporalLayer
from config import gamma
from config import hidden_layer_size

class QNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(QNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        #self.fc1 = nn.Linear(num_inputs, 128)
        self.fc1 = TemporalLayer(num_inputs, hidden_layer_size)
        self.fc2 = TemporalLayer(hidden_layer_size, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        #x = F.relu(self.fc1(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        qvalue = self.fc2(x)
        return qvalue

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.Tensor(batch.action).float().cuda()
        rewards = torch.Tensor(batch.reward).cuda()
        masks = torch.Tensor(batch.mask).cuda()

        pred = online_net(states).squeeze(1)
        next_pred = target_net(next_states).squeeze(1)

        pred = torch.sum(pred.mul(actions), dim=1)

        target = rewards + masks * gamma * next_pred.max(1)[0]

        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, input):
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)
        return action.cpu().numpy()[0]
