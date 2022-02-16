import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

class TemporalLayer(nn.Linear):
    def __init__(self, in_features, out_features, epsilon=1.0e-12, increase_rate=1e-6):
        super(TemporalLayer, self).__init__(in_features, out_features, False)
        self.epsilon = epsilon
        self.infTime = 1.0e6  # Pseudo spike time for a neuron that does not spike
        self.increase_rate = increase_rate

    def reset_parameters(self, varianceScale=1.0):
        low = -1.0 / self.in_features
        high = 4.0 / self.in_features

        self.weight.data.normal_().mul_((high - low) * varianceScale).add_((high + low) / 2)

    def forward(self, input):

        spikeWeights = self.weight

        sortedSpikeTimes, sortedSpikeIndices = input.sort()

        batchOutputTimes = []

        input_size = input.size(0)
        for k in range(input_size):
            sortedSpikeWeights = spikeWeights[:, sortedSpikeIndices[k]] # Spikes considered in order
            cumulativeWeights = sortedSpikeWeights.cumsum(1)

            numerator = (sortedSpikeWeights * sortedSpikeTimes[k].view(1, -1)).cumsum(1) # 分子
            denominator = cumulativeWeights - 1  # 1 is the firing threshold # 分母

            outputTimes_all = torch.zeros_like(denominator) + self.infTime #初始化
            denominatorIsStable = (denominator.abs() > self.epsilon) # For numerical stability
            outputTimes_all[denominatorIsStable.data] \
                = numerator[denominatorIsStable.data] / denominator[denominatorIsStable.data] #更新

            delayedSpikes = F.pad(sortedSpikeTimes[k][1:], (0, 1), value=self.infTime)
            condition_1 = (outputTimes_all < delayedSpikes.view(1, -1))
            condition_2 = (outputTimes_all > 0)
            condition_3 = (cumulativeWeights > 1)
            contribution_mask = condition_1 * condition_2 * condition_3 #####

            infinity_spikes = (contribution_mask.sum(1) == 0)

            contributionPoint = contribution_mask.cpu().max(1)[1]

            outputTimes = outputTimes_all[np.arange(self.out_features), contributionPoint]
            outputTimes[infinity_spikes.data] = 1/self.increase_rate

            batchOutputTimes.append(outputTimes)

        return torch.stack(batchOutputTimes)