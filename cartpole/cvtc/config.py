import torch

env_name = 'CartPole-v0'
gamma = 0.99
batch_size = 32
lr = 0.001
INITIAL_EXPLORATION = 1000
GOAL_SCORE = 200
log_interval = 10
update_target = 100
replay_memory_capacity = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_layer_size = 128
