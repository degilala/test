import numpy as np
import torch
import pickle
from collections import deque
"""
这是一个用于实现经验回放的类ReplayMemory，用于在强化学习中存储和采样记忆。经验回放是一种强化学习中的重要技术，旨在减少样本之间的相关性和噪声，以更好地训练深度神经网络。
ReplayMemory类的构造函数__init__有三个参数：max_size用于指定经验回放缓冲区的最大长度，look_forward_steps用于指定每个记忆的向前步数，gamma是折扣因子，用于计算未来奖励的折扣价值。
类中还定义了__len__方法，用于返回经验回放缓冲区中的样本数量。store方法用于将记忆存储到缓冲区中，而sample方法用于从缓冲区中随机采样指定数量的记忆。
在sample方法中，先随机选择batch_size个记忆，然后根据look_forward_steps和gamma计算每个记忆的折扣奖励。最后返回采样的记忆，其中状态、动作、奖励、
下一个状态和完成标志分别存储在不同的列表中，可以使用numpy进行拼接。hiddens和hiddens_是用于存储神经网络中间状态的列表，但是在这个代码中没有使用。
"""
class ReplayMemory:
    def __init__(self, max_size=300, look_forward_steps=5, gamma=0.9,save_path='F:\peizhun\SPAC-Deformable-Registration-main\code\memory/append.txt'):
        self.buffer = deque(maxlen=max_size)
        self.look_forward_steps = look_forward_steps
        self.gamma = gamma
        self.save_path = save_path

    def __len__(self):
        return len(self.buffer)

    def store(self, memory):
        self.buffer.append(memory)
        #if self.save_path is not None:
         #   with open(self.save_path,'wb') as f:
         #       pickle.dump(self.buffer,f)
         #       f.close()
    def load(self):
        if self.save_path is not None:
            with open(self.save_path,'rb') as f:
                self.buffer = pickle.load(f)

    def sample(self, batch_size):
       # if self.save_path is not None:
        #    self.load()
        sample_indices = np.random.choice(range(len(self.buffer)), batch_size, replace=False)

        ss, aa, rr, ss_, dd = [], [], [], [], []
        hh, hh_ = [], []
        max_index = len(self.buffer)
        for i in sample_indices:
            s = self.buffer[i].s
            a = self.buffer[i].a
            r = self.buffer[i].r
            s_ = self.buffer[i].s_
            d = self.buffer[i].d

            for step in range(self.look_forward_steps):
                if max_index > i + step:
                    r += self.gamma * self.buffer[i + step].r
                else:
                    break

            ss.append(s)
            aa.append(a)
            rr.append(r)
            ss_.append(s_)
            dd.append(d)

        hiddens = None
        hiddens_ = None

        return (np.concatenate(ss, axis=0),
                np.concatenate(aa, axis=0),
                np.array(rr),
                np.concatenate(ss_, axis=0),
                np.array(dd))




