import numpy as np
from tensorboardX import SummaryWriter

class Summary():
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.episode_step_count = []
        self.episode_mean_value = []
        self.episode_rewards = []
        self.writer = SummaryWriter(logdir=self.log_dir)

    def iter_steps(self):
        return len(self.episode_step_count)

    def add_info(self,
                 episode_step_count,
                 episode_value,
                 episode_rewards):
        self.episode_step_count.append(episode_step_count)
        self.episode_mean_value.append(np.sum(episode_value)/float(episode_step_count))
        self.episode_rewards.append(episode_rewards)
        if len(self.episode_step_count) >= 10:
            self.write_info()
            """
            这段代码定义了一个类（class）中的一个方法（method）"add_info"。它的作用是将一个新的 "episode" 的信息（episode_step_count，episode_value 和 episode_rewards）添加到当前类的对象中。

具体来说，它会将 episode_step_count 添加到一个列表中（self.episode_step_count），将 episode_value 的平均值添加到另一个列表中（self.episode_mean_value），并将 episode_rewards 添加到第三个列表中（self.episode_rewards）。

在添加信息后，它会检查 self.episode_step_count 的长度是否大于等于 10，如果是，则调用一个名为 "write_info" 的方法来写入这些信息。这部分代码被省略，但可以推断出 "write_info" 方法将会使用 self.episode_step_count、self.episode_mean_value 和 self.episode_rewards 列表的数据来生成某种信息或输出。

总体来说，这段代码旨在记录和存储每个 "episode" 的信息，并在特定条件下进行处理。
            """

    def write_info(self):

        mean_step = np.mean(self.episode_step_count[-5:])
        mean_value = np.mean(self.episode_mean_value[-5:])
        mean_reward = np.mean(self.episode_rewards[-5:])

        self.writer.add_scalar(tag='performance_reward', scalar_value=float(mean_reward), global_step=self.iter_steps())
        self.writer.add_scalar(tag='performance_length', scalar_value=float(mean_step), global_step=self.iter_steps())
        self.writer.add_scalar(tag='performance_value', scalar_value=float(mean_value), global_step=self.iter_steps())

        # self.writer.close()

    def close(self):
        self.writer.close()




