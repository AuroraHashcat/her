import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import TensorDataset, DataLoader
import random



def gen_net(in_size=1, out_size=1, H=256, n_layers=3, activation='none'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    elif activation == 'softmax':
        net.append(nn.Softmax())
    elif activation == 'relu':
        net.append(nn.ReLU())

    return net


class RewardModel:
    def __init__(self, ds, da, dg, cfg,
                 lr=3e-5, size_segment=1, max_size_eff=1000,max_size_nor=1000, activation='none', capacity=5e5,

                 # reward model parameters
                 reward_model_layers=3,
                 reward_model_hidden=128,
                 weight_decay = 1e-5
                 ):

        # state dim, state dim included goal dim
        self.ds = ds

        # action dim
        self.da = da

        # goal dim
        self.dg = dg

        self.cfg = cfg

        # reward model parameters
        self.reward_model_H = reward_model_hidden
        self.reward_model_layers = reward_model_layers
        self.activation = activation
        self.lr = lr
        self.paramlst = []
        self.opt = None
        self.weight_decay = weight_decay

        # construct reward model network
        self.reward_model = None
        self.construct_reward_predict()

        self.max_size_nor = max_size_nor # 1e4
        self.max_size_eff = max_size_eff # 1e4
        self.CEloss = nn.CrossEntropyLoss()
        self.BCEloss = nn.BCEWithLogitsLoss()# 自带一个sigmoid计算
        self.train_batch_size = 128

        self.capacity = int(capacity)
        self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds + self.da), dtype=np.float32)
        self.buffer_seg2 = np.empty((self.capacity, size_segment, self.ds + self.da), dtype=np.float32)
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False

        # separate data
        self.eff_inputs = []
        self.eff_labels = []
        self.eff_targets = []

        self.nor_inputs = []
        self.nor_targets = []
        self.nor_labels = []

        # concat data
        self.inputs = []
        self.labels = []

    def construct_reward_predict(self):
        model = nn.Sequential(*gen_net(in_size=self.ds + self.da + self.dg,
                                       out_size=1, H=self.reward_model_H, n_layers=self.reward_model_layers,
                                       activation=self.activation)).float().to(self.cfg.cuda)

        self.reward_model = model
        self.paramlst.extend(model.parameters())

        self.opt = torch.optim.Adam(self.paramlst, lr=self.lr, weight_decay=self.weight_decay)

    def construct_pbrl_data(self):
        """
        Construct preference-based dataset.

        Parameters:
        - eff_inputs: effective transitions collected from pretrain phase (list or array)
        - nor_inputs: invalid transitions collected from interaction with env (list or array)

        Returns:
        - dataset: transition pairs and corresponding label
        """

        # Extract the actual arrays from the lists
        # ndarray(5366,39)
        # # original code
        # eff_transitions = self.eff_inputs[0]  # (5366, 39)
        # print("eff_transitions_shape", eff_transitions.shape)
        # nor_transitions = self.nor_inputs[0]  # (5366, 39)
        # print("nor_transitions_shape", nor_transitions.shape)

        # print(len(self.eff_inputs))
        # print(len(self.nor_inputs))

        # 假设我们从每个输入列表中随机采样3个子列表
        num_samples = 10

        while True:
            eff_indices = random.sample(range(len(self.eff_inputs)), num_samples)
            # nor_indices = random.sample(range(len(self.nor_inputs)), num_samples)
            if len(self.nor_inputs) < num_samples:
                print(f"len(self.nor_inputs) < num_samples, sampling all {len(self.nor_inputs)} elements instead.")
                nor_indices = list(range(len(self.nor_inputs)))  # 选择所有索引
            else:
                nor_indices = random.sample(range(len(self.nor_inputs)), num_samples)
            print("eff_indices:", eff_indices, "nor_indices:", nor_indices)

            # 根据 indices 采样子列表
            eff_sampled_sublists = [self.eff_inputs[i] for i in eff_indices]
            nor_sampled_sublists = [self.nor_inputs[i] for i in nor_indices]

            # 将所有子列表拼接成一个大的列表
            eff_transitions = []
            for sublist in eff_sampled_sublists:
                eff_transitions.extend(sublist)

            nor_transitions = []
            for sublist in nor_sampled_sublists:
                nor_transitions.extend(sublist)

            eff_transitions = np.array(eff_transitions)
            nor_transitions = np.array(nor_transitions)

            # 输出结果
            print("eff_transitions_shape:", eff_transitions.shape)
            print("nor_transitions_shape:", nor_transitions.shape)

            # 检查 eff_transitions 的长度是否大于 batch_size
            if eff_transitions.shape[0] >= 128:
                break  # 满足条件时退出循环
            else:
                print("eff_transitions length is less than 128. Resampling...")

        # eff_transitions 和 nor_transitions 满足条件，后续处理逻辑
        print("Final eff_transitions_shape:", eff_transitions.shape)
        print("Final nor_transitions_shape:", nor_transitions.shape)

        # Create pairwise comparisons between effective and invalid transitions
        transition_pairs = []
        labels = []

        # Define the different preference probabilities setting
        # 设置噪声的范围
        noise_range = (-0.05, 0.05)

        if self.cfg.strict_label:
            prob_eff = 1.0 + np.random.uniform(*noise_range)
            prob_nor = 0.0 + np.random.uniform(*noise_range)
        elif self.cfg.smooth_label:
            print("using smooth_label")
            prob_eff = 0.9 + np.random.uniform(*noise_range)
            prob_nor = 0.1 + np.random.uniform(*noise_range)
        elif self.cfg.smooth_label_info:
            print("using smooth_label_info")
            prob_eff = 0.8 + np.random.uniform(*noise_range)
            prob_nor = 0.2 + np.random.uniform(*noise_range)

        # 确保 prob_eff 和 prob_nor 的和不超过 1 或小于 0
        prob_eff = np.clip(prob_eff, 0.0, 1.0)  # 保证 prob_eff 在[0, 1]范围内
        prob_nor = np.clip(prob_nor, 0.0, 1.0)  # 保证 prob_nor 在[0, 1]范围内

        # 确保 prob_eff + prob_nor = 1, 处理噪声带来的波动
        if prob_eff + prob_nor > 1.0:
            prob_eff = 1.0 - prob_nor
        elif prob_eff + prob_nor < 1.0:
            prob_nor = 1.0 - prob_eff


        if self.cfg.random_data:
            if self.cfg.celoss:
                # Use a single loop to match corresponding elements from eff_transitions and nor_transitions
                for eff_tran, nor_tran in zip(eff_transitions, nor_transitions):
                    # Create pairs
                    if np.random.choice([True, False]):
                        # 有效过渡在前
                        transition_pairs.append([eff_tran, nor_tran])
                        labels.append([prob_eff])  # 有效过渡对应标签 1
                    else:
                        # 无效过渡在前
                        transition_pairs.append([nor_tran, eff_tran])
                        labels.append([prob_nor])  # 无效过渡对应标签 1

            elif self.cfg.bceloss:
                for eff_tran, nor_tran in zip(eff_transitions, nor_transitions):
                    # Create pairs
                    if np.random.choice([True, False]):
                        # 有效过渡在前
                        transition_pairs.append([eff_tran, nor_tran])
                        labels.append([prob_eff, prob_nor])  # 有效过渡对应标签 1
                    else:
                        # 无效过渡在前
                        transition_pairs.append([nor_tran, eff_tran])
                        labels.append([prob_nor, prob_eff])  # 无效过渡对应标签 1
        else:
            if self.cfg.celoss:
                # Use a single loop to match corresponding elements from eff_transitions and nor_transitions
                for eff_tran, nor_tran in zip(eff_transitions, nor_transitions):
                    # Create pairs
                    # 有效过渡在前
                    transition_pairs.append([eff_tran, nor_tran])
                    labels.append([prob_eff])  # 有效过渡对应标签 1

            elif self.cfg.bceloss:
                for eff_tran, nor_tran in zip(eff_transitions, nor_transitions):
                    # Create pairs
                    # 有效过渡在前
                    transition_pairs.append([eff_tran, nor_tran])
                    labels.append([prob_eff, prob_nor])  # 有效过渡对应标签 1

        # Convert transition pairs and labels to numpy arrays or tensors as needed
        transition_pairs = np.array(transition_pairs)
        labels = np.array(labels)
        # print(transition_pairs.shape)#(5366, 2, 39)
        # print(labels.shape)#(5366, 1)

        # Return the dataset as a dictionary
        dataset = {
            'transition_pairs': transition_pairs,
            'labels': labels
        }
        print("len_transition_pairs", len(transition_pairs))
        print("len_labels", len(labels))

        # print("dataset:", dataset)

        return dataset

    def add_eff_data_sga(self, obs, act, goal, rew, done):
        # concat state, goal, action
        # ndarray
        sga_t = np.concatenate([obs, goal, act], axis=-1)
        r_t = rew

        # construct flat data: flat_input and flat_target
        flat_input = sga_t.reshape(1, self.da + self.ds + 2) # (1,s+g+a),ds included sg
        # print("flat_input.shape:", flat_input.shape) (1,39)
        r_t = np.array(r_t)
        flat_target = r_t.reshape(1, 1) # (1,1)
        # print("flat_target.shape:", flat_target.shape) (1,1)

        init_data = len(self.eff_inputs) == 0
        if init_data:
            self.eff_inputs.append(flat_input)
            self.eff_targets.append(flat_target)
        elif done:
            # print("对有效数据进行归档，当前序列的数据存储完毕")
            # print( "sagrd:", obs, act, goal, rew, done)
            self.eff_inputs[-1] = np.concatenate([self.eff_inputs[-1], flat_input])
            self.eff_targets[-1] = np.concatenate([self.eff_targets[-1], flat_target])
            # print("当前的eff的数据", self.eff_inputs)
            # print("当前的eff的数据的长度", len(self.eff_inputs))
            # FIFO
            if len(self.eff_inputs) > self.max_size_eff: # 1e4
                self.eff_inputs = self.eff_inputs[1:]
                self.eff_targets = self.eff_targets[1:]
            self.eff_inputs.append([])
            self.eff_targets.append([])
        else:
            # concat append, make sure there is only one list[0], which has a ndarray
            # list
            if len(self.eff_inputs[-1]) == 0:
                self.eff_inputs[-1] = flat_input
                self.eff_targets[-1] = flat_target
            else:
                self.eff_inputs[-1] = np.concatenate([self.eff_inputs[-1], flat_input])
                self.eff_targets[-1] = np.concatenate([self.eff_targets[-1], flat_target])


    def add_nor_data_sga(self, obs, act, goal, rew, done):
        sa_t = np.concatenate([obs, goal, act], axis=-1)
        r_t = rew

        flat_input = sa_t.reshape(1, self.da + self.ds + self.dg)
        r_t = np.array(r_t)
        flat_target = r_t.reshape(1, 1)

        init_data = len(self.nor_inputs) == 0
        if init_data:
            self.nor_inputs.append(flat_input)
            self.nor_targets.append(flat_target)
        elif done:
            print("对无效数据进行归档，当前序列的数据存储完毕")
            self.nor_inputs[-1] = np.concatenate([self.nor_inputs[-1], flat_input])
            self.nor_targets[-1] = np.concatenate([self.nor_targets[-1], flat_target])
            # print("当前的nor的数据", self.nor_inputs)
            print("当前的nor的数据的长度", len(self.nor_inputs))
            # FIFO
            if len(self.nor_inputs) > self.max_size_nor: # 1e4
                self.nor_inputs = self.nor_inputs[1:]
                self.nor_targets = self.nor_targets[1:]
            self.nor_inputs.append([])
            self.nor_targets.append([])
        else:
            if len(self.nor_inputs[-1]) == 0:
                self.nor_inputs[-1] = flat_input
                self.nor_targets[-1] = flat_target
            else:
                self.nor_inputs[-1] = np.concatenate([self.nor_inputs[-1], flat_input])
                self.nor_targets[-1] = np.concatenate([self.nor_targets[-1], flat_target])



    def r_hat(self, x):
        # the network parameterizes r hat in eqn 1 from the paper
        # return the reward prediction value
        logits = self.reward_model(torch.from_numpy(x).float().to(self.cfg.cuda))
        # probs = torch.sigmoid(logits) # [0,1]
        # r_hat = torch.tanh(logits).detach().cpu().numpy() # [-1,1]
        r_hat = torch.sigmoid(logits).detach().cpu().numpy()-1 # [-1,0]
        return r_hat

    def r_hat_batch(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        logits = self.reward_model(torch.from_numpy(x).float().to(self.cfg.cuda))
        # probs = torch.sigmoid(logits) # [0,1]
        # r_hat = torch.tanh(logits).detach().cpu().numpy()  # [-1,1]
        r_hat = torch.sigmoid(logits).detach().cpu().numpy()-1  # [-1,0]
        return r_hat

    def train_reward_model_celoss(self, dataset, batch_size=128, epochs=100, patience=10):

        # Convert transitions and labels into PyTorch tensors
        transition_pairs = torch.tensor(dataset['transition_pairs'], dtype=torch.float32).to(self.cfg.cuda)  # (5366, 2, 39)
        labels_ = torch.tensor(dataset['labels'], dtype=torch.long).to(self.cfg.cuda)  # (5366,)
        # print("transition_pair.shape", transition_pairs.shape) [5366,2,39]
        # print("labels.shape", labels.shape) [5366,1]

        # Create reward model
        reward_model = self.reward_model

        # best_loss = 0.20
        # epochs_no_improve = 0

        for epoch in range(epochs):
            # 初始化累计的loss和acc
            total_loss = 0.0
            total_acc = 0

            self.opt.zero_grad()
            print("Training reward model {} ".format(epoch))

            # sample data
            # index = np.random.choice(len(self.eff_inputs[0]), size=batch_size, replace=False)
            # eff_batch = [self.eff_inputs[0][idx] for idx in index]
            # nor_batch = [[self.eff_inputs[0][idx] for idx in index]]

            # Create DataLoader for batch data
            data = TensorDataset(transition_pairs, labels_)
            data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

            for batch_transition_pairs, batch_labels in data_loader:

                sga_1 = batch_transition_pairs[:,0]
                sga_2 = batch_transition_pairs[:,1]
                labels = batch_labels
                labels = labels.view(labels.size(0))
                # print("labels:", labels)


                # labels = torch.ones(len(sga_1)).long().to(self.self.cfg.device)
                # print("labels_shape:", labels.shape)
                # print(sga_1.shape) # (128，39)
                # print(sga_2.shape) # (128，39)
                # print(labels.shape) # [128]

                # 获取logits
                r_hat1 = reward_model(sga_1)
                r_hat2 = reward_model(sga_2)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                # print("r_hat_1", r_hat1)
                # print("r_hat_2", r_hat2)
                # print("r_hat1_sum", r_hat1)
                # print("r_hat2_sum", r_hat2)

                # 增加一个新的维度
                r_hat1 = r_hat1.unsqueeze(1)  # 形状变为 (128, 1)
                r_hat2 = r_hat2.unsqueeze(1)  # 形状变为 (128, 1)

                # 使用 torch.concat 进行拼接
                r_hat = torch.cat((r_hat1, r_hat2), dim=1)
                # print("r_hat_shape:", r_hat.shape)

                # 计算loss
                curr_loss = self.CEloss(r_hat, labels)
                total_loss += curr_loss.item()  # 累加当前loss
                # print("reward model loss: {}".format(curr_loss.item()))

                # 计算准确率
                # print("r_hat.data:", r_hat.data)
                # print("r_hat.data.shape:", r_hat.data.shape)

                _, predicted = torch.max(r_hat.data, 1)
                # print("predicted:", predicted)  # 每一行最大值的索引
                correct = (predicted == labels).sum().item()
                # print("correct:", correct)
                acc = correct / labels.size(0)
                total_acc += acc  # 累加正确的数量
                # print("reward model acc: {}".format(correct))

                # 反向传播和优化
                curr_loss.backward()
                self.opt.step()

            avg_acc = total_acc / len(data_loader)
            avg_loss = total_loss / len(data_loader)
            print(
            f'Epoch: [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, '
            f'Acc: {avg_acc:.4f} ')

            # Early stopping
            # if avg_loss < best_loss:
            #     best_loss = avg_loss
            #     epochs_no_improve = 0
            #     # 可以保存模型
            #     # torch.save(reward_model.state_dict(), 'best_reward_model.pth')
            # else:
            #     epochs_no_improve += 1
            #
            # if epochs_no_improve >= patience:
            #     print("Early stopping triggered!")
            #     break

        return curr_loss, acc  # 返回总的正确数量和总的loss

    def train_reward_model_bceloss(self, dataset, reward_model_path, epochs=100, batch_size=128):
        """
        Train the reward model based on preference data.

        Parameters:
        - dataset: a dictionary containing 'transitions' and 'labels'
        - input_dim: dimension of input (state, goal, action concatenated)
        - epochs: number of training epochs
        - batch_size: batch size for training
        - lr: learning rate for optimizer
        """

        # Create reward model
        reward_model = self.reward_model
        # Convert transitions and labels into PyTorch tensors
        transition_pairs = torch.tensor(dataset['transition_pairs'], dtype=torch.float32).to(self.cfg.cuda)  # (N, 2, 39)
        labels = torch.tensor(dataset['labels'], dtype=torch.float32).to(self.cfg.cuda)  # (N,)

        # Create DataLoader for batching
        data = TensorDataset(transition_pairs, labels)
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

        # Define loss function and optimizer
        criterion = self.BCEloss  # Binary classification for pairwise preference (1 or 0)
        # optimizer = optim.Adam(reward_model.parameters(), lr=lr)
        optimizer = self.opt

        # Training loop with early stopping
        best_loss = float('inf')  # Initialize best_loss
        patience = 5  # Number of epochs to wait before stopping
        trigger_times = 0  # Counter for early stopping
        save_interval = 10

        # training loop
        for epoch in range(epochs):
            total_loss = 0.0
            acc = 0.0
            reward_mean_value = 0.0
            # print("Training ... {} ".format(epoch))

            for batch_transition_pairs, batch_labels in data_loader:
                # print("batch_transition_pairs shape:", batch_transition_pairs.shape) # (128,2,39)
                # print("batch_labels shape:", batch_labels.shape) # (128,2)
                optimizer.zero_grad()  # 清零梯度

                # Forward pass: compute rewards for both transitions in the pair
                rewards_1 = reward_model(batch_transition_pairs[:, 0])
                rewards_2 = reward_model(batch_transition_pairs[:, 1])
                # print("reward_1:", rewards_1.shape) #[128,1]
                # print("reward_2:", rewards_2.shape)

                # get logits
                r_hat1 = rewards_1.sum(axis=1)
                r_hat2 = rewards_2.sum(axis=1)
                if torch.any(torch.isnan(r_hat1)) or torch.any(torch.isinf(r_hat2)):
                    print("r_hat contains NaN or Inf")

                # Concatenate logits for BCE
                r_hat = torch.cat((r_hat1.unsqueeze(1), r_hat2.unsqueeze(1)), dim=1)
                # print("r_hat",r_hat)

                # Check that r_hat and labels are compatible for loss calculation
                if r_hat.shape[0] != batch_labels.shape[0]:
                    print(f"Shape mismatch: r_hat shape {r_hat.shape}, labels shape {batch_labels.shape}")

                labels = batch_labels
                # print("labels",labels)
                # print("labels_shape",labels.shape)

                # Compute loss
                curr_loss = criterion(r_hat, labels)
                total_loss += curr_loss.item()  # 仅加上当前损失的标量值

                # Compute acc
                # print("r_hat_data:", r_hat.data)
                _, predicted = torch.max(r_hat.data, 1)
                _, true_indices = torch.max(labels, dim=1)
                # print("predicted:", predicted)
                # print("true_indices", true_indices)
                correct = (predicted == true_indices).sum().item()
                # print("corrent: {}".format(correct))
                # print("labels.shape[0]", labels.shape[0])
                curr_acc = correct / labels.shape[0]
                acc += curr_acc

                # Backward pass and optimize
                curr_loss.backward()  # 反向传播
                optimizer.step()  # 更新参数

                # 记录奖励总和
                # print(f"true_indices: {true_indices}")
                # print(f"r_hat.data.shape: {r_hat.data.shape}")

                reward_sum = r_hat.data[true_indices].sum()
                # print("reward_sum: {}".format(reward_sum))
                corr_r_mean = reward_sum / labels.shape[0]
                # print("corr_r_mean: {}".format(corr_r_mean))
                reward_mean_value += corr_r_mean

            acc = acc / len(data_loader)
            loss = total_loss / len(data_loader)
            reward = reward_mean_value / len(data_loader)
            # if acc > 0.97:
            #     print("acc > 0.97, skip update reward model")
            #     break;
            # print(f'Epoch: [{epoch + 1}/{epochs}], Loss: {loss:.4f}, Acc: {acc:.4f} ,Reward: {reward:.4f}')

            # Early stopping logic
            if loss < best_loss:
                best_loss = loss
                trigger_times = 0  # Reset counter
                # Optionally, save the model
                # self.save(reward_model_path)
            else:
                trigger_times += 1

            if trigger_times >= patience:
                print("Early stopping...")
                break  # Exit training loop if no improvement

                # Periodic model saving
            if (epoch + 1) % save_interval == 0:
                print(f"Saving model at epoch {epoch + 1}")
                self.save(reward_model_path)  # Save model periodically

        return reward_model

    def save(self, path):
        # os.makedirs(model_dir, exist_ok=True)  # 确保目录存在
        # model_path = os.path.join(model_dir, 'reward_model.pt')  # 使用 os.path.join 处理路径
        # torch.save(self.reward_model.state_dict(), model_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Ensure model is on CPU before saving
        self.reward_model.cpu()
        torch.save(self.reward_model.state_dict(), path)
        # Optionally, move it back to GPU
        self.reward_model.to(self.cfg.cuda)
        # print("the reward model is saved")


    def load(self, path):
        # model_path = os.path.join(model_dir, 'reward_model.pt')  # 使用 os.path.join 处理路径
        # if os.path.exists(model_path):  # 检查文件是否存在
        #     self.reward_model.load_state_dict(torch.load(model_path))
        # else:
        #     raise FileNotFoundError(f"Model file not found: {model_path}")
        state_dict = torch.load(path, map_location=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'))
        self.reward_model.load_state_dict(state_dict)
        print("the pre trained reward model is loaded")

