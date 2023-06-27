import os
import torch
import numpy as np
from trainers.base_trainer import BaseTrainer

from utils.scheduler import epsilon_scheduler
from utils.hparams import hparams
from utils.replay_buffer import ReplayBuffer
from utils.tb_logger import TensorBoardLogger
from utils.torch_utils import *
from utils.class_utils import *
from utils.checkpoint_utils import *
from utils.os_utils import *

import copy
import numba
import logging
import tqdm


@numba.njit
def decimal2binary_arr(x, length=10):
    h = []
    for _ in range(length):  # int类型的位置信息的二进制编码
        h.append(x % 2)
        x = x // 2
    return np.array(h)

@numba.njit
def binary_arr2decimal(bin_arr):
    length = len(bin_arr) # 二进制转为十进制编码
    x = 0
    for i in range(length):
        x += bin_arr[i] * 2 ** i
    return x

def get_trans_list(trans_type):
    two_d_arr = np.zeros((27, 27))
    # 用一维数组的值填充二维数组的圆形区域
    OBS_RANGE = 13
    idx = 0
    index_list = []
    for offset_x in range(-OBS_RANGE, OBS_RANGE):
        for offset_y in range(-OBS_RANGE, OBS_RANGE):
            if (offset_x ** 2 + offset_y ** 2) ** 0.5 <= OBS_RANGE:
                two_d_arr[13+offset_x, 13+offset_y] = idx
                idx += 1
    if trans_type == 0:
        rotated_arr = np.flip(two_d_arr, 1)
    elif trans_type == 1:
        rotated_arr = np.flip(two_d_arr, 0)
    elif trans_type == 2:
        rotated_arr = np.rot90(two_d_arr, -1)
    elif trans_type == 3:
        rotated_arr = np.rot90(two_d_arr, -2)
    elif trans_type ==4:
        rotated_arr = np.rot90(two_d_arr, -3)
    elif trans_type ==5:
        rotated_arr = np.rot90(two_d_arr, -4)
    else:
        raise Exception("变换类型未实现！")
    for offset_x in range(-OBS_RANGE, OBS_RANGE):
        for offset_y in range(-OBS_RANGE, OBS_RANGE):
            if (offset_x ** 2 + offset_y ** 2) ** 0.5 <= OBS_RANGE:
                index_list.append(int(rotated_arr[13+offset_x, 13+offset_y]))
    return index_list
                
#0表示水平镜面对称，1表示垂直镜面对称，2表示顺时针90°，3表示顺时针180°，4表示顺时针270°,5表示原来数据不做变换
def trans_obs(obs, trans_type,index_map_list, t_list): #改变obs的函数
    world_size = hparams["world_size"]
    x = binary_arr2decimal(obs[0:10])
    y = binary_arr2decimal(obs[10:20])
    map1 = obs[20:20+527]
    map2 = obs[20+527:20+527*2]
    map3 = obs[20+527*2:20+527*3]
    temp_map1 = np.copy(map1)
    temp_map2 = np.copy(map2)
    temp_map3 = np.copy(map3)
    v = obs[20+527*3:20+527*3+2]
    map1 = temp_map1[index_map_list]
    map2 = temp_map2[index_map_list]
    map3 = temp_map3[index_map_list]

    if trans_type == 0:
        new_x = x
        new_y = world_size[1] - y - 1
        obs[0:10] = decimal2binary_arr(new_x, length=10)
        obs[10:20] = decimal2binary_arr(new_y, length=10)
        new_vx = v[0] #* (-1)
        new_vy = v[1]  * (-1)
        obs[20+527*3] = new_vx
        obs[20+527*3+1] = new_vy
        map1[0] = t_list[0]
        map2[0] = t_list[1]
        map3[0] = t_list[2]
    elif trans_type == 1:
        new_x = world_size[0] - x - 1
        new_y = y
        obs[0:10] = decimal2binary_arr(new_x, length=10)
        obs[10:20] = decimal2binary_arr(new_y, length=10)
        new_vx = v[0] * (-1)
        new_vy = v[1] #* (-1)
        obs[20+527*3] = new_vx
        obs[20+527*3+1] = new_vy
        map1[251] = t_list[3]
        map2[251] = t_list[4]
        map3[251] = t_list[5]
    elif trans_type == 2:
        new_x = y
        new_y = world_size[0] - x - 1
        obs[0:10] = decimal2binary_arr(new_x, length=10)
        obs[10:20] = decimal2binary_arr(new_y, length=10)
        new_vx = v[1]
        new_vy = v[0] * (-1)
        obs[20+527*3] = new_vx
        obs[20+527*3+1] = new_vy
        map1[0] = t_list[3]
        map2[0] = t_list[4]
        map3[0] = t_list[5]
    elif trans_type == 3:
        new_x = world_size[0] - x - 1
        new_y = world_size[1] - y - 1
        obs[0:10] = decimal2binary_arr(new_x, length=10)
        obs[10:20] = decimal2binary_arr(new_y, length=10)
        new_vx = v[0] * (-1)
        new_vy = v[1] * (-1)
        obs[20+527*3] = new_vx
        obs[20+527*3+1] = new_vy
        map1[0] = t_list[0]
        map2[0] = t_list[1]
        map3[0] = t_list[2]
        map1[251] = t_list[3]
        map2[251] = t_list[4]
        map3[251] = t_list[5]
    elif trans_type == 4:
        new_x = world_size[1] - y - 1
        new_y = x
        obs[0:10] = decimal2binary_arr(new_x, length=10)
        obs[10:20] = decimal2binary_arr(new_y, length=10)
        new_vx = v[1] * (-1)
        new_vy = v[0]
        obs[20+527*3] = new_vx
        obs[20+527*3+1] = new_vy
        map1[251] = t_list[0]
        map2[251] = t_list[1]
        map3[251] = t_list[2]
    elif trans_type == 5:
        new_x = x
        new_y = y
        obs[0:10] = decimal2binary_arr(new_x, length=10)
        obs[10:20] = decimal2binary_arr(new_y, length=10)
        new_vx = v[0] 
        new_vy = v[1]
        obs[20+527*3] = new_vx
        obs[20+527*3+1] = new_vy
    else:
        raise Exception("变换类型未实现！")

    del temp_map1, temp_map2, temp_map3
    obs[20:20+527] = map1
    obs[20+527:20+527*2] = map2
    obs[20+527*2:20+527*3] = map3
    return obs

def trans_action(action, trans_type): #改变action的函数
    if trans_type == 0:
        action_map = [0,7,6,5,4,3,2,1,8,15,14,13,12,11,10,9,16]
    elif trans_type == 1:
        action_map = [4,3,2,1,0,7,6,5,12,11,10,9,8,15,14,13,16]
    elif trans_type == 2:
        action_map = [2,3,4,5,6,7,0,1,10,11,12,13,14,15,8,9,16]
    elif trans_type == 3:
        action_map = [4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11,16]
    elif trans_type == 4:
        action_map = [6,7,0,1,2,3,4,5,14,15,8,9,10,11,12,13,16]
    elif trans_type == 5:
        action_map = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    else:
        raise Exception("变换类型未实现！")
    return action_map[int(action)]

def trans_adj(adj, trans_type): #改变邻接关系的函数
    return adj

# 实现深拷贝
def deep_copy_complex_dict(d):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            new_dict[k] = deep_copy_complex_dict(v)  # 对于字典类型的值递归深拷贝
        elif isinstance(v, torch.Tensor):
            new_dict[k] = v.clone()  # 对于 PyTorch Tensor 的值使用 .clone() 方法
        else:
            new_dict[k] = copy.deepcopy(v)  # 对于其他类型的值使用 deepcopy 函数
    return new_dict

class ValueBasedTrainer(BaseTrainer):
    """
    This is the Main Controller for training a *Value-based* DRL algorithm.
    """

    def __init__(self):
        self.work_dir = os.path.join("checkpoints", hparams['exp_name'])
        os.makedirs(self.work_dir, exist_ok=True)
        self.log_dir = self.work_dir if 'log_dir' not in hparams else os.path.join(self.work_dir, hparams['log_dir'])
        os.makedirs(self.log_dir, exist_ok=True)

        self.env = get_cls_from_path(hparams['scenario_path'])()
        self.agent = get_cls_from_path(hparams['algorithm_path'])(self.env.obs_dim, self.env.act_dim).cuda()
        self.replay_buffer = ReplayBuffer()
        
        #保存变换数据的buffer
        self.trans_replay_buffer = ReplayBuffer()
        
        self.optimizer = torch.optim.Adam(self.agent.learned_model.parameters(), lr=hparams['learning_rate'])
        self.tb_logger = TensorBoardLogger(self.log_dir)

        self.i_iter_critic = 0
        self.i_episode = 0
        # Note that if word_dir already has config.yaml, it might override your manual setting!
        # So delete the old config.yaml when you want to do some modifications.
        self.load_from_checkpoint_if_possible()
        self.best_eval_reward = -1e15
        self.save_best_ckpt = False

    @property
    def i_iter_dict(self):
        return {'i_critic': self.i_iter_critic, 'i_episode': self.i_episode}

    def _load_i_iter_dict(self, i_iter_dict):
        self.i_iter_critic = i_iter_dict['i_critic']
        self.i_episode = i_iter_dict['i_episode']

    def _load_checkpoint(self, checkpoint):
        self.agent.load_state_dict(checkpoint['agent'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self._load_i_iter_dict(checkpoint['i_iter_dict'])
        logging.info("Checkpoint loaded successfully!")

    def load_from_checkpoint_if_possible(self):
        ckpt, ckpt_path = get_last_checkpoint(self.work_dir)
        if ckpt is None:
            logging.info("No checkpoint found, learn the agent from scratch!")
        else:
            logging.info(f"Latest checkpoint found at f{ckpt_path}, try loading...")
            try:
                self._load_checkpoint(checkpoint=ckpt)
            except:
                logging.warning("Checkpoint loading failed, now learn from scratch!")

    def save_checkpoint(self):
        # before save checkpoint, first delete redundant old checkpoints
        all_ckpt_path = get_all_ckpts(self.work_dir)
        if len(all_ckpt_path) >= hparams['num_max_keep_ckpt'] - 1:
            ckpt_to_delete = all_ckpt_path[hparams['num_max_keep_ckpt'] - 1:]
            remove_files(ckpt_to_delete)
        ckpt_path = os.path.join(self.work_dir, f"model_ckpt_episodes_{self.i_episode}.ckpt")
        checkpoint = {}
        checkpoint['agent'] = self.agent.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()
        checkpoint['i_iter_dict'] = self.i_iter_dict
        torch.save(checkpoint, ckpt_path)
        if self.save_best_ckpt:
            ckpt_path = os.path.join(self.work_dir, f"model_ckpt_best.ckpt")
            torch.save(checkpoint, ckpt_path)

    def _interaction_step(self, log_vars, map_list):
        # obs, adj = self.env.reset()
        obs, adj, total_t_list = self.env.reset()
        self.i_episode += 1
        epsilon = epsilon_scheduler(self.i_episode)
        self.tb_logger.add_scalars({'Epsilon': (self.i_episode, epsilon)})
        if hasattr(self.agent, 'reset_hidden_states'):
            self.agent.reset_hidden_states(obs.shape[0])
        tmp_reward_lst = []
        
        # 保存所有sample
        sample_list = []
        temp_t_list = []
        for t in range(hparams['episode_length']):
            action = self.agent.action(obs, adj, epsilon=epsilon, action_mode=hparams['training_action_mode'])
            temp_t_list.append(total_t_list)

            reward, next_obs, next_adj, done, total_t_list = self.env.step(action)
            sample = {'obs': obs, 'adj': adj, 'action': action, 'reward': reward, 'next_obs': next_obs,
                      'next_adj': next_adj, 'done': done}
            if hasattr(self.agent, 'get_hidden_states'):
                sample.update(self.agent.get_hidden_states())
                
            # 记录所有的sample
            sample_list.append(sample)
            # self.replay_buffer.push(sample)
            # self.trans_replay_buffer.push(sample)

            obs, adj = next_obs, next_adj
            tmp_reward_lst.append(sum(reward))
        temp_t_list.append(total_t_list)

        for index,trans_type in enumerate(hparams["data_trans_list"]):
            index_map_list = map_list[index]
            if hasattr(self.agent, 'get_hidden_states'):
                    self.agent.reset_hidden_states(sample_list[0]["obs"].shape[0])
            # for sample in sample_list:
            for j, sample in enumerate(sample_list):
                temp_sample = deep_copy_complex_dict(sample)

                for i in range(hparams["env_num_agent"]):
                    # temp_sample["obs"][i] = trans_obs(temp_sample["obs"][i], trans_type, index_map_list)
                    # temp_sample["action"][i] = trans_action(temp_sample["action"][i], trans_type)
                    # temp_sample["next_obs"][i] = trans_obs(temp_sample["next_obs"][i], trans_type, index_map_list)
                    temp_sample["obs"][i] = trans_obs(temp_sample["obs"][i], trans_type, index_map_list, temp_t_list[j][i])
                    temp_sample["action"][i] = trans_action(temp_sample["action"][i], trans_type)
                    temp_sample["next_obs"][i] = trans_obs(temp_sample["next_obs"][i], trans_type, index_map_list,  temp_t_list[j+1][i])

                if hasattr(self.agent, 'get_hidden_states'):
                    _ = self.agent.my_action(temp_sample["obs"], temp_sample["adj"])
                    return_dict = self.agent.get_hidden_states()
                    temp_sample["cri_hid"] = return_dict["cri_hid"]
                    temp_sample["next_cri_hid"] = return_dict["next_cri_hid"]

                self.trans_replay_buffer.push(temp_sample)
        del sample_list            
            
        log_vars['Interaction/episodic_reward'] = (self.i_episode, sum(tmp_reward_lst))
        if hasattr(self.env, "get_log_vars"):
            tmp_env_log_vars = {f"Interaction/{k}": (self.i_episode, v) for k, v in self.env.get_log_vars().items()}
            log_vars.update(tmp_env_log_vars)

    def _training_step(self, log_vars):
        if not self.i_episode % hparams['training_interval'] == 0:
            return
        for _ in range(hparams['training_times']):
            self.i_iter_critic += 1
            # batched_sample = self.replay_buffer.sample(hparams['batch_size'])
            batched_sample = self.trans_replay_buffer.sample(hparams['batch_size'])
            
            if batched_sample is None:
                # The replay buffer has not store enough sample.
                break
            losses = {}
            self.agent.cal_q_loss(batched_sample, losses, log_vars=log_vars, global_steps=self.i_iter_critic)
            total_loss = sum(losses.values())
            self.optimizer.zero_grad()
            total_loss.backward()
            for loss_name, loss in losses.items():
                log_vars[f'Training/{loss_name}'] = (self.i_iter_critic, loss.item())
            log_vars['Training/q_grad'] = (self.i_iter_critic, get_grad_norm(self.agent.learned_model, l=2))
            self.optimizer.step()

            if self.i_iter_critic % 5 == 0:
                self.agent.update_target()

    def _testing_step(self, log_vars):
        if not self.i_episode % hparams['testing_interval'] == 0:
            return
        episodic_reward_lst = []
        if hasattr(self.env, "get_log_vars"):
            episodic_env_log_vars = {}
        for _ in tqdm.tqdm(range(1, hparams['testing_episodes'] + 1), desc='Testing Episodes: '):
            # obs, adj = self.env.reset() 
            obs, adj, _ = self.env.reset()
            if hasattr(self.agent, 'reset_hidden_states'):
                self.agent.reset_hidden_states(obs.shape[0])
            tmp_reward_lst = []
            for t in range(hparams['episode_length']):
                epsilon = hparams['min_epsilon']
                action = self.agent.action(obs, adj, epsilon=epsilon, action_mode=hparams['testing_action_mode'])
                reward, next_obs, next_adj, done, _ = self.env.step(action)
                obs, adj = next_obs, next_adj
                tmp_reward_lst.append(sum(reward))
            episodic_reward_lst.append(sum(tmp_reward_lst))
            if hasattr(self.env, "get_log_vars"):
                tmp_env_log_vars = self.env.get_log_vars()
                for k, v in tmp_env_log_vars.items():
                    if k not in episodic_env_log_vars.keys():
                        episodic_env_log_vars[k] = []
                    episodic_env_log_vars[k].append(v)
        if hasattr(self.env, "get_log_vars"):
            episodic_env_log_vars = {f"Testing/{k}": (self.i_episode, np.mean(v)) for k, v in
                                     episodic_env_log_vars.items()}
            log_vars.update(episodic_env_log_vars)
        # Record the total reward obtain by all agents at each time step
        episodic_reward_mean = np.mean(episodic_reward_lst)
        episodic_reward_std = np.std(episodic_reward_lst)
        log_vars['Testing/mean_episodic_reward'] = (self.i_episode, episodic_reward_mean)
        log_vars['Testing/std_episodic_reward'] = (self.i_episode, episodic_reward_std)

        logging.info(
            f"Episode {self.i_episode} evaluation reward: mean {episodic_reward_mean},"
            f" std {episodic_reward_std}")
        # Save checkpoint when each testing phase is end.
        if episodic_reward_mean > self.best_eval_reward:
            self.save_best_ckpt = True
            logging.info(
                f"Best evaluation reward update: {self.best_eval_reward} ==> {episodic_reward_mean}")
            self.best_eval_reward = episodic_reward_mean
        else:
            self.save_best_ckpt = False
        self.save_checkpoint()

    def run_training_loop(self):
        start_episode = self.i_episode
        map_list = []
        for trans_type in hparams["data_trans_list"]:
            index_map_list = get_trans_list(trans_type)
            map_list.append(index_map_list)

        for _ in tqdm.tqdm(range(start_episode, hparams['num_episodes'] + 1), desc='Training Episode: '):
            
            log_vars = {}  # e.g. {'Training/q_loss':(16000, 0.999)}
            # Interaction Phase
            self._interaction_step(log_vars=log_vars, map_list=map_list)
            # Training Phase
            self._training_step(log_vars=log_vars)
            # Testing Phase
            self._testing_step(log_vars=log_vars)
            self.tb_logger.add_scalars(log_vars)

    def run_display_loop(self):
        while True:
            obs, adj = self.env.reset()
            if hasattr(self.agent, 'reset_hidden_states'):
                self.agent.reset_hidden_states(obs.shape[0])
            for t in range(hparams['episode_length']):
                self.env.render()
                epsilon = hparams['min_epsilon']
                action = self.agent.action(obs, adj, epsilon=epsilon, action_mode=hparams['testing_action_mode'])
                reward, next_obs, next_adj, done = self.env.step(action)
                obs, adj = next_obs, next_adj

    def run_eval_loop(self):
        rew_array = np.zeros(shape=[hparams['eval_episodes']])
        for i_episode in tqdm.tqdm(range(0, hparams['eval_episodes']), desc='Eval Episodes: '):
            tmp_reward_lst = []
            obs, adj = self.env.reset()
            if hasattr(self.agent, 'reset_hidden_states'):
                self.agent.reset_hidden_states(obs.shape[0])
            for t in range(hparams['episode_length']):
                epsilon = hparams['min_epsilon']
                action = self.agent.action(obs, adj, epsilon=epsilon, action_mode=hparams['testing_action_mode'])
                reward, next_obs, next_adj, done = self.env.step(action)
                obs, adj = next_obs, next_adj
                tmp_reward_lst.append(sum(reward))
            rew_array[i_episode] = sum(tmp_reward_lst) / hparams['episode_length']
        np.savetxt(os.path.join(self.work_dir, hparams['eval_result_name']), rew_array, delimiter=',')
        mean, std = rew_array.mean(), rew_array.std()
        logging.info(f"Evaluation complete, reward mean {mean}, std {std} .")
        logging.info(f"Evaluation result is saved at {os.path.join(self.work_dir, hparams['eval_result_name'])}.")

    def run(self):
        if hparams['display']:
            self.run_display_loop()
        elif hparams['evaluate']:
            self.run_eval_loop()
        else:
            self.run_training_loop()
