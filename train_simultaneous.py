from __future__ import print_function
from agent.config import ConfigDirection, ConfigGap
from agent.agent_direction import AgentDirection
from agent.agent_gap import AgentGap
from hrl_agent import HRLNode
import numpy as np
import copy
from environment import Environment
from multiprocessing import Queue, Process
from collections import deque
import time
from agent.model_gap import TrainMethodTimeSeriesAttention as TrainMethodGap
from agent.model_direction import TrainMethodTimeSeriesAttention as TrainMethodDir
import tensorflow as tf


class EnvInteraction(object):
    def __init__(self):
        self.config_dir = ConfigDirection()
        self.config_gap = ConfigGap()
        self.train_gap = TrainParameterGap()
        self.train_dir = TrainParameterDir()
        communication_buffer_gap = Queue(1200)
        communication_buffer_dir = Queue(1200)
        para_buffer_gap = Queue(1)
        para_buffer_dir = Queue(1)
        Process(target=self.train_dir.train_process, args=(communication_buffer_dir, para_buffer_dir)).start()
        Process(target=self.train_gap.train_process, args=(communication_buffer_gap, para_buffer_gap)).start()

        self.agent = HRLNode()
        self.agent.get_init_in_HRL()      
        single_buffer_dir, single_buffer_gap, ts_command_dir = list(), list(), list()
        ts_buffer_gap, ts_buffer_dir = dict(), dict()
        compare = dict()
        positive_act, negative_act = {"left":list(), "right":list()}, {"left":list(), "right":list()}
        ts_env = {"init":copy.deepcopy(self.agent.env)}
        env_indicator = {"init": True}
        while True:
            print("-" * 60, communication_buffer_gap.qsize(), communication_buffer_dir.qsize())
            if para_buffer_gap.qsize() > 0:
                net_parameters = para_buffer_gap.get()
                self.agent.agent_gap.copy_net(net_parameters)
                self.agent.agent_gap.net_save()
            if para_buffer_dir.qsize() > 0:
                net_parameters = para_buffer_dir.get()
                self.agent.agent_dir.copy_net(net_parameters)
                self.agent.agent_dir.net_save()

            for key in ts_env:
                if env_indicator[key]:
                    self.agent.update_env(ts_env[key])
                    if self.agent.get_result(ts_env[key], 0):
                        print("result", key)
                        self.result_process(ts_buffer_gap, key, self.config_gap.lat_dim, positive_act, negative_act)
                        print()
                        env_indicator[key] = False
            if sum(env_indicator.values()) == 0:
                print(positive_act, negative_act)
                for key in ts_buffer_gap:
                    self.reward_system_gap(ts_buffer_gap[key], key, self.config_gap.lat_dim, positive_act, negative_act)
                for key in ts_buffer_dir:
                    self.reward_system_dir(ts_buffer_dir[key], key, self.config_dir.lat_dim, positive_act, negative_act)
                self.negative_supplement(ts_buffer_gap, "left", positive_act, negative_act)
                self.negative_supplement(ts_buffer_gap, "right", positive_act, negative_act)
                self.negative_supplement(ts_buffer_dir, None, positive_act, negative_act)

                # for key in ts_buffer_gap:
                #     print(key)
                #     for i in ts_buffer_gap[key]:
                #         print(i["ra"], i["ro"], i["am"])
                # for key in ts_buffer_dir:
                #     print(key)
                #     for i in ts_buffer_dir[key]:
                #         print(i["ra"], i["ro"], i["am"])

                for key in ts_buffer_gap:
                    self.buffer_process(ts_buffer_gap[key], communication_buffer_gap, self.config_gap.lat_dim, self.config_gap.time_steps)
                for key in ts_buffer_dir:
                    self.buffer_process(ts_buffer_dir[key], communication_buffer_dir, self.config_dir.lat_dim, self.config_dir.time_steps)
                self.agent.init_env()
                self.agent.init_flag()
                ts_buffer_gap, ts_buffer_dir = dict(), dict()
                positive_act, negative_act = {"left":list(), "right":list()}, {"left":list(), "right":list()}
                single_loop = 0
                env_indicator = {"init": True}
                ts_env = {"init":copy.deepcopy(self.agent.env)}

            if self.agent.flag_dir:
                env_data = ts_env["init"].get_env()
                if self.agent.is_new[0]:
                    _ = self.agent.get_direction(env_data, single_buffer_dir)
                    print("get the init hidden state of dir")
                    self.agent.is_new[0] = False
                    continue
                else:
                    dir_command = self.agent.get_direction(env_data, single_buffer_dir)
                    print("dir command", dir_command)
                if dir_command[0] == 1:
                    self.agent.flag_dir = False
                    self.agent.flag_gap = True
                    ts_command_dir, dir_data = self.tree_search(self.agent.agent_dir, env_data, single_buffer_dir)
                    ts_env["left"] = copy.deepcopy(ts_env["init"])
                    ts_env["right"] = copy.deepcopy(ts_env["init"])
                    ts_buffer_dir["left"] = dir_data[1]
                    ts_buffer_dir["right"] = dir_data[0]
                    single_buffer_dir = list()
                    env_indicator["left"] = True
                    env_indicator["right"] = False
                    env_indicator["init"] = False
                    del ts_env["init"]
                    del dir_data
                    print(ts_buffer_dir.keys(), ts_command_dir)
                    cur_dir = "left"
                    continue
            index = {"left": 1, "right": 0}
            if self.agent.flag_gap:
                env_data = ts_env[cur_dir].get_env()
                self.agent.dir_to_gap(ts_env[cur_dir].veh_num, ts_command_dir[index[cur_dir]], env_data)
                # print(cur_dir, env_data, ts_env[cur_dir].veh_num)
                if self.agent.is_new[1]:
                    _ = self.agent.get_gap(env_data, single_buffer_gap)
                    print("get the init hidden state of gap")
                    self.agent.is_new[1] = False
                    continue
                else:
                    gap_command = self.agent.get_gap(env_data, single_buffer_gap)
                    print("gap command", gap_command)
                ts_env[cur_dir].set_ego_command(ts_command_dir[index[cur_dir]], gap_command)
                if gap_command[0] == 2:
                    ts_command_gap, gap_data = self.tree_search(self.agent.agent_gap, env_data, single_buffer_gap)
                    for i, j in zip(ts_command_gap, gap_data):
                        ts_env[cur_dir+str(i[1])] = copy.deepcopy(ts_env[cur_dir])
                        ts_env[cur_dir+str(i[1])].set_ego_command(ts_command_dir[index[cur_dir]], i)
                        ts_buffer_gap[cur_dir+str(i[1])] = copy.deepcopy(j)
                        env_indicator[cur_dir+str(i[1])] = True
                    single_buffer_gap = list()
                    env_indicator[cur_dir] = False
                    del ts_env[cur_dir]
                    del gap_data
                    if cur_dir == "left":
                        self.agent.is_new[1] = True
                        cur_dir = "right"
                        env_indicator[cur_dir] = True
                    else:
                        self.agent.flag_gap = False
                    print(ts_buffer_gap.keys(), ts_command_gap)
                    print()

    def negative_supplement(self, data_buffer, key, positive_act, negative_act):
        if key == None:
            if len(positive_act["left"]) == 0 and len(positive_act["right"]) == 0 and len(negative_act["left"]) != 0 and len(negative_act["right"]) != 0:
                for i in data_buffer:
                    for data in data_buffer[i]:
                        data["ro"] = 0
                data_buffer["none"] = copy.deepcopy(data_buffer["left"])
                data_buffer["none"][-1]["at"][1] = 0
                data_buffer["none"][-1]["at"][0] = 1
                data_buffer["none"][-1]["am"] = data_buffer["none"][-1]["at"]
                for data in data_buffer["none"]:
                    data["ra"] = abs(data["ra"]) / 4
        else:
            if len(positive_act[key]) == 0 and len(negative_act[key]) != 0:
                for i in data_buffer:
                    if i[:-1] == key:
                        for data in data_buffer[i]:
                            data["ro"] = 0
                data_buffer[key+"&"] = copy.deepcopy(data_buffer[key+"0"])
                data_buffer[key+"&"][-1]["at"][2] = 0
                data_buffer[key+"&"][-1]["at"][np.random.choice([0, 1], 1, p=[0.2, 0.8])[0]] = 1
                data_buffer[key+"&"][-1]["am"] = data_buffer[key+"&"][-1]["at"]
                for data in data_buffer[key+"&"]:
                    data["ra"] = abs(data["ra"]) / 4

    def tree_search(self, agent, env_data, single_buffer):
        state, s_mask = agent.get_input(env_data)
        ts_buffer = list()
        ts_command = list()
        print("\n", "start tree search!!!")
        for i in range(int(sum(s_mask))):
            # ts_buffer.append(copy.deepcopy(single_buffer[:-1]))
            ts_buffer.append(single_buffer[:-1])
            action = agent.tree_search_based_decision(s_mask)
            command, mask = agent.get_output(action, s_mask)
            print("show tree search", action, command)
            ts_command.append(command)
            # ts_buffer[key+str(command[1])] = copy.deepcopy(single_buffer[:-1])
            ts_buffer[-1].append({"st": state, "at": action, "am": mask, "sm": s_mask})
        return ts_command, ts_buffer

    def result_process(self, data_buffer, key, lat_dim, positive_act, negative_act):
        if self.agent.CRASH:
            negative_act[key[:-1]].append(np.argmax(data_buffer[key][-1]["am"][lat_dim:]))
        if self.agent.SUCCESS:
            positive_act[key[:-1]].append(np.argmax(data_buffer[key][-1]["am"][lat_dim:]))

    def reward_system_gap(self, data_buffer, key, lat_dim, positive_act, negative_act, best_act=None):
        gamma_lon = 0.1
        crash_punish = -20
        success_encourage = 50
        hesitate_punish = -5
        lon_reward = 2.4
        base_punish = -0.2
        best_encourage = 20

        action_lon = np.argmax(data_buffer[-1]["am"][lat_dim:])
        if action_lon in positive_act[key[:-1]]:
            cumulation_reward_lat = success_encourage
        elif action_lon in negative_act[key[:-1]]:
            cumulation_reward_lat = crash_punish
        else:
            cumulation_reward_lat = hesitate_punish
        cumulation_reward_lon = 0
        for i, data in enumerate(reversed(data_buffer)):
            single_lon = np.argmax(data["am"][lat_dim:])
            if single_lon in positive_act[key[:-1]]:
                cumulation_reward_lon += lon_reward
            elif single_lon in negative_act[key[:-1]]:
                cumulation_reward_lon -= lon_reward
            data["ra"] = cumulation_reward_lat
            data["ro"] = cumulation_reward_lon
            gamma_lat = 0.1 if i < 1 else 0.98
            cumulation_reward_lat = cumulation_reward_lat * gamma_lat + base_punish
            cumulation_reward_lon = cumulation_reward_lon * gamma_lon + 0

    def reward_system_dir(self, data_buffer, key, lat_dim, positive_act, negative_act, best_act=None):
        gamma_lon = 0.1
        crash_punish = -20
        success_encourage = 50
        hesitate_punish = -5
        lon_reward = 2.4
        base_punish = -0.2
        best_encourage = 20
        action_lon = np.argmax(data_buffer[-1]["am"][lat_dim:])
        if len(positive_act[key]) > 0:
            cumulation_reward_lat = success_encourage
        elif len(negative_act[key]) > 0:
            cumulation_reward_lat = crash_punish
        else:
            cumulation_reward_lat = success_encourage
        cumulation_reward_lon = 0
        index = {1:"right", 0:"left"}
        for i, data in enumerate(reversed(data_buffer)):
            single_lon = np.argmax(data["am"][lat_dim:])
            if len(positive_act[index[single_lon]]) > 0:
                cumulation_reward_lon += lon_reward
            elif len(negative_act[index[single_lon]]) == 0:
                cumulation_reward_lon += lon_reward
            else:
                cumulation_reward_lon -= lon_reward
            data["ra"] = cumulation_reward_lat
            data["ro"] = cumulation_reward_lon
            gamma_lat = 0.1 if i < 1 else 0.98
            cumulation_reward_lat = cumulation_reward_lat * gamma_lat + base_punish
            cumulation_reward_lon = cumulation_reward_lon * gamma_lon + 0

    def buffer_process(self, data_buffer, buffer, lat_dim, time_steps):
        reward_lat, reward_lon, state, action, state_mask, action_mask = [], [], [], [], [], []
        for data in data_buffer:
            state.append(data["st"])
            action.append(data["at"])
            action_mask.append(data["am"])
            reward_lat.append(data["ra"])
            reward_lon.append(data["ro"])
            state_mask.append(data["sm"])
        le = len(data_buffer)
        imp = True
        action_lon = np.argmax(action[-1][lat_dim:])
        output_dim = len(action[-1])
        action_lon = action_lon + 1 if reward_lat[-1] > 0 else -action_lon - 1
        # statistic = [reward_lat[-1], action_lon] if np.argmax(action[-1][:lat_dim]) == 2 else [reward_lat[-1], 0]
        statistic = [reward_lat[-1], action_lon] if sum(reward_lon) != 0 else [reward_lat[-1], 0]
        action = np.concatenate((np.asarray(action), np.zeros(shape=[time_steps - le, output_dim])), axis=0)
        action_mask = np.concatenate((np.asarray(action_mask), np.zeros(shape=[time_steps - le, output_dim])), axis=0)
        reward_lat = np.concatenate((np.asarray(reward_lat), np.zeros(shape=[time_steps - le])))
        reward_lon = np.concatenate((np.asarray(reward_lon), np.zeros(shape=[time_steps - le])))
        state_mask = np.concatenate((np.asarray(state_mask), np.zeros(shape=[time_steps - le, output_dim-lat_dim])), axis=0)
        state = np.concatenate((np.asarray(state), np.zeros(shape=np.concatenate((np.asarray([time_steps- le]), np.shape(state)[1:])))), axis=0)
        # state = np.concatenate((np.asarray(state), np.zeros(shape=[time_steps - le, 3, self.config_dir.veh_num, self.config_dir.veh_dim])), axis=0)
        buffer.put({"st": state, "sm": state_mask, "at": action, "am": action_mask,
                    "ra": reward_lat, "ro": reward_lon, "imp": imp, "statis": statistic, "len":le})

        # print(reward_lat, reward_lon, le, statistic, np.shape(np.asarray(state)))


class TrainParameterGap(object):
    def __init__(self):
        self.replay_buffer = dict()
        self.replay_buffer["PH"] = deque(maxlen=3000)
        self.replay_buffer["NH"] = deque(maxlen=3000)
        self.replay_buffer["PM"] = deque(maxlen=3000)
        self.replay_buffer["NM"] = deque(maxlen=3000)
        self.replay_buffer["PF"] = deque(maxlen=3000)
        self.replay_buffer["NF"] = deque(maxlen=3000)
        self.replay_buffer["PR"] = deque(maxlen=3000)
        self.replay_buffer["NR"] = deque(maxlen=3000)
        self.epoch = 0
        self.statistic = dict()
        self.begin_train = False

    def train_process(self, data_buffer, para_buffer):
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.config_gap = ConfigGap()
        self.train_network = TrainMethodGap(self.config_gap, self.sess)
        self.sess.run(tf.global_variables_initializer())
        while True:
            self.single_training(data_buffer, para_buffer)

    def single_training(self, data_buffer, para_buffer):
        for _ in range(data_buffer.qsize()):
            data = data_buffer.get()
            if data["statis"][1] == 3:
                self.replay_buffer["PM"].append(data)
            elif data["statis"][1] > 3:
                self.replay_buffer["PR"].append(data)
            elif data["statis"][1] > 0:
                self.replay_buffer["PF"].append(data)
            elif data["statis"][1] == -3:
                self.replay_buffer["NM"].append(data)
            elif data["statis"][1] < -3:
                self.replay_buffer["NR"].append(data)
            elif data["statis"][1] < 0:
                self.replay_buffer["NF"].append(data)
            elif data["statis"][1] == 0 and data["statis"][0] >= 0:
                self.replay_buffer["PH"].append(data)
            elif data["statis"][1] == 0 and data["statis"][0] < 0:
                self.replay_buffer["NH"].append(data)
        for i in self.replay_buffer.keys():
            self.statistic[i] = len(self.replay_buffer[i])
        if self.begin_train or self.epoch % 10000 == 1:    
            print(" " * 90, "gap statistic:", self.statistic, sum(self.statistic.values()))
        if sum(self.statistic.values()) >= self.config_gap.batch_size * 20 and min(self.statistic.values()) >= self.config_gap.batch_size / 3:
            feed_dict = self.get_batch_data(self.replay_buffer)
            _, loss, test = self.train_network.actor_ppo_train(feed_dict)
            print(" " * 90, self.epoch, "gap loss", loss, test[0][1, 0, :], test[1][1, 0, :])
            self.begin_train = True
        self.epoch += 1
        if self.epoch % 10000 == 1 and self.begin_train:
            net_parameters = self.get_parameters()
            para_buffer.put(net_parameters)

    def get_batch_data(self, data_buffer):
        feed_dict = dict()
        state, action, state_mask, action_mask, advantage_lat, advantage_lon = [], [], [], [], [], []
        for i in data_buffer.keys():
            random_seed = np.random.randint(0, len(data_buffer[i]), size=int(self.config_gap.batch_size / len(data_buffer)))
            for j in random_seed:
                state.append(data_buffer[i][j]['st'])
                action.append(data_buffer[i][j]["at"][1:, :])
                action_mask.append(data_buffer[i][j]["am"][1:, :])
                advantage_lat.append(data_buffer[i][j]["ra"][1:])
                advantage_lon.append(data_buffer[i][j]["ro"][1:])
                state_mask.append(data_buffer[i][j]["sm"])
        action = np.asarray(action).transpose((1, 0, 2))
        action_mask = np.asarray(action_mask).transpose((1, 0, 2))
        advantage_lat = np.asarray(advantage_lat).transpose((1, 0))
        advantage_lon = np.asarray(advantage_lon).transpose((1, 0))
        feed_dict[self.train_network.old_action] = action
        feed_dict[self.train_network.action_mask] = action_mask
        feed_dict[self.train_network.advantage_lat] = advantage_lat
        feed_dict[self.train_network.advantage_lon] = advantage_lon
        state = np.asarray(state).transpose((1, 2, 0, 3))
        state_mask = np.asarray(state_mask).transpose((1, 0, 2))
        for i in range(self.config_gap.time_steps):
            feed_dict[self.train_network.state[i]] = state[i]
            feed_dict[self.train_network.state_mask[i]] = state_mask[i]
        return feed_dict

    def get_parameters(self):
        actor = self.sess.run(self.train_network.net_vars)
        # print "train net"
        # for n, var in zip(self.train_network.a_vars, actor):
            # print n.name, np.shape(var)
        return actor


class TrainParameterDir(object):
    def __init__(self):
        self.replay_buffer = dict()
        self.replay_buffer["PH"] = deque(maxlen=3000)
        self.replay_buffer["NH"] = deque(maxlen=3000)
        self.replay_buffer["PL"] = deque(maxlen=3000)
        self.replay_buffer["NL"] = deque(maxlen=3000)
        self.replay_buffer["PR"] = deque(maxlen=3000)
        self.replay_buffer["NR"] = deque(maxlen=3000)
        self.epoch = 0
        self.statistic = dict()
        self.begin_train = False

    def train_process(self, data_buffer, para_buffer):
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.config_dir = ConfigDirection()
        self.train_network = TrainMethodDir(self.config_dir, self.sess)
        self.sess.run(tf.global_variables_initializer())
        while True:
            self.single_training(data_buffer, para_buffer)

    def single_training(self, data_buffer, para_buffer):
        for _ in range(data_buffer.qsize()):
            data = data_buffer.get()
            if data["statis"][1] == 1:
                self.replay_buffer["PL"].append(data)
            elif data["statis"][1] == 2:
                self.replay_buffer["PR"].append(data)
            elif data["statis"][1] == -1:
                self.replay_buffer["NL"].append(data)
            elif data["statis"][1] == -2:
                self.replay_buffer["NR"].append(data)
            elif data["statis"][1] == 0 and data["statis"][0] >= 0:
                self.replay_buffer["PH"].append(data)
            elif data["statis"][1] == 0 and data["statis"][0] < 0:
                self.replay_buffer["NH"].append(data)
        for i in self.replay_buffer.keys():
            self.statistic[i] = len(self.replay_buffer[i])
        if self.begin_train or self.epoch % 10000 == 1:
            print(" " * 90, "dir statistic:", self.statistic, sum(self.statistic.values()))
        if sum(self.statistic.values()) >= self.config_dir.batch_size * 20 and min(self.statistic.values()) >= self.config_dir.batch_size / 3:
            feed_dict = self.get_batch_data(self.replay_buffer)
            _, loss, test = self.train_network.actor_ppo_train(feed_dict)
            print(" " * 90, self.epoch, "dir loss", loss, test[0][1, 0, :], test[1][1, 0, :])
            self.begin_train = True
        self.epoch += 1
        if self.epoch % 10000 == 1 and self.begin_train:
            net_parameters = self.get_parameters()
            para_buffer.put(net_parameters)

    def get_batch_data(self, data_buffer):
        feed_dict = dict()
        state, action, action_mask, advantage_lat, advantage_lon = [], [], [], [], []
        for i in data_buffer.keys():
            random_seed = np.random.randint(0, len(data_buffer[i]), size=int(self.config_dir.batch_size / len(data_buffer)))
            for j in random_seed:
                state.append(data_buffer[i][j]['st'])
                action.append(data_buffer[i][j]["at"][1:, :])
                action_mask.append(data_buffer[i][j]["am"][1:, :])
                advantage_lat.append(data_buffer[i][j]["ra"][1:])
                advantage_lon.append(data_buffer[i][j]["ro"][1:])
        action = np.asarray(action).transpose((1, 0, 2))
        action_mask = np.asarray(action_mask).transpose((1, 0, 2))
        advantage_lat = np.asarray(advantage_lat).transpose((1, 0))
        advantage_lon = np.asarray(advantage_lon).transpose((1, 0))
        feed_dict[self.train_network.old_action] = action
        feed_dict[self.train_network.action_mask] = action_mask
        feed_dict[self.train_network.advantage_lat] = advantage_lat
        feed_dict[self.train_network.advantage_lon] = advantage_lon
        state = np.asarray(state).transpose((1, 0, 2, 3, 4))
        for i in range(self.config_dir.time_steps):
            feed_dict[self.train_network.state[i]] = state[i]
        return feed_dict

    def get_parameters(self):
        actor = self.sess.run(self.train_network.net_vars)
        # print "train net"
        # for n, var in zip(self.train_network.a_vars, actor):
            # print n.name, np.shape(var)
        return actor


if __name__ == '__main__':
    a = EnvInteraction()


