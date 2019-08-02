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

TRAIN_OBJ = "gap"  # "gap" or "dir"


class TrainParameter(object):
    def __init__(self):
        self.tools = TrainTools()
        data_buffer = Queue(1200)
        para_buffer = Queue(1)
        Process(target=self.tools.env_interaction, args=(data_buffer, para_buffer)).start()
        global tf
        import tensorflow as tf
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.config_gap = ConfigGap()
        self.config_dir = ConfigDirection()
        if TRAIN_OBJ == "gap":
            from agent.model_gap import TrainMethodTimeSeriesAttention
            self.train_network = TrainMethodTimeSeriesAttention(self.config_gap, self.sess)
        else:
            from agent.model_direction import TrainMethodTimeSeriesAttention
            self.train_network = TrainMethodTimeSeriesAttention(self.config_dir, self.sess)
        self.sess.run(tf.global_variables_initializer())
        # self.import_buffer = deque(maxlen=10000)
        self.replay_buffer = dict()
        if TRAIN_OBJ == "gap":
            self.replay_buffer["PH"] = deque(maxlen=3000)
            self.replay_buffer["NH"] = deque(maxlen=3000)
            self.replay_buffer["PM"] = deque(maxlen=3000)
            self.replay_buffer["NM"] = deque(maxlen=3000)
            self.replay_buffer["PF"] = deque(maxlen=3000)
            self.replay_buffer["NF"] = deque(maxlen=3000)
            self.replay_buffer["PR"] = deque(maxlen=3000)
            self.replay_buffer["NR"] = deque(maxlen=3000)
        else:
            self.replay_buffer["PH"] = deque(maxlen=3000)
            self.replay_buffer["NH"] = deque(maxlen=3000)
            self.replay_buffer["PL"] = deque(maxlen=3000)
            self.replay_buffer["NL"] = deque(maxlen=3000)
            self.replay_buffer["PR"] = deque(maxlen=3000)
            self.replay_buffer["NR"] = deque(maxlen=3000)
        self.epoch = 0
        self.statistic = dict()
        time.sleep(15)
        while True:
            self.single_training(data_buffer, para_buffer)

    def single_training(self, data_buffer, para_buffer):
        batch_size = self.config_gap.batch_size if TRAIN_OBJ == "gap" else self.config_dir.batch_size
        for _ in range(data_buffer.qsize()):
            data = data_buffer.get()
            if TRAIN_OBJ == "gap":
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
            else:
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
        print(" " * 90, "statistic_data:", self.statistic, sum(self.statistic.values()))
        if sum(self.statistic.values()) >= batch_size * 20 and min(self.statistic.values()) >= batch_size / 3:
            feed_dict = self.get_batch_data(self.replay_buffer)
            _, loss, test = self.train_network.actor_ppo_train(feed_dict)
            print(" " * 90, self.epoch, "ppo loss", loss, test[0][-1, 0, :], test[1][-1, 0, :])
            self.epoch += 1
        if self.epoch % 10000 == 1:
            net_parameters = self.get_parameters()
            para_buffer.put(net_parameters)

    def get_batch_data(self, data_buffer):
        batch_size = self.config_gap.batch_size if TRAIN_OBJ == "gap" else self.config_dir.batch_size
        feed_dict = dict()
        state, action, state_mask, action_mask, advantage_lat, advantage_lon = [], [], [], [], [], []
        for i in data_buffer.keys():
            random_seed = np.random.randint(0, len(data_buffer[i]), size=int(batch_size / len(data_buffer)))
            for j in random_seed:
                state.append(data_buffer[i][j]['st'])
                action.append(data_buffer[i][j]["at"][1:, :])
                action_mask.append(data_buffer[i][j]["am"][1:, :])
                advantage_lat.append(data_buffer[i][j]["ra"][1:])
                advantage_lon.append(data_buffer[i][j]["ro"][1:])
                if TRAIN_OBJ == "gap":
                    state_mask.append(data_buffer[i][j]["sm"])
        action = np.asarray(action).transpose((1, 0, 2))
        action_mask = np.asarray(action_mask).transpose((1, 0, 2))
        advantage_lat = np.asarray(advantage_lat).transpose((1, 0))
        advantage_lon = np.asarray(advantage_lon).transpose((1, 0))
        feed_dict[self.train_network.old_action] = action
        feed_dict[self.train_network.action_mask] = action_mask
        feed_dict[self.train_network.advantage_lat] = advantage_lat
        feed_dict[self.train_network.advantage_lon] = advantage_lon

        if TRAIN_OBJ == "gap":
            state = np.asarray(state).transpose((1, 2, 0, 3))
            state_mask = np.asarray(state_mask).transpose((1, 0, 2))
            for i in range(self.config_gap.time_steps):
                feed_dict[self.train_network.state[i]] = state[i]
                feed_dict[self.train_network.state_mask[i]] = state_mask[i]
        else:
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


class TrainTools(object):
    def __init__(self):
        self.config_dir = ConfigDirection()
        self.config_gap = ConfigGap()

    def env_interaction(self, buffer=Queue(1000), para_buffer=Queue(1)):
        self.agent = HRLNode()
        self.agent.get_init_in_HRL()
        single_loop = 0
        while True:
            print("-" * 60, single_loop, len(self.agent.single_buffer), buffer.qsize())
            if para_buffer.qsize() > 0:
                net_parameters = para_buffer.get()
                if TRAIN_OBJ == "gap":
                    self.agent.agent_gap.copy_net(net_parameters)
                    self.agent.agent_gap.net_save()
                else:
                    self.agent.agent_dir.copy_net(net_parameters)
                    self.agent.agent_dir.net_save()
            self.agent.update_env(self.agent.env)
            # self.agent.print_env(self.agent.env)
            if self.agent.get_result(self.agent.env, len(self.agent.single_buffer)):
                if len(self.agent.single_buffer) > 0:
                    positive_act, negative_act = list(), list()
                    self.result_process(self.agent.single_buffer, positive_act, negative_act)
                    self.reward_system(self.agent.single_buffer, positive_act, negative_act)
                    self.buffer_process(buffer, self.agent.single_buffer)
                    self.agent.single_buffer = list()
                self.agent.init_flag()
                self.agent.init_env()
                single_loop = 0
                continue
            single_loop += 1
            # self.agent.hierarchical_logic()
            env_data = self.agent.env.get_env()
            if self.agent.flag_dir:
                if self.agent.is_new[0]:
                    _ = self.agent.get_direction(env_data)
                    print("get the init hidden state of dir")
                    self.agent.is_new[0] = False
                    continue
                else:
                    self.agent.dir_command = self.agent.get_direction(env_data)
                    print("dir command", self.agent.dir_command)
                    if self.agent.dir_command[0] == 1:
                        self.agent.flag_dir = False
                        self.agent.flag_gap = True
            if self.agent.dir_command[0] == 1 and self.config_dir.tree_search and TRAIN_OBJ == "dir":
                self.tree_search_dir(env_data, buffer)
            self.agent.dir_to_gap(self.agent.env.veh_num, self.agent.dir_command, env_data)
            if self.agent.flag_gap:
                if self.agent.is_new[1]:
                    _ = self.agent.get_gap(env_data)
                    print("get the init hidden state of gap")
                    self.agent.is_new[1] = False
                    continue
                else:
                    self.agent.gap_command = self.agent.get_gap(env_data)
                    print("gap command", self.agent.gap_command)
                    if self.agent.gap_command[0] == 2:
                        self.agent.flag_gap = False
                self.agent.env.set_ego_command(self.agent.dir_command, self.agent.gap_command)
            if self.agent.gap_command[0] == 2 and self.agent.config_gap.tree_search and TRAIN_OBJ == "gap":
                self.tree_search_gap(env_data, buffer)

    def result_process(self, single_buffer, positive_act, negative_act, undetermined_act=list(), dir_no_suit_gap = False):
        lat_dim = self.config_gap.lat_dim if TRAIN_OBJ == "gap" else self.config_dir.lat_dim
        if self.agent.CRASH:
            negative_act.append(np.argmax(single_buffer[-1]['at'][lat_dim:]))
        if self.agent.SUCCESS:
            positive_act.append(np.argmax(single_buffer[-1]['at'][lat_dim:]))
        if dir_no_suit_gap:
            undetermined_act.append(np.argmax(single_buffer[-1]['at'][lat_dim:]))

    def reward_system(self, single_buffer, positive_act, negative_act, best_act=None, undetermined_act=[None]):
        lat_dim = self.config_gap.lat_dim if TRAIN_OBJ == "gap" else self.config_dir.lat_dim
        gamma_lon = 0.1
        crash_punish = -20 if TRAIN_OBJ == "gap" else -30
        success_encourage = 50
        hesitate_punish = -5
        best_encourage = 20 if TRAIN_OBJ == "gap" else 2.4
        base_punish = -0.2 if TRAIN_OBJ == "gap" else -0.5 # v3 
        lon_reward = 2.4 # if TRAIN_OBJ == "gap" else 3.6 # v4

        action_lon = np.argmax(single_buffer[-1]['am'][lat_dim:])
        cumulation_reward_lon = 0
        if action_lon in negative_act:
            cumulation_reward_lat = crash_punish
        elif action_lon in positive_act:
            cumulation_reward_lat = success_encourage
        elif action_lon in undetermined_act:
            cumulation_reward_lat = crash_punish
        else:
            cumulation_reward_lat = hesitate_punish
        # if self.CRASH:
        #     if np.argmax(data["at"][: lat_dim]) != 2 and np.argmax(data["at"][lat_dim:]) != crash_action:
        #         cumulation_reward_lat += 1.05
        for i, data in enumerate(reversed(single_buffer)):
            single_lon = np.argmax(data["am"][lat_dim:])
            if single_lon in negative_act:
                cumulation_reward_lon -= lon_reward
            elif single_lon in positive_act:
                cumulation_reward_lon += lon_reward
            elif single_lon in undetermined_act:
                cumulation_reward_lon += 0
            if single_lon == best_act:
                cumulation_reward_lon += best_encourage
            data["ra"] = cumulation_reward_lat
            data["ro"] = cumulation_reward_lon
            # print("final reward:", cumulation_reward_lat, cumulation_reward_lon)
            gamma_lat = 0.1 if i < 1 else 0.98 # v5 = 0.8
            cumulation_reward_lat = cumulation_reward_lat * gamma_lat + base_punish
            cumulation_reward_lon = cumulation_reward_lon * gamma_lon + 0

    def buffer_process(self, buffer, single_buffer):
        time_steps = self.config_gap.time_steps if TRAIN_OBJ == "gap" else self.config_dir.time_steps
        lat_dim = self.config_gap.lat_dim if TRAIN_OBJ == "gap" else self.config_dir.lat_dim
        output_dim = self.config_gap.output_dim if TRAIN_OBJ == "gap" else self.config_dir.output_dim
        reward_lat, reward_lon, state, action, state_mask, action_mask = [], [], [], [], [], []
        for data in single_buffer:
            state.append(data["st"])
            action.append(data["at"])
            action_mask.append(data["am"])
            reward_lat.append(data["ra"])
            reward_lon.append(data["ro"])
            if TRAIN_OBJ == "gap":
                state_mask.append(data["sm"])
        le = len(single_buffer)
        imp = True
        # if reward_lat[0] >= 0:
        #     imp = True if statistic_dict[action_lon + 1] <= 1.5 * np.min(list(statistic_dict.values())) else False
        # else:
        #     imp = True if statistic_dict[-action_lon - 1] < statistic_dict[action_lon + 1] else False
        action_lon = np.argmax(action[-1][lat_dim:])
        action_lon = action_lon + 1 if reward_lat[-1] > 0 else -action_lon - 1
        # statistic = [reward_lat[-1], action_lon] if np.argmax(action[-1][:lat_dim]) == 2 else [reward_lat[-1], 0]
        statistic = [reward_lat[-1], action_lon] if sum(reward_lon) != 0 else [reward_lat[-1], 0]
        action = np.concatenate((np.asarray(action), np.zeros(shape=[time_steps - le, output_dim])), axis=0)
        action_mask = np.concatenate((np.asarray(action_mask), np.zeros(shape=[time_steps - le, output_dim])), axis=0)
        reward_lat = np.concatenate((np.asarray(reward_lat), np.zeros(shape=[time_steps - le])))
        reward_lon = np.concatenate((np.asarray(reward_lon), np.zeros(shape=[time_steps - le])))
        if TRAIN_OBJ == "gap":
            state = np.concatenate((np.asarray(state), np.zeros(shape=[time_steps - le, self.config_gap.veh_num, self.config_gap.veh_dim])), axis=0)
            state_mask = np.concatenate((np.asarray(state_mask), np.zeros(shape=[time_steps - le, self.config_gap.gap_num])), axis=0)
            buffer.put({"st": state, "sm": state_mask, "at": action, "am": action_mask,
                        "ra": reward_lat, "ro": reward_lon, "imp": imp, "statis": statistic, "len":le})
        else:
            state = np.concatenate((np.asarray(state), np.zeros(shape=[time_steps - le, 3, self.config_dir.veh_num, self.config_dir.veh_dim])), axis=0)
            buffer.put({"st": state, "at": action, "am": action_mask,
                        "ra": reward_lat, "ro": reward_lon, "imp": imp, "statis": statistic, "len":le})

    def tree_search_dir(self, env_data, total_buffer):
        state, _ = self.agent.agent_dir.get_input(env_data)
        ts_buffer = list()
        compare = dict()
        print("\n", "start dir tree search!!!", "\n")
        positive_act, negative_act, undetermined_act = list(), list(), list()
        for i in range(2):
            ts_env_data = copy.deepcopy(env_data)
            ts_env = copy.deepcopy(self.agent.env)
            ts_buffer.append(copy.deepcopy(self.agent.single_buffer[:-1]))
            action = self.agent.agent_dir.tree_search_based_decision()
            command, mask = self.agent.agent_dir.get_output(action)
            print("show tree search", action, command)
            ts_buffer[-1].append({"st": state, "at": action, "am": mask})
            self.agent.dir_to_gap(self.agent.env.veh_num, command, ts_env_data)
            self.agent.flag_gap = True
            self.agent.is_new[1] = True
            process_time = 0
            total_gap = 0
            while True:
                if total_gap >= self.config_gap.time_steps:
                    self.result_process(ts_buffer[-1], positive_act, negative_act, undetermined_act, True)
                    del ts_env
                    print("UNDETERMINED")
                    break
                if self.agent.flag_gap:
                    if self.agent.is_new[1]:
                        _ = self.agent.get_gap(ts_env_data)
                        # print("get the init hidden state of gap")
                        self.agent.is_new[1] = False
                        continue
                    else:
                        self.agent.gap_command = self.agent.get_gap(ts_env_data)
                        # print("gap command", self.agent.gap_command)
                        if self.agent.gap_command[0] == 2:
                            self.agent.flag_gap = False
                    ts_env.set_ego_command(command, self.agent.gap_command)
                    total_gap += 1
                self.agent.update_env(ts_env)
                # self.agent.print_env(ts_env)
                process_time += 0.1
                if self.agent.get_result(ts_env, len(ts_buffer[-1])):
                    self.result_process(ts_buffer[-1], positive_act, negative_act)
                    if self.agent.SUCCESS:
                        compare[positive_act[-1]] = (ts_env.ego.x - self.agent.env.ego.x) / process_time
                    del ts_env
                    break
        if len(positive_act) >= 2:
            best_act = compare.keys()[np.argmax(compare.values())]
        elif len(positive_act) == 1:
            best_act = None
        else:
            best_act = None
        # v7
        # if len(undetermined_act) == 1 and len(positive_act) == 1:
        #     negative_act += undetermined_act
        #     # positive_act += undetermined_act #v6
        # elif len(undetermined_act) == 1 and len(negative_act) == 1:
        #     positive_act += undetermined_act
        # elif len(undetermined_act) == 2:
        #     negative_act += undetermined_act
        print("positive_result", positive_act, "best_result", best_act, "negative_result", negative_act, "undetermined", undetermined_act)
        for i in range(len(ts_buffer)):
            self.reward_system(ts_buffer[i], positive_act, negative_act, best_act, undetermined_act)
        # if len(undetermined_act) == 2:
        if len(positive_act) == 0: # v7
            for single_buffer in ts_buffer:
                for data in single_buffer:
                    data["ro"] = 0
            ts_buffer.append(copy.deepcopy(ts_buffer[0]))
            ts_buffer[-1][-1]["at"][1] = 0
            ts_buffer[-1][-1]["at"][0] = 1
            ts_buffer[-1][-1]["am"] = ts_buffer[-1][-1]["at"]
            for data in ts_buffer[-1]:
                data["ra"] = abs(data["ra"]) / 4
        for data in ts_buffer:
            # print(data[-1])
            # for i in data:
                # print(i["ra"], i["ro"])
            self.buffer_process(total_buffer, data)
        del ts_buffer
        self.agent.init_flag()
        self.agent.init_env()
        self.agent.single_buffer = list()

    def tree_search_gap(self, env_data, total_buffer):
        state, s_mask = self.agent.agent_gap.get_input(env_data)
        ts_buffer = list()
        compare = dict()
        print("\n", "start gap tree search!!!", "\n")
        positive_act, negative_act = list(), list()
        for i in range(int(sum(s_mask))):
            ts_env = copy.deepcopy(self.agent.env)
            ts_buffer.append(copy.deepcopy(self.agent.single_buffer[:-1]))
            action = self.agent.agent_gap.tree_search_based_decision(s_mask)
            command, a_mask = self.agent.agent_gap.get_output(action, s_mask)
            print("show tree search", action, command)
            ts_buffer[-1].append({"st": state, "sm": s_mask, "at": action, "am": a_mask})
            ts_env.set_ego_command(self.agent.dir_command, command)
            process_time = 0
            while True:
                self.agent.update_env(ts_env)
                # self.agent.print_env(ts_env)
                process_time += 0.1
                if self.agent.get_result(ts_env, len(ts_buffer[-1])):
                    self.result_process(ts_buffer[-1], positive_act, negative_act)
                    if self.agent.SUCCESS:
                        compare[positive_act[-1]] = (ts_env.ego.x - self.agent.env.ego.x) / process_time
                    del ts_env
                    break
        if len(positive_act) >= 2:
            best_act = compare.keys()[np.argmax(compare.values())]
        elif len(positive_act) == 1:
            best_act = None
        else:
            best_act = None
        print("positive_result", positive_act, "best_result", best_act)

        for i in range(len(ts_buffer)):
            self.reward_system(ts_buffer[i], positive_act, negative_act, best_act)
        if len(positive_act) == 0 and len(ts_buffer) != 0:
            for single_buffer in ts_buffer:
                for data in single_buffer:
                    data["ro"] = 0
            ts_buffer.append(copy.deepcopy(ts_buffer[0]))
            ts_buffer[-1][-1]["at"][2] = 0
            ts_buffer[-1][-1]["at"][np.random.choice([0, 1], 1, p=[0.2, 0.8])[0]] = 1
            ts_buffer[-1][-1]["am"] = ts_buffer[-1][-1]["at"]
            for data in ts_buffer[-1]:
                data["ra"] = abs(data["ra"]) / 4
        for data in ts_buffer:
            # print(data[-1])
            # for i in data:
            #     print(i["ra"])
            self.buffer_process(total_buffer, data)
        del ts_buffer
        self.agent.init_flag()
        self.agent.init_env()
        self.agent.single_buffer = list()

if __name__ == '__main__':
    a = TrainParameter()
