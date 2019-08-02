import numpy as np
from agent.model_gap import TestMethodTimeSeriesAttention
import tensorflow as tf
import os


class AgentGap(object):
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess
        self.network = TestMethodTimeSeriesAttention(self.config, self.sess)
        self.hidden = np.zeros([self.config.num_layers, 1, self.config.hidden_dim], dtype=float)
        self.cell = np.zeros([self.config.num_layers, 1, self.config.hidden_dim], dtype=float)
        self.noise_hold_num = -1
        self.search_num = 0
        self.saver = tf.train.Saver(self.network.net_vars)

    def get_input(self, data):
        # data is sorted, [veh_num, 4], dim-2 is: x, y, speed, str_lane
        # the data[0] is ego
        output = np.zeros(shape=[self.config.veh_num, self.config.veh_dim])
        mask = np.zeros(shape=[self.config.gap_num], dtype=float)
        output[0, 2] = data[0][2]
        pos, neg = 0, 0
        object_lane = []
        for i in data[1:]:
            if i[3] == data[0][3]:
                output[1, 0] = 1
                output[1, 1] = i[0]
                output[1, 2] = i[2] - data[0][2]
            else:
                object_lane.append(i)
                if i[0] >= 0:
                    pos += 1
                else:
                    neg += 1
        for i, veh in enumerate(object_lane):
            i += 2 - pos
            mask[i] = 1
            output[2 + i, 0] = 1
            output[2 + i, 1] = veh[0]
            output[2 + i, 2] = veh[2] - data[0][2]
            mask[i+1] = 1 if i == len(object_lane) + 1 - pos else 0
        return output, mask

    def get_output(self, action, state_mask):
        # action is [output_dim]
        command_lat = np.argmax(action[: self.config.lat_dim])
        command_lon = np.argmax(action[self.config.lat_dim:])
        mask = np.zeros(shape=np.shape(action), dtype=float)
        mask[command_lat] = 1
        mask[self.config.lat_dim + command_lon] = 1
        command_lon = np.argmax(action[3 + np.where(state_mask[:] != 0)[0]])
        return [command_lat, command_lon], mask

    def model_based_decision(self, state, mask, new_traj):
        if new_traj:
            self.hidden = np.zeros([self.config.num_layers, 1, self.config.hidden_dim], dtype=float)
            self.cell = np.zeros([self.config.num_layers, 1, self.config.hidden_dim], dtype=float)
        feed_dict = dict()
        feed_dict[self.network.state[0]] = state[:, np.newaxis, :]
        feed_dict[self.network.mask[0]] = mask[np.newaxis, :]
        feed_dict[self.network.init_hidden] = self.hidden
        feed_dict[self.network.init_cell] = self.cell
        feed_dict[self.network.new_traj] = new_traj
        self.hidden, self.cell, net_output = self.network.actor_run(feed_dict)
        # para = self.sess.run(self.network.net_vars)
        # for n, var in zip(self.network.net_vars, para):
        #     print(n.name, np.shape(var))
        return net_output

    def noise_based_decision(self, state_mask):
        output = np.zeros(shape=[self.config.output_dim], dtype=float)
        if self.noise_hold_num == -1:
            self.noise_hold_num = np.random.choice(range(1, self.config.time_steps - 1), 1)[0]
        if self.noise_hold_num > 0:
            action_lat = np.random.choice([0, 1], 1, p=[0.2, 0.8])[0]
            action_lon = np.random.choice(range(self.config.lon_dim), 1, p=state_mask / np.sum(state_mask))[0]
        else:
            action_lat = 2
            action_lon = np.random.choice(range(self.config.lon_dim), 1, p=state_mask / np.sum(state_mask))[0]
        output[action_lat] = 1
        output[action_lon + self.config.lat_dim] = 1
        self.noise_hold_num -= 1
        return output

    def tree_search_based_decision(self, state_mask):
        output = np.zeros(shape=[self.config.output_dim], dtype=float)
        if self.search_num == 0:
            self.search_num = int(sum(state_mask))
        output[self.config.lat_dim - 1] = 1
        j = 0
        for k, i in enumerate(state_mask):
            j += i
            if j == self.search_num:
                output[self.config.lat_dim + k] = 1
                break
        self.search_num -= 1
        return output

    def copy_net(self, trained_parameters):
        self.sess.run([tf.assign(var_env, var) for var_env, var in zip(self.network.net_vars, trained_parameters)])

    def net_restore(self):
        self.saver.restore(self.sess, 
            os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/model/0709/current_actor_gap.model")
        print " " * 140, "gap model have been restored"

    def net_save(self):
        self.saver.save(self.sess, 
            os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/model/current_actor_gap.model")
        print " " * 140, "gap model have been saved"
