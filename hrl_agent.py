from __future__ import print_function
import tensorflow as tf
import numpy as np
from agent.agent_direction import AgentDirection
from agent.agent_gap import AgentGap
from collections import deque
from agent.config import ConfigDirection, ConfigGap
from environment import Environment

TRAIN_OBJ = ["dir", "gap"]  # "gap" or "dir" or None


class HRLNode(object):
    def __init__(self):
        self.flag_dir = True
        self.flag_gap = False
        self.is_new = [True, True]
        self.SUCCESS = False
        self.CRASH = False
        self.HESITATE = False
        self.dir_command = [0, 0]
        self.gap_command = [0, 0]

        self.config_dir = ConfigDirection()
        self.config_gap = ConfigGap()
        self.success_rate = [0, 0]
        self.eva_sys = 1

    def get_init_in_HWP(self):
        sess = tf.Session()
        self.agent_dir = AgentDirection(self.config_dir, sess)
        self.agent_gap = AgentGap(self.config_gap, sess)
        sess.run(tf.global_variables_initializer())
        self.agent_gap.net_restore()
        self.history_state = deque(maxlen=2)
        self.history_vehnum = deque(maxlen=2)
        self.single_loop = 0
        print("python node have been init")

    def get_init_in_HRL(self):
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)
        self.env = Environment()
        self.single_buffer = list()
        self.agent_dir = AgentDirection(self.config_dir, sess)
        self.agent_gap = AgentGap(self.config_gap, sess)
        sess.run(tf.global_variables_initializer())
        self.agent_gap.net_restore()
        self.agent_dir.net_restore()
        self.init_env()

    def get_input(self, data):
        output = list()
        output.append([0, 0, data[0], "99_1_-2"])
        self.history_vehnum.append([0, 0, 0])
        if data[1][0] != -1:
            output.append([data[1][0], data[1][1], data[1][2], "99_1_-2"])
            self.history_vehnum[-1][1] += 1
        for i in data[2:6]:
            if i[0] != -1:
                output.append([i[0], i[1], i[2], "99_1_-1"])
                self.history_vehnum[-1][0] += 1
        for i in data[6:]:
            if i[0] != -1:
                output.append([i[0], i[1], i[2], "99_1_-3"])
                self.history_vehnum[-1][2] += 1
        self.history_state.append(output)
        print("python node input", output, self.history_vehnum)

    def get_output(self, direction):
        if len(self.history_state) == 0:
            print("warning: NOTHING have been input!")
            return [0, 0]
        if self.is_new[1] and len(self.history_state) == 2:
            if direction == 1:
                s0 = 1 + self.history_vehnum[0][1] + self.history_vehnum[0][0]
                s1 = 1 + self.history_vehnum[1][1] + self.history_vehnum[1][0]
                state_0 = self.history_state[0][:s0]
                state_1 = self.history_state[1][:s1]
            elif direction == 2:
                s0 = 1 + self.history_vehnum[0][1]
                t0 = 1 + self.history_vehnum[0][1] + self.history_vehnum[0][0]
                s1 = 1 + self.history_vehnum[1][1]
                t1 = 1 + self.history_vehnum[1][1] + self.history_vehnum[1][0]
                state_0 = self.history_state[0][:s0] + self.history_state[0][t0:]
                state_1 = self.history_state[1][:s1] + self.history_state[1][t1:]
            else:
                print("wrong input direction:", direction)
            _ = self.get_gap(state_0)
            print("get the init hidden state of gap")
            self.is_new[1] = False
            self.gap_command = self.get_gap(state_1)
        elif self.is_new[1]:
            if direction == 1:
                s0 = 1 + self.history_vehnum[0][1] + self.history_vehnum[0][0]
                state_0 = self.history_state[0][:s0]
            elif direction == 2:
                s0 = 1 + self.history_vehnum[0][1]
                t0 = 1 + self.history_vehnum[0][1] + self.history_vehnum[0][0]
                state_0 = self.history_state[0][:s0] + self.history_state[0][t0:]
            else:
                print("wrong input direction:", direction)
            _ = self.get_gap(state_0)
            print("get the init hidden state of gap")
            self.is_new[1] = False
            self.gap_command = [0, 0]
        else:
            if direction == 1:
                s1 = 1 + self.history_vehnum[1][1] + self.history_vehnum[1][0]
                state_1 = self.history_state[1][:s1]
            elif direction == 2:
                s1 = 1 + self.history_vehnum[1][1]
                t1 = 1 + self.history_vehnum[1][1] + self.history_vehnum[1][0]
                state_1 = self.history_state[1][:s1] + self.history_state[1][t1:]
            else:
                print("wrong input direction:", direction)
            self.gap_command = self.get_gap(state_1)
        print("gap command", self.gap_command)
        if self.gap_command[0] == 2 or self.single_loop >= self.config_gap.time_steps:
            self.is_new = [True, True]
            self.single_loop = 0
        self.single_loop += 1
        return self.gap_command

    def test_model(self):
        total_loop = 0
        if "gap" in TRAIN_OBJ:
            self.agent_gap.net_save()
        if "dir" in TRAIN_OBJ:
            self.agent_dir.net_save()
        total_gap = 0
        while True:
            print("-" * 60, total_loop, 100.0 * self.success_rate[0] / (self.success_rate[1] + 1e-6))
            self.update_env(self.env)
            # self.print_env(self.env)
            if self.get_result(self.env, max(len(self.single_buffer), total_gap)):
                total_gap = 0
                self.init_flag()
                self.init_env()
                self.success_rate[1] += 1
                if self.SUCCESS or self.HESITATE:
                    self.success_rate[0] += 1
                self.single_buffer = list()
                total_loop += 1
                continue
            env_data = self.env.get_env()
            if self.flag_dir:
                if self.is_new[0]:
                    _ = self.get_direction(env_data)
                    print("get the init hidden state of dir")
                    self.is_new[0] = False
                    continue
                else:
                    self.dir_command = self.get_direction(env_data)
                    print("dir command", self.dir_command)
                    if self.dir_command[0] == 1:
                        self.flag_dir = False
                        self.flag_gap = True
                        self.single_buffer = list()
            self.dir_to_gap(self.env.veh_num, self.dir_command, env_data)
            if self.flag_gap:
                if self.is_new[1]:
                    _ = self.get_gap(env_data)
                    print("get the init hidden state of gap")
                    self.is_new[1] = False
                    continue
                else:
                    self.gap_command = self.get_gap(env_data)
                    print("gap command", self.gap_command)
                    if self.gap_command[0] == 2:
                        self.flag_gap = False
                self.env.set_ego_command(self.dir_command, self.gap_command)
                total_gap += 1

    def get_result(self, env, len_loop):
        self.SUCCESS = False
        self.CRASH = False
        self.HESITATE = False
        if env.ego.t != "99_1_-2":
            num = 0
            for i in env.env_veh:
                if i.t == env.ego.t and i.x >= env.ego.x:
                    num += 1
            if num == env.gap[1]:
                print("SUCCESS")
                self.SUCCESS = True
            else:
                print("CRASH by wrong gap", num, env.gap[1])
                self.CRASH = True
        if (len_loop >= self.config_gap.time_steps and "gap" in TRAIN_OBJ) or \
                (len_loop >= self.config_dir.time_steps and "dir" in TRAIN_OBJ):
            print("HESITATE")
            self.HESITATE = True
        # if len_loop >= self.config_gap.time_steps and "gap" in TRAIN_OBJ:
        #     print("HESITATE")
        #     self.HESITATE = True
        # if len_loop_dir is None:
        #     if len_loop >= self.config_dir.time_steps and "dir" in TRAIN_OBJ:
        #         print("HESITATE")
        #         self.HESITATE = True
        # else:
        #     if len_loop_dir >= self.config_dir.time_steps and "dir" in TRAIN_OBJ:
        #         print("HESITATE")
        #         self.HESITATE = True
        if self.check_collision(env):
            print("CRASH by collision")
            self.CRASH = True
            self.SUCCESS = False
            self.HESITATE = False
        return self.SUCCESS or self.CRASH or self.HESITATE

    @staticmethod
    def check_collision(env):
        damage = False
        # dis_1, dis_2 = 2.5, 2.5
        for i in env.env_veh:
            dis_1 = env.ego.v * 0.85 + 2.5
            dis_2 = i.v * 0.75 + 2.5
            left = min(env.ego.y + 1, i.y + 1)
            right = max(env.ego.y - 1, i.y - 1)
            front = min(env.ego.x + dis_1, i.x + dis_2)
            rear = max(env.ego.x - dis_1, i.x - dis_2)
            w = max(0, left - right)
            h = max(0, front - rear)
            area = w * h
            if area > 0:
                damage = True
                break
        return damage

    @staticmethod
    def update_env(env):
        env.pre_update_env()
        env.pre_update_ego()
        env.update_env(0.1)

    @staticmethod
    def print_env(env):
        env_data = env.get_env()
        ego = env_data[0][:-1]
        front = env_data[1][:-1] if env.veh_num[1] == 1 else None
        env_veh = env_data[(1 + env.veh_num[1]):(1 + env.veh_num[1] + env.veh_num[0])]
        obj_veh = []
        obj_veh += [[i[0], i[2]] for i in env_veh]
        print("update_env:", env.veh_num, "ego:", ego, "front:", front)
        print(" "*11, "env_veh_left:", obj_veh)
        env_veh = env_data[(1 + env.veh_num[1] + env.veh_num[0]):]
        obj_veh = []
        obj_veh += [[i[0], i[2]] for i in env_veh]
        print(" "*11, "env_veh_right:", obj_veh)
        print(" "*11, "env_gap:", env.gap, "env_dir:", env.dir)

    def get_gap(self, env_data, single_buffer=None):
        state, s_mask = self.agent_gap.get_input(env_data)
        # print("gap state", state, s_mask)
        if sum(s_mask) == 0:
            command = [2, 0]
        else:
            # action = self.agent_gap.model_based_decision(state, s_mask, self.is_new[1])
            action = self.agent_gap.noise_based_decision(s_mask)
            command, a_mask = self.agent_gap.get_output(action, s_mask)
            # print("gap action", action, a_mask)
            if "gap" in TRAIN_OBJ:
                self.single_buffer.append({"st": state, "sm": s_mask, "at": action, "am": a_mask})
                if single_buffer is not None:
                    single_buffer.append({"st": state, "sm": s_mask, "at": action, "am": a_mask})
        # if self.eva_sys == 1:
        #     command = [2, 0]
        # elif self.eva_sys == 2:
        #     command = [2, 0]
        # elif self.eva_sys == 3:
        #     command = [2, 1]
        # elif self.eva_sys == 4:
        #     command = [2, 0]
        # elif self.eva_sys == 5:
        #     command = [2, 1]
        # elif self.eva_sys == 6:
        #     command = [2, 2]
        # elif self.eva_sys == 7:
        #     command = [2, 0]
        # elif self.eva_sys == 8:
        #     command = [2, 1]
        # elif self.eva_sys == 9:
        #     command = [2, 2]
        # elif self.eva_sys == 10:
        #     command = [2, 3]
        # elif self.eva_sys == 11:
        #     command = [2, 1]
        # elif self.eva_sys == 12:
        #     command = [2, 2]
        # elif self.eva_sys == 13:
        #     command = [2, 3]
        # elif self.eva_sys == 14:
        #     command = [2, 4]
        return command

    def get_direction(self, env_data, single_buffer=None):
        if "dir" in TRAIN_OBJ:
            state, s_mask = self.agent_dir.get_input(env_data)
            # print("dir state", np.shape(state))
            # action = self.agent_dir.model_based_decision(state, self.is_new[0])
            action = self.agent_dir.noise_based_decision()
            command, a_mask = self.agent_dir.get_output(action)
            # print("dir action", action, mask)
            self.single_buffer.append({"st": state, "at": action, "am": a_mask})
            if single_buffer is not None:
                single_buffer.append({"st": state, "at": action, "am": a_mask, "sm": s_mask})
        else:
            command = [1, 0]
        return command

    @staticmethod
    def dir_to_gap(veh_num, command, env_data):
        # command is [2], env_data is [veh_num, 4], veh_num is [l, m ,r]
        if command[0] == 1:
            if command[1] == 0:
                s = 1 + veh_num[0] + veh_num[1]
                del env_data[s:]
            else:
                s = 1 + veh_num[1]
                t = 1 + veh_num[0] + veh_num[1]
                del env_data[s: t]

    def init_evaluate_env(self):
        self.env.reset_ego(0, -5.1, 22.2, "99_1_-2")
        self.env.base_speed = 22.2
        x_bias = self.env.ego.x
        left, mid, right = "99_1_-1", "99_1_-2", "99_1_-3"
        y1, y2, y3 = -1.7, -5.1, -8.5
        env_veh = []
        self.eva_sys += 1
        print(self.eva_sys)
        if self.eva_sys == 1:
            env_veh.append([100 + x_bias, y2, 16.7, mid])
        elif self.eva_sys == 2:
            env_veh.append([100 + x_bias, y2, 16.7, mid])
            env_veh.append([-50 + x_bias, y1, 22.2, left])
        elif self.eva_sys == 3:
            env_veh.append([100 + x_bias, y2, 16.7, mid])
            env_veh.append([50 + x_bias, y1, 22.2, left])
        elif self.eva_sys == 4:
            env_veh.append([100 + x_bias, y2, 16.7, mid])
            env_veh.append([20 + x_bias, y1, 16.7, left])
            env_veh.append([-30 + x_bias, y1, 16.7, left])
        elif self.eva_sys == 5:
            env_veh.append([100 + x_bias, y2, 16.7, mid])
            env_veh.append([50 + x_bias, y1, 22.2, left])
            env_veh.append([-50 + x_bias, y1, 22.2, left])
        elif self.eva_sys == 6:
            env_veh.append([100 + x_bias, y2, 16.7, mid])
            env_veh.append([30 + x_bias, y1, 22.2, left])
            env_veh.append([-30 + x_bias, y1, 22.2, left])
        elif self.eva_sys == 7:
            env_veh.append([100 + x_bias, y2, 16.7, mid])
            env_veh.append([10 + x_bias, y1, 16.7, left])
            env_veh.append([-30 + x_bias, y1, 16.7, left])
            env_veh.append([-70 + x_bias, y1, 16.7, left])
        elif self.eva_sys == 8:
            env_veh.append([100 + x_bias, y2, 16.7, mid])
            env_veh.append([50 + x_bias, y1, 22.2, left])
            env_veh.append([-40 + x_bias, y1, 22.2, left])
            env_veh.append([-80 + x_bias, y1, 22.2, left])
        elif self.eva_sys == 9:
            env_veh.append([100 + x_bias, y2, 16.7, mid])
            env_veh.append([90 + x_bias, y1, 22.2, left])
            env_veh.append([40 + x_bias, y1, 22.2, left])
            env_veh.append([-50 + x_bias, y1, 22.2, left])
        elif self.eva_sys == 10:
            env_veh.append([100 + x_bias, y2, 16.7, mid])
            env_veh.append([80 + x_bias, y1, 22.2, left])
            env_veh.append([30 + x_bias, y1, 22.2, left])
            env_veh.append([-20 + x_bias, y1, 22.2, left])
        # elif self.eva_sys == 11:
        #     env_veh.append([100 + x_bias, y2, 16.7, mid])
        #     env_veh.append([90 + x_bias, y1, 16.7, left])
        #     env_veh.append([10 + x_bias, y1, 16.7, left])
        #     env_veh.append([-30 + x_bias, y1, 16.7, left])
        #     env_veh.append([-70 + x_bias, y1, 16.7, left])
        # elif self.eva_sys == 12:
        #     env_veh.append([100 + x_bias, y2, 16.7, mid])
        #     env_veh.append([80 + x_bias, y1, 22.2, left])
        #     env_veh.append([30 + x_bias, y1, 22.2, left])
        #     env_veh.append([-60 + x_bias, y1, 22.2, left])
        #     env_veh.append([-100 + x_bias, y1, 22.2, left])
        # elif self.eva_sys == 13:
        #     env_veh.append([100 + x_bias, y2, 16.7, mid])
        #     env_veh.append([90 + x_bias, y1, 22.2, left])
        #     env_veh.append([50 + x_bias, y1, 22.2, left])
        #     env_veh.append([-0.1 + x_bias, y1, 22.2, left])
        #     env_veh.append([-90 + x_bias, y1, 22.2, left])
        # elif self.eva_sys == 14:
        #     env_veh.append([100 + x_bias, y2, 16.7, mid])
        #     env_veh.append([90 + x_bias, y1, 22.2, left])
        #     env_veh.append([40 + x_bias, y1, 22.2, left])
        #     env_veh.append([-10 + x_bias, y1, 22.2, left])
        #     env_veh.append([-50 + x_bias, y1, 22.2, left])
        else:
            self.eva_sys = 1
            env_veh.append([100 + x_bias, y2, 16.7, mid])
        self.env.reset_env(env_veh)
        self.print_env(self.env)

    def init_env(self):
        # self.env.base_speed = np.random.uniform(8.0, 21.4, 1)[0]
        self.env.reset_ego(0, -5.1, np.random.uniform(20.8, 22.2, 1)[0], "99_1_-2")
        x_bias = self.env.ego.x
        left, mid, right = "99_1_-1", "99_1_-2", "99_1_-3"
        y1, y2, y3 = -1.7, -5.1, -8.5
        veh_f = np.random.choice([0, 1], 1, p=[0.1, 0.9])[0]
        veh_lf = np.random.choice([0, 1, 2], 1, p=[0.2, 0.4, 0.4])[0]
        veh_lr = np.random.choice([0, 1, 2], 1, p=[0.2, 0.4, 0.4])[0]
        veh_rf = np.random.choice([0, 1, 2], 1, p=[0.2, 0.4, 0.4])[0]
        veh_rr = np.random.choice([0, 1, 2], 1, p=[0.2, 0.4, 0.4])[0]
        # veh_lf = 2
        # veh_lr = 2
        # veh_rf = 2
        # veh_rr = 2

        # self.env.base_speed = np.random.choice([17, 18, 19, 20, 21, 22], 1, p=[0.4, 0.05, 0.05, 0.05, 0.05, 0.4])[0]
        self.env.base_speed = np.random.uniform(16.7, 22.2, 1)[0]
        env_ran_l, env_ran_h = self.env.base_speed * 1.5, 100
        speed_low = max(self.env.base_speed - 0.5, 16.7)
        speed_high = min(self.env.base_speed + 0.5, 22.2)

        def part_env(num, low, high, limit, yy, lane, veh_list):
            if num == 1:
                xx = round(np.random.uniform(low, min(high, limit), 1)[0], 1) + x_bias
                # vv = round(np.random.uniform(self.env.base_speed - 2.6, self.env.base_speed + 1.4, 1)[0], 1)
                vv = round(np.random.uniform(speed_low, speed_high, 1)[0], 1)
                veh_list.append([xx, yy, vv, lane])
                bottom = xx - env_ran_l
            elif num == 2:
                dis = np.random.uniform(env_ran_l, min(env_ran_h - 5, env_ran_h + limit - 1), 1)[0]
                xx = round(np.random.uniform(low + dis, min(high, limit), 1)[0], 1) + x_bias
                # vv = round(np.random.uniform(self.env.base_speed - 2.6, self.env.base_speed + 1.4, 1)[0], 1)
                vv = round(np.random.uniform(speed_low, speed_high, 1)[0], 1)
                veh_list.append([xx, yy, vv, lane])
                xx -= dis
                # vv = round(np.random.uniform(self.env.base_speed - 2.6, self.env.base_speed + 1.4, 1)[0], 1)
                vv = round(np.random.uniform(speed_low, speed_high, 1)[0], 1)
                veh_list.append([xx, yy, vv, lane])
                bottom = xx - env_ran_l
            else:
                bottom = env_ran_h
            return bottom

        env_veh = []
        if veh_f == 1:
            # x = round(np.random.uniform(self.env.base_speed * 1.2 + 10, env_ran_h, 1)[0], 1) + x_bias
            x = round(np.random.uniform(90, env_ran_h, 1)[0], 1) + x_bias
            # v = round(np.random.uniform(self.env.base_speed - 3.4, self.env.base_speed - 1.2, 1)[0], 1)
            v = round(np.random.uniform(15.3, 18.1, 1)[0], 1)
            env_veh.append([x, y2, v, mid])
        limit = part_env(veh_lf, 0.1, env_ran_h, env_ran_h, y1, left, env_veh)
        _ = part_env(veh_lr, -env_ran_h, -0.1, limit, y1, left, env_veh)
        limit = part_env(veh_rf, 0.1, env_ran_h, env_ran_h, y3, right, env_veh)
        _ = part_env(veh_rr, -env_ran_h, -0.1, limit, y3, right, env_veh)
        self.env.reset_env(env_veh)
        self.print_env(self.env)

    def init_flag(self):
        self.flag_dir = True
        self.flag_gap = False
        self.is_new = [True, True]
        self.dir_command = [0, 0]
        self.gap_command = [0, 0]

if __name__ == '__main__':
    a = HRLNode()
    a.get_init_in_HRL()
    a.test_model()
    # a.get_init_in_HWP()
    # result = a.get_output(1)
    # print(result)
