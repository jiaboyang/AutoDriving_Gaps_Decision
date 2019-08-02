from __future__ import print_function
import numpy as np

SPEED_LIMIT = 22.2

class Veh(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.v = -1
        self.t = str()
        self.dy = 0
        self.a = 0

    def set(self, x, y, v, t):
        self.x = x
        self.y = y
        self.v = v
        self.t = t

    def get(self):
        return [self.x, self.y, self.v, self.t]

    def update(self, time, base_speed):
        self.x += time * self.v
        self.y = np.clip(self.y + time * self.dy, -8.5, -1.7)
        # self.v = np.clip(self.v + time * self.a, base_speed - 4.6, base_speed + 5.2)
        self.v = np.clip(self.v + time * self.a, 13.9, SPEED_LIMIT)
        if self.y >= -1.71 and self.t == "99_1_-2":
            self.t = "99_1_-1"
        elif self.y <= -8.49 and self.t == "99_1_-2":
            self.t = "99_1_-3"
        # print "show update", self.x, self.y, self.v, self.l

    def set_delta_y(self, dy):
        # left -> right : -, right -> left : +
        self.dy = dy

    def set_a(self, a):
        self.a = a


class Environment(object):
    def __init__(self):
        self.ego = Veh()
        self.env_veh = []
        self.veh_num = [0, 0, 0]
        self.dy = None
        self.t = None
        self.dir = [0, 0]
        self.gap = [0, 0]
        self.base_speed = 0

    def reset_ego(self, x, y, v, l):
        self.ego.set(x, y, v, l)
        self.ego.set_a(0)
        self.ego.set_delta_y(0)

    def reset_env(self, env_veh):
        # input is absolute coord
        self.gap = [0, 0]
        self.dir = [0, 0]
        self.env_veh = []
        self.veh_num = [0, 0, 0]
        for i in env_veh:
            veh = Veh()
            veh.set(i[0], i[1], i[2], i[3])
            self.env_veh.append(veh)
            if i[3] == "99_1_-1":
                self.veh_num[0] += 1
            elif i[3] == "99_1_-2":
                self.veh_num[1] += 1
            elif i[3] == "99_1_-3":
                self.veh_num[2] += 1

    def get_env(self):
        data = list()
        data.append(self.ego.get())
        for i in self.env_veh:
            veh = i.get()
            data.append([veh[0] - self.ego.x, veh[1] - self.ego.y, veh[2], veh[3]])
        # env veh is relative coord, ego is absolute coord
        return data

    def pre_update_env(self):
        def update_lane(lane_str, st_id):
            lane = []
            for veh in self.env_veh:
                if veh.t == lane_str:
                    lane.append(veh.x - self.ego.x)
            last_x = 10000
            last_v = SPEED_LIMIT
            for j, veh_x in enumerate(lane):
                veh_v = self.env_veh[j + st_id].v
                if veh_v <= last_v:
                    self.env_veh[j + st_id].set_a(np.random.uniform(-0.3, 0.3, 1)[0])
                else:
                    delta_s = last_x - veh_x
                    delta_v = (veh_v - last_v) ** 2
                    min_acc = delta_v / (2 * delta_s)
                    self.env_veh[j + st_id].set_a(np.random.uniform(-min_acc - 1, -min_acc, 1)[0])
                last_x = veh_x
                last_v = veh_v

        if len(self.env_veh) > 0:
            if self.env_veh[0].t == "99_1_-2":
                up_down = np.random.choice([-1, 1], 1, p=[0.3, 0.7])[0]
                if self.env_veh[0].v <= SPEED_LIMIT:
                    acc = np.random.uniform(0, 1, 1)[0]
                    self.env_veh[0].set_a(up_down * acc)
                else:
                    self.env_veh[0].set_a(-0.5)
            update_lane("99_1_-1", self.veh_num[1])
            update_lane("99_1_-3", self.veh_num[0] + self.veh_num[1])

    def update_env(self, time):
        self.ego.update(time, self.base_speed)
        for i in self.env_veh:
            i.update(time, self.base_speed)

        def check_veh_num(lane_str, st_id, t_id):
            f, r = 0, 0
            for j, veh in enumerate(self.env_veh):
                if veh.t == lane_str:
                    if veh.x >= self.ego.x:
                        f += 1
                    else:
                        r += 1
            if f > 2:
                del self.env_veh[st_id]
                self.veh_num[t_id] -= 1
                if self.gap[1] != 0:
                    if self.dir[1] == 0 and lane_str == "99_1_-1":
                        self.gap[1] -= 1
                    elif self.dir[1] == 1 and lane_str == "99_1_-3":
                        self.gap[1] -= 1
                    else:
                        pass
            if r > 2:
                del self.env_veh[st_id + self.veh_num[t_id] - 1]
                if self.gap[1] == self.veh_num[0] and self.dir[1] == 0:
                    self.gap[1] -= 1
                elif self.gap[1] == self.veh_num[2] and self.dir[1] == 1:
                    self.gap[1] -= 1
                self.veh_num[t_id] -= 1
        check_veh_num("99_1_-1", self.veh_num[1], 0)
        check_veh_num("99_1_-3", self.veh_num[0] + self.veh_num[1], 2)

    def set_ego_command(self, direction, gap):
        max_acc = 1.5
        self.dir = direction
        self.gap = gap
        if self.dir[0] == 1 and self.gap[0] == 2:
            delta_s, object_v = self.get_delta()
            delta_v = object_v - self.ego.v
            # print("test", delta_s, delta_v)
            if delta_s >= 0:
                sign = np.sqrt(4 * delta_v ** 2 + 6 * delta_s)
                # print("test", (4 * delta_v - 2 * sign) / (2 * max_acc), (4 * delta_v + 2 * sign) / (2 * max_acc))
                if (4 * delta_v - 2 * sign) / (2 * max_acc) >= 8:
                    self.t = min((4 * delta_v - 2 * sign) / (2 * max_acc), 16)
                else:
                    self.t = min(max((abs(delta_v) + 2 * sign) / (2 * max_acc), 8), 16)
            else:
                sign = np.sqrt(4 * delta_v ** 2 - 6 * delta_s)
                self.t = min(max((2 * sign - 4 * delta_v) / (2 * max_acc), 8), 16)
            if delta_s == 0:
                single_dy = 3.4 * 3 / self.t if self.dir[1] == 0 else -3.4 * 3 / self.t
                self.dy = [0] * int(20 * self.t / 3) + [single_dy] * int(self.t * 10 / 3 + 1)
            else:
                single_dy = 3.4 * 5 / self.t if self.dir[1] == 0 else -3.4 * 5 / self.t
                self.dy = [0] * int(40 * self.t / 5) + [single_dy] * int(self.t * 10 / 5 + 1)
            # print("command in env-ego", self.t, self.dy)

    def get_delta(self):
        coef1 = 1.8
        coef2 = 1.6
        bias = 9
        object_lane = []
        if self.dir[1] == 0:  # dir = 0 or 1
            env_veh_t = "99_1_-1"
        else:
            env_veh_t = "99_1_-3"
        for i in self.env_veh:
            if i.t == env_veh_t:
                object_lane.append([i.x - self.ego.x, i.v])
        if len(object_lane) > 0:
            if self.gap[1] == 0:
                delta_s = max(0, object_lane[0][0] + object_lane[0][1] * coef1 + bias)
                object_v = max(object_lane[0][1], self.ego.v)
            elif self.gap[1] >= len(object_lane):
                delta_s = min(0, object_lane[-1][0] - self.ego.v * coef2 - bias)
                object_v = min(object_lane[-1][1], self.ego.v)
            else:
                if (object_lane[self.gap[1] - 1][0] - self.ego.v * coef2 - bias >= 0) and \
                        (object_lane[self.gap[1]][0] + object_lane[self.gap[1]][1] * coef1 + bias <= 0):
                    delta_s = 0
                    object_v = object_lane[self.gap[1] - 1][1] + 1
                elif abs(object_lane[self.gap[1] - 1][0] - self.ego.v * coef2 - bias) <= \
                        abs(object_lane[self.gap[1]][0] + object_lane[self.gap[1]][1] * coef1 + bias):
                    delta_s = object_lane[self.gap[1] - 1][0] - self.ego.v * coef2 - bias
                    if object_lane[self.gap[1]][0] + object_lane[self.gap[1]][1] * coef1 + bias <= 0:
                        object_v = min(object_lane[self.gap[1]][1], object_lane[self.gap[1] - 1][1]) - 2
                    else:
                        object_v = max(object_lane[self.gap[1]][1], object_lane[self.gap[1] - 1][1]) + 2
                else:
                    delta_s = object_lane[self.gap[1]][0] + object_lane[self.gap[1]][1] * coef1 + bias
                    if object_lane[self.gap[1] - 1][0] - self.ego.v * coef2 - bias >= 0:
                        object_v = max(object_lane[self.gap[1]][1], object_lane[self.gap[1] - 1][1]) + \
                                   object_lane[self.gap[1] - 1][0] / self.ego.v
                    else:
                        object_v = min(object_lane[self.gap[1]][1], object_lane[self.gap[1] - 1][1]) - 2
        else:
            delta_s = 0
            object_v = min(self.ego.v + 1, SPEED_LIMIT + 1)
        return delta_s, object_v

    def pre_update_ego(self):
        if self.gap[0] == 0:
            if self.veh_num[1] == 1:
                object_v = self.env_veh[0].v
                delta_s = self.env_veh[0].x - self.ego.x
            else:
                object_v = SPEED_LIMIT
                delta_s = 100
            acc = (self.ego.v - object_v) ** 2 / (2 * delta_s)
            if object_v < self.ego.v:
                acc *= -1
            self.ego.set_a(acc)
            # print("0, follow")
            return True
        elif self.gap[0] == 1:
            delta_s, object_v = self.get_delta()
            acc = self.fakeacc(SPEED_LIMIT, delta_s, object_v, self.ego.v)
            self.ego.set_a(acc)
            # print("1, adjust")
            return True
        else:
            delta_s, object_v = self.get_delta()
            # print(delta_s, object_v, self.t)
            acc = self.fakeacc(SPEED_LIMIT, delta_s, object_v, self.ego.v)
            dy = self.dy.pop(0)
            self.t -= 0.1
            # print("2, implement")
            if acc >= 2 or acc <= -5:
                self.ego.set_a(0)
                self.ego.set_delta_y(0)
                return False
            else:
                self.ego.set_a(acc)
                self.ego.set_delta_y(dy)
                return True

    @staticmethod
    def fakeacc(speed_limit, position, targetspeed, egospeed):
        if position >= 0:
            if egospeed > targetspeed:
                safeposition = position - (egospeed - targetspeed) * (egospeed - targetspeed) / 2 / 1.5
            else:
                safeposition = position
        else:
            if egospeed < targetspeed:
                safeposition = position + (egospeed - targetspeed) * (egospeed - targetspeed) / 2 / 1
            else:
                safeposition = position
        if speed_limit > targetspeed:
            acc_speed = 0.9 * (targetspeed - egospeed)
        else:
            acc_speed = 0.9 * (speed_limit - egospeed)
        acc_dist = 0.45 * safeposition

        if acc_speed > 1.5:
            acc_speed = 1.5
        elif acc_speed < -4.0:
            acc_speed = -4.0

        if acc_dist > 1.5:
            acc_dist = 1.5
        elif acc_dist < -4.0:
            acc_dist = -4.0

        acc = acc_speed + acc_dist
        if egospeed + acc * 0.1 < 0.0:
            acc = -egospeed * 10.0
        elif egospeed + acc * 0.1 > speed_limit:
            acc = (speed_limit - egospeed) * 10.0
        if acc > 1.5:
            acc = 1.5
        elif acc < -1.5:
            acc = -1.5
        return acc
