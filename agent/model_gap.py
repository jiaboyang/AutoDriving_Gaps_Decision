import tensorflow as tf


class TrainMethodTimeSeriesAttention(object):
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess
        self.init_variable()
        self.actor = ActionTimeSeriesAttention(self.config, self.config.batch_size, self.config.time_steps, True)
        self.basic_net_out()
        self.PPO_train_method()

    def init_variable(self):
        with tf.name_scope("input"):
            self.state = list()
            self.state_mask = list()
            for _ in range(self.config.time_steps):
                self.state.append(tf.placeholder(tf.float32, [self.config.veh_num, self.config.batch_size, self.config.veh_dim]))
                self.state_mask.append(tf.placeholder(tf.float32, [self.config.batch_size, self.config.gap_num]))
            self.action_mask = tf.placeholder(tf.float32, shape=[self.config.time_steps - 1, self.config.batch_size, self.config.output_dim])
            self.old_action = tf.placeholder(tf.float32, shape=[self.config.time_steps - 1, self.config.batch_size, self.config.output_dim])
            self.advantage_lat = tf.placeholder(tf.float32, shape=[self.config.time_steps - 1, self.config.batch_size])
            self.advantage_lon = tf.placeholder(tf.float32, shape=[self.config.time_steps - 1, self.config.batch_size])

    def basic_net_out(self):
        hidden = tf.constant(0, dtype=tf.float32, shape=[self.config.num_layers, self.config.batch_size, self.config.hidden_dim])
        cell = tf.constant(0, dtype=tf.float32, shape=[self.config.num_layers, self.config.batch_size, self.config.hidden_dim])
        new_traj = tf.constant(True, dtype=tf.bool, shape=())
        _, _, self.action = self.actor.action_network(self.state, self.state_mask, hidden, cell, new_traj, "gap_action")
        self.net_vars = [var for var in tf.trainable_variables() if var.name.startswith('gap_action')]

    def PPO_train_method(self):
        with tf.name_scope("PPO_loss"):
            action_lat = tf.slice(self.action[0], [1, 0, 0], [self.config.time_steps - 1, self.config.batch_size, self.config.lat_dim])
            action_lon = tf.slice(self.action[1], [1, 0, 0], [self.config.time_steps - 1, self.config.batch_size, self.config.lon_dim])
            mask_lat = tf.slice(self.action_mask, [0, 0, 0], [self.config.time_steps - 1, self.config.batch_size, self.config.lat_dim])
            mask_lon = tf.slice(self.action_mask, [0, 0, self.config.lat_dim], [self.config.time_steps - 1, self.config.batch_size, self.config.lon_dim])
            old_action_lat = tf.slice(self.old_action, [0, 0, 0], [self.config.time_steps - 1, self.config.batch_size, self.config.lat_dim])
            old_action_lon = tf.slice(self.old_action, [0, 0, self.config.lat_dim], [self.config.time_steps - 1, self.config.batch_size, self.config.lon_dim])
            cur_responsible_lat = tf.log(tf.reduce_sum(action_lat * mask_lat, 2) + 1e-6)
            cur_responsible_lon = tf.log(tf.reduce_sum(action_lon * mask_lon, 2) + 1e-6)
            old_responsible_lat = tf.log(tf.reduce_sum(old_action_lat * mask_lat, 2) + 1e-6)
            old_responsible_lon = tf.log(tf.reduce_sum(old_action_lon * mask_lon, 2) + 1e-6)
            # cur/old ratio is [time_steps - 1, batch_size]
            ratio_lat = tf.exp(cur_responsible_lat - old_responsible_lat)
            ratio_lon = tf.exp(cur_responsible_lon - old_responsible_lon)
            clip_ratio_lat = tf.clip_by_value(ratio_lat, 1 - self.config.epsilon, 1 + self.config.epsilon)
            clip_ratio_lon = tf.clip_by_value(ratio_lon, 1 - self.config.epsilon, 1 + self.config.epsilon)
            ppo_loss_lat = tf.negative(tf.reduce_mean(tf.reduce_min([tf.multiply(ratio_lat, self.advantage_lat), tf.multiply(clip_ratio_lat, self.advantage_lat)], 0)))
            ppo_loss_lon = tf.negative(tf.reduce_mean(tf.reduce_min([tf.multiply(ratio_lon, self.advantage_lon), tf.multiply(clip_ratio_lon, self.advantage_lon)], 0)))
            self.ppo_loss = ppo_loss_lat + ppo_loss_lon
        self.ppo_optimizer = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.ppo_loss, var_list=self.net_vars)

    def actor_ppo_train(self, feed_dict):
        return self.sess.run([self.ppo_optimizer, self.ppo_loss, self.action], feed_dict=feed_dict)


class TestMethodTimeSeriesAttention(object):
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess
        self.init_variable()
        self.actor = ActionTimeSeriesAttention(self.config, 1, 1, False)
        self.basic_net_out()

    def init_variable(self):
        with tf.name_scope("input"):
            self.state = [tf.placeholder(tf.float32, [self.config.veh_num, 1, self.config.veh_dim])]
            self.mask = [tf.placeholder(tf.float32, [1, self.config.gap_num])]
            self.init_cell = tf.placeholder(tf.float32, [self.config.num_layers, 1, self.config.hidden_dim])
            self.init_hidden = tf.placeholder(tf.float32, [self.config.num_layers, 1, self.config.hidden_dim])
            self.new_traj = tf.placeholder(tf.bool, shape=())

    def basic_net_out(self):
        hidden, cell, action = self.actor.action_network(self.state, self.mask, self.init_hidden, self.init_cell, self.new_traj, "gap_action_env")
        self.net_vars = [var for var in tf.trainable_variables() if var.name.startswith('gap_action_env')]
        # self.action is [output_dim]
        self.action = tf.squeeze(tf.concat(action, 2))
        self.hidden = tf.reshape(hidden, [self.config.num_layers, 1, self.config.hidden_dim])
        self.cell = tf.reshape(cell, [self.config.num_layers, 1, self.config.hidden_dim])

    def actor_run(self, feed_dict):
        return self.sess.run([self.hidden, self.cell, self.action], feed_dict=feed_dict)
        

class ActionTimeSeriesAttention(object):
    def __init__(self, config, batch_size, time_steps, is_train):
        self.config = config
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.is_train = is_train

    def action_network(self, state, mask, hidden, cell, new_traj, name):
        with tf.variable_scope(name):
            outputs, cell_states, hidden_states = [], [], []
            for id, value in enumerate(zip(state, mask)):
                reuse = True if id >= 1 else False
                input, mask_i = value
                feature_input = tf.slice(input, [2, 0, 0], [4, self.batch_size, self.config.veh_dim])
                ego_input = tf.slice(input, [0, 0, 0], [2, self.batch_size, self.config.veh_dim])
                code_feature = self.bi_lstm(feature_input, mask_i, reuse=reuse)
                code_mid = self.dense(ego_input, reuse=reuse)
                new_traj = tf.cond(tf.cast(id == 0, tf.bool), lambda: new_traj, lambda: tf.constant(False, dtype=tf.bool, shape=()))
                code = self.attention(code_feature, hidden[0], mask_i, new_traj, reuse=reuse)
                code = tf.concat([code, code_mid], 1)
                # code is [batch_size, hidden_dim]
                hidden, cell, output = self.mutilayer_lstm(code, hidden, cell, reuse=reuse)
                outputs.append(output)
                hidden_states.append(hidden)
                cell_states.append(cell)
            # hidden_states and cell_states is [time_steps, num_layers, batch_size, hidden_dim]
            # outputs is [time_steps, batch_size, hidden_dim]
            lat, lon = self.twohead_dense(outputs, mask)
        return hidden_states, cell_states, [lat, lon]

    def twohead_dense(self, input, mask):
        # input and mask is [time_steps, batch_size, hidden_dim]
        mask = tf.reshape(mask, [self.time_steps * self.batch_size, self.config.gap_num])
        input = tf.reshape(input, [self.time_steps * self.batch_size, self.config.hidden_dim])
        with tf.variable_scope('output'):
            weight = tf.get_variable(name='w', shape=[self.config.hidden_dim, self.config.output_dim],
                                     initializer=tf.truncated_normal_initializer(0, 0.01))
            bias = tf.get_variable(name='b', shape=[self.config.output_dim], initializer=tf.constant_initializer(0.1))
            output = tf.nn.xw_plus_b(input, weight, bias)
            # output is [time_steps*batch_size, output_dim]
            output_one = tf.reshape(tf.nn.softmax(tf.slice(output, [0, 0], [self.time_steps*self.batch_size, self.config.lat_dim])),
                                    [self.time_steps, self.batch_size, self.config.lat_dim])
            output_two = tf.slice(output, [0, self.config.lat_dim], [self.time_steps*self.batch_size, self.config.lon_dim])
            output_two = output_two - tf.expand_dims(tf.reduce_max(output_two, 1), 1)
            output_two = tf.reshape(((tf.exp(output_two) + 1e-6) * mask + 1e-6) / tf.expand_dims(tf.reduce_sum((tf.exp(output_two) + 1e-6) * mask + 1e-6, 1), 1),
                                    [self.time_steps, self.batch_size, self.config.lon_dim])
        return output_one, output_two

    def attention(self, codes, weight, mask, new_traj, reuse):
        # codes is [gap_num, batch_size, feature_dim*2], weight is [batch_size, hidden_dim], mask is [batch_size, gap_num]
        with tf.variable_scope('attention', reuse=reuse):
            mask = tf.transpose(mask, [1, 0])
            w = tf.get_variable(name='w', shape=[self.config.hidden_dim, self.config.feature_dim * 2], initializer=tf.truncated_normal_initializer(0, 0.01))
            b = tf.get_variable(name='b', shape=[self.config.feature_dim * 2], initializer=tf.constant_initializer(0.1))
            weight = tf.tanh(tf.nn.xw_plus_b(weight, w, b))
            # scores and distribution is [gap_num, batch_size]
            scores = tf.reduce_sum(tf.multiply(codes, tf.expand_dims(weight, 0)), 2)
            scores = tf.cond(new_traj, lambda: tf.constant(0, dtype=tf.float32, shape=[self.config.gap_num, self.batch_size]), lambda: scores)
            distribution = (tf.exp(scores) * mask + 1e-6) / tf.reduce_sum(tf.exp(scores) * mask + 1e-6, 0)
            output = tf.reduce_sum(tf.multiply(codes, tf.expand_dims(distribution, 2)), 0)
            # output is [batch_size, feature_dim*2]
        return output

    def dense(self, input, reuse):
        # input is [2, batch_size, veh_dim]
        input = tf.transpose(input, [1, 0, 2])
        input = tf.reshape(input, [self.batch_size, 2 * self.config.veh_dim])
        input = tf.slice(input, [0, 0], [self.batch_size, 4])
        with tf.variable_scope('dense', reuse=reuse):
            weight = tf.get_variable(name='w', shape=[self.config.front_dim, self.config.feature_dim], initializer=tf.truncated_normal_initializer(0, 0.01))
            bias = tf.get_variable(name='b', shape=[self.config.feature_dim], initializer=tf.constant_initializer(0.1))
            output = tf.nn.relu(tf.nn.xw_plus_b(input, weight, bias))
            # output is [batch_size, feature_dim]
        return output

    def bilstm_cell(self, input, hidden, cell, fw_bk):
        # input is [batch_size, veh_dim], hidden and cell is [batch_size, feature_dim]
        if self.is_train:
            if fw_bk:
                all_gates = tf.nn.dropout(tf.matmul(input, self.ifcox_f2r), 0.5) + tf.matmul(hidden, self.ifcoh_f2r) + self.ifcob_f2r
            else:
                all_gates = tf.nn.dropout(tf.matmul(input, self.ifcox_r2f), 0.5) + tf.matmul(hidden, self.ifcoh_r2f) + self.ifcob_r2f
        else:
            if fw_bk:
                all_gates = tf.matmul(input, self.ifcox_f2r) + tf.matmul(hidden, self.ifcoh_f2r) + self.ifcob_f2r
            else:
                all_gates = tf.matmul(input, self.ifcox_r2f) + tf.matmul(hidden, self.ifcoh_r2f) + self.ifcob_r2f
        input_gate = tf.sigmoid(all_gates[:, 0:self.config.feature_dim])
        forget_gate = tf.sigmoid(all_gates[:, self.config.feature_dim: 2 * self.config.feature_dim])
        update = tf.tanh(all_gates[:, 2 * self.config.feature_dim: 3 * self.config.feature_dim])
        output_gate = tf.sigmoid(all_gates[:, 3 * self.config.feature_dim:])
        cell_state = forget_gate * cell + input_gate * update
        hidden_state = output_gate * tf.tanh(cell_state)
        # hidden_state and cell_state is [batch_size, feature_dim]
        return hidden_state, cell_state

    def bi_lstm(self, input, mask, reuse):
        # input is [veh_num, batch_size, veh_dim]
        with tf.variable_scope('bi-lstm', reuse=reuse):
            with tf.variable_scope('f2r'):
                self.ifcox_f2r = tf.get_variable(name='x', shape=[self.config.veh_dim, 4*self.config.feature_dim], initializer=tf.truncated_normal_initializer(0, 0.01))
                self.ifcoh_f2r = tf.get_variable(name='h', shape=[self.config.feature_dim, 4*self.config.feature_dim], initializer=tf.truncated_normal_initializer(0, 0.01))
                self.ifcob_f2r = tf.get_variable(name='b', shape=[1, 4*self.config.feature_dim], initializer=tf.constant_initializer(0.1))
            with tf.variable_scope('r2f'):
                self.ifcox_r2f = tf.get_variable(name='x', shape=[self.config.veh_dim, 4*self.config.feature_dim], initializer=tf.truncated_normal_initializer(0, 0.01))
                self.ifcoh_r2f = tf.get_variable(name='h', shape=[self.config.feature_dim, 4*self.config.feature_dim], initializer=tf.truncated_normal_initializer(0, 0.01))
                self.ifcob_r2f = tf.get_variable(name='b', shape=[1, 4*self.config.feature_dim], initializer=tf.constant_initializer(0.1))

        hidden = tf.constant(0, dtype=tf.float32, shape=[self.batch_size, self.config.feature_dim])
        cell = tf.constant(0, dtype=tf.float32, shape=[self.batch_size, self.config.feature_dim])
        fw_input = tf.concat([tf.zeros([1, self.batch_size, self.config.veh_dim]), input], 0)
        bk_input = tf.reverse(tf.concat([input, tf.zeros([1, self.batch_size, self.config.veh_dim])], 0), [0])
        # fw, bk input is [gap_num, batch_size, veh_dim]
        i = tf.constant(0)

        def cond(a, b, c, d, e, f):
            return a < self.config.gap_num

        def body_fw(a, b, c, d, e, f):
            c, d = self.bilstm_cell(b[a], c, d, True)
            c *= f[:, a:a+1]
            d *= f[:, a:a+1]
            e = tf.concat([e, tf.expand_dims(c, 0)], 0)
            a += 1
            return a, b, c, d, e, f

        def body_bk(a, b, c, d, e, f):
            c, d = self.bilstm_cell(b[a], c, d, False)
            c *= f[:, a:a+1]
            d *= f[:, a:a+1]
            e = tf.concat([e, tf.expand_dims(c, 0)], 0)
            a += 1
            return a, b, c, d, e, f

        _, _, _, _, fw_hidden, _ = tf.while_loop(cond, body_fw, [i, fw_input, hidden, cell, tf.zeros([1, self.batch_size, self.config.feature_dim]), mask],
                                              shape_invariants=[i.get_shape(), fw_input.get_shape(), hidden.get_shape(), cell.get_shape(), tf.TensorShape([None, self.batch_size, self.config.feature_dim]), mask.get_shape()])
        _, _, _, _, bk_hidden, _ = tf.while_loop(cond, body_bk, [i, bk_input, hidden, cell, tf.zeros([1, self.batch_size, self.config.feature_dim]), tf.reverse(mask, [1])],
                                              shape_invariants=[i.get_shape(), bk_input.get_shape(), hidden.get_shape(), cell.get_shape(), tf.TensorShape([None, self.batch_size, self.config.feature_dim]), mask.get_shape()])
        fw_hidden = tf.slice(fw_hidden, [1, 0, 0], [self.config.gap_num, self.batch_size, self.config.feature_dim])
        bk_hidden = tf.slice(bk_hidden, [1, 0, 0], [self.config.gap_num, self.batch_size, self.config.feature_dim])
        feature_code = tf.concat([fw_hidden, tf.reverse(bk_hidden, [0])], 2)
        # feature_code is [gap_num, batch_size, feature_dim * 2]
        return feature_code

    def lstm_cell(self, input, hidden, cell, layer_id):
        if self.is_train:
            all_gates = tf.nn.dropout(tf.matmul(input, self.ifcox[layer_id]), 0.5) + tf.matmul(hidden, self.ifcoh[layer_id]) + self.ifcob[layer_id]
        else:
            all_gates = tf.matmul(input, self.ifcox[layer_id]) + tf.matmul(hidden, self.ifcoh[layer_id]) + self.ifcob[layer_id]
        # all_gates is [batch_size, 4*hidden_dim]
        input_gate = tf.sigmoid(all_gates[:, 0:self.config.hidden_dim])
        forget_gate = tf.sigmoid(all_gates[:, self.config.hidden_dim: 2*self.config.hidden_dim])
        output_gate = tf.sigmoid(all_gates[:, 2*self.config.hidden_dim: 3*self.config.hidden_dim])
        update = tf.tanh(all_gates[:, 3*self.config.hidden_dim:])
        cell_state = forget_gate * cell + input_gate * update
        hidden_state = output_gate * tf.tanh(cell_state)
        return hidden_state, cell_state

    def mutilayer_lstm(self, input, hiddens, cells, reuse):
        # hiddens and cells is [num_layers, batch_size, hidden_dim]
        # input is [batch_size, hidden_dim]
        with tf.variable_scope('lstm', reuse=reuse):
            self.ifcox, self.ifcoh, self.ifcob = [], [], []
            for i in range(self.config.num_layers):
                self.ifcox.append(tf.get_variable(name='x'+str(i), shape=[self.config.hidden_dim, 4*self.config.hidden_dim], initializer=tf.truncated_normal_initializer(0, 0.01)))
                self.ifcoh.append(tf.get_variable(name='h'+str(i), shape=[self.config.hidden_dim, 4*self.config.hidden_dim], initializer=tf.truncated_normal_initializer(0, 0.01)))
                self.ifcob.append(tf.get_variable(name='b'+str(i), shape=[1, 4*self.config.hidden_dim], initializer=tf.constant_initializer(0.1)))
            hidden_out = []
            cell_out = []
            for i in range(self.config.num_layers):
                hidden, cell = self.lstm_cell(input, hiddens[i], cells[i], i)
                hidden_out.append(hidden)
                cell_out.append(cell)
                input = hidden
        # hidden_out and cell_out is [num_layers, batch_size, hidden_dim]
        # hidden is [batch_size, hidden_dim]
        return tf.convert_to_tensor(hidden_out), tf.convert_to_tensor(cell_out), hidden
