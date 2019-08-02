

class ConfigDirection(object):
    version = 1

    tree_search = True
    learning_rate = 1e-4
    epsilon = 0.2
    batch_size = 120
    time_steps = 20

    feature_dim = 96
    filter_size = [1, 2, 3, 4]
    num_filters = [48, 96, 96, 48]
    num_layers = 3
    hidden_dim = 384

    pre_frame_num = 1
    veh_num = 4
    veh_dim = 3
    front_dim = 4
    lat_dim = 2
    lon_dim = 2
    output_dim = lat_dim + lon_dim


class ConfigGap(object):
    version = 1

    tree_search = True
    learning_rate = 1e-4
    epsilon = 0.2
    batch_size = 120
    time_steps = 20
    
    feature_dim = 128
    num_layers = 3
    hidden_dim = 384

    pre_frame_num = 1
    veh_num = 6
    gap_num = 5
    veh_dim = 3
    front_dim = 4
    lat_dim = 3
    lon_dim = 5
    output_dim = lat_dim + lon_dim
