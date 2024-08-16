results_folder_name = 'results/summer'
maximum_number_of_frames = 25
resx = 768
resy = 432
iters_num = 25000
samples_batch = 500000
data_folder = '/data/users/mpilligua/hashing-nvd/data/kite-walk/'
uv_mapping_scales = [0.9, 0.9, 0.9, 0.6]
pretrain_iter_number = 50
load_checkpoint = False
checkpoint_path = ''
folder_suffix = 'kite-walk_chg_enc'
use_alpha_gt = False

logger = dict(
    period = 25001,
    log_time = True,
    log_loss = True,
    log_alpha = True)

evaluation = dict(
    interval = 200,
    interval_save = 1000,
    samples_batch = 500000)

losses = dict(
    rgb = dict(
        weight = 5),
    gradient = dict(
        weight = 1),
    sparsity = dict(
        weight = 1),
    alpha_bootstrapping = dict(
        weight = 2,
        stop_iteration = 25000),
    alpha_reg = dict(
        weight = 0.1),
    flow_alpha = dict(
        weight = 0.05),
    optical_flow = dict(
        weight = 0.01),
    rigidity = dict(
        weight = 0.001,
        derivative_amount = 1),
    global_rigidity = dict(
        weight = [0.005, 0.05],
        stop_iteration = 5000,
        derivative_amount = 100),
    residual_reg = dict(
        weight = 0.5),
    residual_consistent = dict(
        weight = 0.1))

config_xyt = {
    'module': 'gridencoder',
    'otype': 'HashGrid',
    'n_levels': 16,
    'base_resolution': 16,
    'desired_resolution': 2048,
    'log2_hashmap_size': 15,
    'n_features_per_level': 2,
    'per_level_scale': 1.25}
config_uv = {
    'module': 'gridencoder',
    'otype': 'HashGrid',
    'n_levels': 16,
    'base_resolution': 16,
    'desired_resolution': 2048,
    'log2_hashmap_size': 15,
    'n_features_per_level': 2,
    'per_level_scale': 1.25}
model_mapping = [{
    'model_type': 'EncodingMappingNetwork',
    'pretrain': True,
    'texture': {
        'model_type': 'EncodingTextureNetwork',
        'encoding_config': config_uv},
    'residual': {
        'model_type': 'ResidualEstimator',
        'encoding_config': config_xyt},
    'encoding_config': None,
    'num_layers': 4,
    'num_neurons': 256
}, {
    'model_type': 'EncodingMappingNetwork',
    'pretrain': True,
    'texture': {
        'model_type': 'EncodingTextureNetwork',
        'encoding_config': config_uv},
    'residual': {
        'model_type': 'ResidualEstimator',
        'encoding_config': config_xyt},
    'encoding_config': None,
    'num_layers': 4,
    'num_neurons': 256
}]
alpha = {
    'model_type': 'EncodingAlphaNetwork',
    'encoding_config': config_xyt}
