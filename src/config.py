from utils import colour_str, CHARTS, SCENE_VARIABLES, SIC_LOOKUP, SOD_LOOKUP, FLOE_LOOKUP
from functions import f1_metric, r2_metric
SCENE_VARIABLES = [
    # -- Sentinel-1 variables -- #
    'nersc_sar_primary',
    'nersc_sar_secondary',
    'sar_incidenceangle',

    # -- Geographical variables -- #
    'distance_map',

    # -- AMSR2 channels -- #
    # 'btemp_6_9h', 'btemp_6_9v',
    # 'btemp_7_3h', 'btemp_7_3v',
    # 'btemp_10_7h', 'btemp_10_7v',
    'btemp_18_7h', 'btemp_18_7v',
    # 'btemp_23_8h', 'btemp_23_8v',
    'btemp_36_5h', 'btemp_36_5v',
    # 'btemp_89_0h', 'btemp_89_0v',

    # -- Environmental variables -- #
    'u10m_rotated', 'v10m_rotated',
    't2m',
    # 'skt',
    'tcwv', 'tclw',

    # -- Auxilary Variables -- #
    'aux_time',
    'aux_lat',
    'aux_long'
]

train_options = {'train_variables': SCENE_VARIABLES,
                #   'train_variables': ['nersc_sar_primary', 'nersc_sar_secondary'],
                 'seed': 42,
                 'compile_model': False,  # Whether to compile the model with torch.compile. Only for PyTorch 2.0 and above.
                 'path_to_train_data': '/full_path_to/data/train',
                 'path_to_test_data': '/full_path_to/data/test',
                 'test_gt_json': 'datalists/test_gt.json',
                 'test_x_json': 'datalists/test_x.json',
                 'compute_classwise_f1score': True,
                 'plot_confusion_matrix': True,
                #  'path_to_env' : "/mnt/beegfs/spillal2/STAC/unet/",
                'path_to_env' : "/full_path_to/src/",
                 'optimizer': {
                     'type': 'SGD',
                     'lr': 0.001,  # Optimizer learning rate.
                     'momentum': 0.9,
                     'dampening': 0,
                     'nesterov': False,
                     'weight_decay': 0.01
                 },
                # 'optimizer': {
                #         'type': 'AdamW',
                #         'lr': 0.004,  # Optimizer learning rate.
                #         'weight_decay': 0.05,
                #         'b1': 0.9,
                #         'b2': 0.999

                #         },

                 'scheduler': {
                     'type': 'CosineAnnealingWarmRestartsLR',  # Name of the schedulers
                     'EpochsPerRestart': 20,  # Number of epochs for the first restart
                     # This number will be used to increase or descrase the number of epochs to restart after each restart.
                     'RestartMult': 1,
                     'lr_min': 0,  # Minimun learning rate
                 },

                 'batch_size': 16,
                 'num_workers': 0,  # Number of parallel processes to fetch data.
                 'num_workers_val': 0,  # Number of parallel processes during validation.
                 'patch_size': 256,
                 'down_sample_scale': 10,
                 'loader_downsampling': 'nearest',
                 'loader_upsampling': 'nearest',

                 'data_augmentations': {
                     'Random_h_flip': 0.5,
                     'Random_v_flip': 0.5,
                     'Random_rotation_prob': 0.5,
                     'Random_rotation': 90,
                     'Random_scale_prob': 0.5,
                     'Random_scale': (0.9, 1.1),
                     'Cutmix_beta': 1.0,
                     'Cutmix_prob': 0.5,
                 },
                 'amsrenv_pixel_spacing' : 2000,
                 'charts' : ['SIC', 'SOD', 'FLOE'],  # Which charts are going to be used for training.
                 # -- Model selection -- #
                 'model_selection': 'moe_full',
                 'unet_conv_filters': [32, 64, 64, 64],
                 'epochs': 300,  # Number of epochs before training stop.
                 'epoch_len': 500,  # Number of batches for each epoch.
                 "binary_water_classifier": False,
                 # Size of patches sampled. Used for both Width and Height.
                 'patch_size': 256,
                 'pixel_spacing': 80,  # Pixel spacing for the input data. Used for both Width and Height.
                 # Which train set is going to be used
                 'cross_val_run':0,
                 'train_list_path': 'datalists/dataset.json',
                 'charts': CHARTS,  # Charts to train on.
                 'n_classes': {  # number of total classes in the reference charts, including the mask.
                        'SIC': SIC_LOOKUP['n_classes'],
                        'SOD': SOD_LOOKUP['n_classes'],
                        'FLOE': FLOE_LOOKUP['n_classes']
                    },
                    # SAR pixel spacing. 80 for the ready-to-train AI4Arctic Challenge dataset.
                    'pixel_spacing': 80,
                    'train_fill_value': 0,  # Mask value for SAR training data.
                    'class_fill_values': {  # Mask value for class/reference data.
                        'SIC': SIC_LOOKUP['mask'],
                        'SOD': SOD_LOOKUP['mask'],
                        'FLOE': FLOE_LOOKUP['mask'],
                    },
                
                 # Which validation set is going to be used
                 'val_path': 'datalists/valset2.json',
                 'task_weights': [1, 3, 3],
                 'chart_loss': {  # Loss for the task
                     'SIC': {
                         'type': 'MSELossWithIgnoreIndex',
                         'ignore_index': 255,
                     },
                     'SOD': {
                         'type': 'FLOELoss',
                         'ignore_index': 255,
                     },
                     'FLOE': {
                         'type': 'FLOELoss',
                         'ignore_index': 255,
                     },
                 },
                 'accumulation_steps':8,
                 'gpu_id':0,
                 'unet_conv_filters': [32, 64, 64, 64],
                 'conv_kernel_size': (3, 3),  # Size of convolutional kernels.
                'conv_stride_rate': (1, 1),  # Stride rate of convolutional kernels.
                'conv_dilation_rate': (1, 1),  # Dilation rate of convolutional kernels.
                'conv_padding': (1, 1),  # Number of padded pixels in convolutional layers.
                'conv_padding_style': 'zeros',  # Style of padding.
                 'latitude': {
                    'mean': 69.12526250065734,
                    'std': 7.03179625261593
                },

                'longitude': {
                    'mean': -56.38966259295485,
                    'std': 31.32935694114249
                },
                'edge_consistency_loss':0,
                'chart_metric': {  # Metric functions for each ice parameter and the associated weight.
        'SIC': {
            'func': r2_metric,
            'weight': 1,
        },
        'SOD': {
            'func': f1_metric,
            'weight': 3,
        },
        'FLOE': {
            'func': f1_metric,
            'weight': 3,
        },
    },
                 }