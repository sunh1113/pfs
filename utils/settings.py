DATASETTINGS = {
    'c10': {
        'num_classes': 10,
        'img_size': 32,
        'num_data': 50000,
        'crop_pad': 4,
        'flip': True,

        'epochs': 70,
        'decay_steps': [35, 55],
    },
    'c100': {
        'num_classes': 100,
        'img_size': 32,
        'num_data': 50000,
        'crop_pad': 4,
        'flip': True,

        'epochs': 70,
        'decay_steps': [30, 50],
    },
    'i200': {
        'num_classes': 200,
        'img_size': 64,
        'num_data': 100000,
        'crop_pad': 8,
        'flip': True,

        'epochs': 100,
        'decay_steps': [30, 60, 80],
    },
}
