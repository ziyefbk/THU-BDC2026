# Enhanced config parameters
sequence_length = 60
feature_num = 'enhanced_158+39'
config = {
    'sequence_length': sequence_length,
    'd_model': 384,          # increased from 256
    'nhead': 8,             # increased from 4
    'num_layers': 4,        # increased from 3
    'dim_feedforward': 768,  # increased from 512
    'batch_size': 8,        # increased from 4
    'num_epochs': 50,
    'learning_rate': 3e-5,   # increased from 1e-5
    'dropout': 0.1,
    'feature_num': feature_num,
    'max_grad_norm': 5.0,

    # ranking loss weights
    'pairwise_weight': 1,
    'base_weight': 1.0,
    'top5_weight': 2.5,      # increased from 2.0

    # early stopping
    'patience': 15,

    # cross-stock attention heads
    'cross_nhead': 4,

    # label normalization per date
    'label_normalize': True,

    # temperature for softmax scoring
    'score_temperature': 1.0,

    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data',
}
