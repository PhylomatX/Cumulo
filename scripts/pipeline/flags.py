from absl import flags

# --- predict ---
flags.DEFINE_string('m_name', 'val_best_weighted', help='Model name')
flags.DEFINE_string('o_path', None, help='Location for output')
flags.DEFINE_bool('raw_predictions', True, help='Save network outputs for later visualization')
flags.DEFINE_integer('pred_num', None, help='Number of prediction files')

# --- train ---
flags.DEFINE_string('d_path', '/storage/group/dataset_mirrors/01_incoming/satellite/Cumulo/unprocessed/nc/clean/', help='Data path')
flags.DEFINE_string('m_path', None, help='Model path')
flags.DEFINE_string('filetype', 'nc', help='File type for dataset')
flags.DEFINE_integer('r_seed', 1, help='Random seed')
flags.DEFINE_integer('nb_epochs', 200, help='Number of epochs')
flags.DEFINE_integer('num_workers', 4, help='Number of workers for the dataloader.')
flags.DEFINE_integer('bs', 1, help='Batch size for training and validation.')
flags.DEFINE_integer('dataset_bs', 16, help='Batch size for training and validation.')
flags.DEFINE_integer('tile_num', None, help='Tile number / data set size.')
flags.DEFINE_integer('tile_size', 256, help='Tile size.')
flags.DEFINE_integer('nb_classes', 9, help='Number of classes.')
flags.DEFINE_integer('center_distance', None, help='Distance between base points of tile extraction.')
flags.DEFINE_bool('val', True, help='Flag for validation after each epoch.')
flags.DEFINE_string('model', 'weak', help='Option for choosing between UNets.')
flags.DEFINE_string('norm', 'none', help='Type of normalization, one of [bn, gn, none]')
flags.DEFINE_bool('merged', True, help='Flag for indicating use of merged dataset')
flags.DEFINE_bool('examples', True, help='Save some training examples in each epoch')
flags.DEFINE_bool('local_norm', False, help='Standardize each image channel-wise. If False the statistics of a data subset will be used.')
flags.DEFINE_integer('analysis_freq', 1, help='Validation and example save frequency')
flags.DEFINE_integer('rot', 2, help='Number of elements in rotation group')
flags.DEFINE_integer('offset', 46, help='Cropping offset for labels in case of valid convolutions')
flags.DEFINE_integer('valid_convolution_offset', 46, help='Cropping offset for labels in case of valid convolutions')
flags.DEFINE_integer('padding', 0, help='Padding for convolutions')
flags.DEFINE_integer('class_weight', 2, help='Weight of class loss')
flags.DEFINE_integer('mask_weight', 1, help='Weight of mask loss')
flags.DEFINE_integer('auto_weight', 1, help='Weight of auto loss')
flags.DEFINE_float('augment_prob', 0.5, help='Augmentation probability')
flags.DEFINE_float('rotation_probability', 0.5, help='Augmentation probability')

FLAGS = flags.FLAGS
