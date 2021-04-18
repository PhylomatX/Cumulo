from absl import flags

# --- predict ---
flags.DEFINE_string('model_name', 'val_best', help='Model name')
flags.DEFINE_bool('raw_predictions', False, help='')
flags.DEFINE_string('output_path', None, help='Path to folder where output should get saved')
flags.DEFINE_integer('prediction_number', None, help='Number of files for which predictions should get generated')
flags.DEFINE_bool('immediate_evaluation', True, help='Do not save predictions, but evaluate them immediately and save images and reports to file')
flags.DEFINE_boolean('use_continuous_colors', True, help='Use predictions for color weighting')
flags.DEFINE_boolean('cloud_mask_as_binary', False, help='Make cloud mask binary')
flags.DEFINE_boolean('to_file', True, help='Save examples as images')
flags.DEFINE_integer('label_dilation', 10, help='Size of label dilation')

# --- train ---
flags.DEFINE_string('d_path', '/storage/group/dataset_mirrors/01_incoming/satellite/Cumulo/unprocessed/nc/clean/', help='Data path')
flags.DEFINE_string('m_path', None, help='Model path')
flags.DEFINE_string('filetype', 'nc', help='File type for dataset')
flags.DEFINE_integer('r_seed', 1, help='Random seed')
flags.DEFINE_integer('epoch_number', 200, help='Number of epochs')
flags.DEFINE_integer('nb_epochs', 200, help='Number of epochs')
flags.DEFINE_integer('num_workers', 4, help='Number of workers for the dataloader.')
flags.DEFINE_integer('bs', 1, help='Batch size for training and validation.')
flags.DEFINE_integer('dataset_bs', 16, help='Batch size for training and validation.')
flags.DEFINE_integer('tile_num', None, help='Tile number / data set size.')
flags.DEFINE_integer('tile_size', 256, help='Tile size.')
flags.DEFINE_integer('nb_classes', 9, help='Number of classes.')
flags.DEFINE_integer('center_distance', None, help='Distance between base points of tile extraction.')
flags.DEFINE_bool('val', True, help='Flag for validation after each epoch.')
flags.DEFINE_string('model', 'unet', help='Option for choosing between UNets.')
flags.DEFINE_string('norm', 'none', help='Type of normalization, one of [bn, gn, none]')
flags.DEFINE_bool('merged', True, help='Flag for indicating use of merged dataset')
flags.DEFINE_bool('filter_cloudy_labels', False, help='Flag for only extracting tiles at cloudy labels')
flags.DEFINE_bool('examples', True, help='Save some training examples in each epoch')
flags.DEFINE_bool('demo', False, help='Use demo mode (only affects dataset splits.')
flags.DEFINE_bool('local_norm', False, help='Standardize each image channel-wise. If False the statistics of a data subset will be used.')
flags.DEFINE_bool('most_frequent_clouds_as_GT', True, help='Reduce the cloud type GT to the most frequent cloud type of all pixels. If False, the lowest cloud type is taken.')
flags.DEFINE_integer('analysis_freq', 1, help='Validation and example save frequency')
flags.DEFINE_integer('rot', 4, help='Number of elements in rotation group')
flags.DEFINE_integer('valid_convolution_offset', 46, help='Cropping offset for labels in case of valid convolutions')
flags.DEFINE_integer('padding', 0, help='Padding for convolutions')
flags.DEFINE_integer('class_weight', 2, help='Weight of class loss')
flags.DEFINE_integer('mask_weight', 1, help='Weight of mask loss')
flags.DEFINE_integer('auto_weight', 0, help='Weight of auto loss')
flags.DEFINE_float('rotation_probability', 0.5, help='Augmentation probability')
flags.DEFINE_string('nc_exclude_path', None, help='Augmentation probability')

FLAGS = flags.FLAGS
