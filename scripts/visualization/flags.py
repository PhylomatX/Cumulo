from absl import flags

flags.DEFINE_string('path', None, help='Directory where npz files are located.')
flags.DEFINE_string('type', 'train', help='Type of example (val or train)')
flags.DEFINE_list('epoch_interval', [0, 100], help='Only training examples from epochs within this interval are selected')
flags.DEFINE_boolean('use_continuous_colors', True, help='Use predictions for color weighting')
flags.DEFINE_boolean('cloud_mask_as_binary', True, help='Make cloud mask binary')
flags.DEFINE_boolean('to_file', True, help='Save examples as images')
flags.DEFINE_integer('label_dilation', 10, help='Size of label dilation')

FLAGS = flags.FLAGS
