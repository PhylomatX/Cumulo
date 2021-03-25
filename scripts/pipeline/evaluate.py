import os
import glob
from tqdm import tqdm
import numpy as np
from absl import app
from absl import flags
from cumulo.utils.evaluation import evaluate_clouds, evaluate_file

flags.DEFINE_string('path', None, help='Location of predictions')
flags.DEFINE_string('o_path', None, help='Save location, defaults to above path')
flags.DEFINE_bool('all', True, help='Include file-wise evaluation in report.')
flags.DEFINE_bool('mask', False, help='Include file-wise cloud mask eval.')
flags.DEFINE_bool('mask_total', False, help='Include total cloud mask eval.')
FLAGS = flags.FLAGS


def main(_):
    total_report = ''
    mask_names = [0, 1]  # cloud mask targets ('no cloud', 'cloud')
    label_names = list(range(8))  # cloud class targets (8 different cloud types)
    files = glob.glob(os.path.join(FLAGS.path, "*.npz"))
    total_labels = np.array([])
    total_probabilities = None

    if FLAGS.o_path is None:
        o_path = FLAGS.path
    else:
        if not os.path.exists(FLAGS.o_path):
            os.makedirs(FLAGS.o_path)
        o_path = FLAGS.o_path

    for file in tqdm(files):
        data = np.load(file)
        total_report, total_labels, total_probabilities = \
            evaluate_file(file, data['outputs'], data['labels'], data['cloud_mask'],
                          label_names, mask_names, total_report, total_probabilities, total_labels)
        with open(os.path.join(o_path, 'report.txt'), 'w') as f:
            f.write(total_report)

    # --- Generate total evaluation and save final report ---
    total_report += '#### TOTAL ####\n\n'
    total_file = os.path.join(FLAGS.o_path, 'total.npz')
    report, matrix = evaluate_clouds(total_probabilities, total_labels, label_names, total_file)
    total_report += 'Cloud class eval:\n\n' + report + '\n\n'
    total_report += matrix
    with open(os.path.join(o_path, 'report.txt'), 'w') as f:
        f.write(total_report)


if __name__ == '__main__':
    app.run(main)
