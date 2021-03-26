import os
import glob
from tqdm import tqdm
import numpy as np
from absl import app
from absl import flags
from cumulo.utils.evaluation import evaluate_clouds, evaluate_file

flags.DEFINE_string('path', None, help='Location of predictions')
flags.DEFINE_string('output_path', None, help='Save location, defaults to above path')
FLAGS = flags.FLAGS


def main(_):
    total_report = ''
    mask_names = [0, 1]  # cloud mask targets ('no cloud', 'cloud')
    label_names = list(range(8))  # cloud class targets (8 different cloud types)
    files = glob.glob(os.path.join(FLAGS.path, "*.npz"))
    total_labels = np.array([])
    total_probabilities = None

    if FLAGS.output_path is None:
        output_path = FLAGS.path
    else:
        if not os.path.exists(FLAGS.output_path):
            os.makedirs(FLAGS.output_path)
        output_path = FLAGS.output_path

    for file in tqdm(files):
        data = np.load(file)
        report, probabilities, labels = \
            evaluate_file(file, data['outputs'], data['labels'], data['cloud_mask'],
                          label_names, mask_names)
        # --- Save intermediate report and merge probabilities and labels for total evaluation ---
        total_report += report
        with open(os.path.join(output_path, 'report.txt'), 'w') as f:
            f.write(total_report)
        total_labels = np.append(total_labels, labels)
        if total_probabilities is None:
            total_probabilities = probabilities
        else:
            total_probabilities = np.hstack((total_probabilities, probabilities))

    # --- Generate total evaluation and save final report ---
    total_report += '#### TOTAL ####\n\n'
    total_file = os.path.join(output_path, 'total.npz')
    report, matrix = evaluate_clouds(total_probabilities, total_labels, label_names, total_file, detailed=True)
    total_report += 'Cloud class eval:\n\n' + report + '\n\n'
    total_report += matrix
    with open(os.path.join(output_path, 'report.txt'), 'w') as f:
        f.write(total_report)


if __name__ == '__main__':
    app.run(main)
