import os
import glob
from tqdm import tqdm
import numpy as np
from absl import app
from absl import flags
import sklearn.metrics as sm
from cumulo.utils.utils import include_cloud_mask

flags.DEFINE_string('path', None, help='Location of predictions')
flags.DEFINE_string('o_path', None, help='Save location, defaults to above path')
flags.DEFINE_bool('full', True, help='Include file-wise evaluation in report.')
FLAGS = flags.FLAGS


def write_confusion_matrix(cm: np.array, names: list) -> str:
    txt = f"{'':<15}"
    for name in names:
        txt += f"{name:<15}"
    txt += '\n'
    for ix, name in enumerate(names):
        txt += f"{name:<15}"
        for num in cm[ix]:
            txt += f"{num:<15}"
        txt += '\n'
    return txt


def get_target_names(gtl: np.ndarray, hcl: np.ndarray, targets: list) -> list:
    """ Extracts the names of the labels which appear in gtl and hcl. """
    targets = np.array(targets)
    total = np.unique(np.concatenate((gtl, hcl), axis=0)).astype(int)
    return list(targets[total])


def main(_):
    report = ''
    cm_targets = [0, 1]
    ct_targets = list(range(9))
    files = glob.glob(os.path.join(FLAGS.path, "*.npz"))
    total_labels = np.array([])
    total_c_labels = np.array([])
    total_predictions = np.array([])
    total_c_predictions = np.array([])
    for file in tqdm(files):
        data = np.load(file)
        labels = data['labels']  # Raw labels without cloud mask
        cloud_mask = data['cloud_mask']  # Raw cloud mask
        prediction = data['prediction']  # predictions where cloud mask and labels have been merged

        # --- exclude pixels at the borders which were too small for another tile ---
        valid = prediction != -1
        cloud_mask = cloud_mask[valid]
        labels = labels[valid]
        prediction = prediction[valid]

        # --- cloud mask evaluation ---
        c_prediction = prediction.copy()
        c_prediction[prediction != 0] = 1
        total_c_labels = np.append(total_c_labels, cloud_mask)
        total_c_predictions = np.append(total_c_predictions, c_prediction)

        # --- cloud type evaluation ---
        labels = include_cloud_mask(labels, cloud_mask)
        mask = np.logical_and(labels != -1, labels != 0)
        labels = labels[mask]
        prediction = prediction[mask]
        total_labels = np.append(total_labels, labels)
        total_predictions = np.append(total_predictions, prediction)
        if FLAGS.full:
            report += f"#### {file.replace(FLAGS.path, '')} ####\n\n"
            cm_cr_txt = sm.classification_report(cloud_mask, c_prediction, labels=cm_targets)
            report += 'Cloud mask eval:\n\n' + cm_cr_txt + '\n\n'
            cf_matrix = sm.confusion_matrix(cloud_mask, c_prediction)
            report += write_confusion_matrix(cf_matrix, get_target_names(cloud_mask, c_prediction, cm_targets)) + '\n\n'
            ct_cr_txt = sm.classification_report(labels, prediction, labels=ct_targets)
            report += 'Cloud type eval:\n\n' + ct_cr_txt + '\n\n'
            cf_matrix = sm.confusion_matrix(labels, prediction)
            report += write_confusion_matrix(cf_matrix, get_target_names(labels, prediction, ct_targets)) + '\n\n\n\n'
    report += '#### TOTAL ####\n\n'
    cm_cr_txt = sm.classification_report(total_c_labels, total_c_predictions, labels=cm_targets)
    report += 'Cloud mask eval:\n\n' + cm_cr_txt + '\n\n'
    cf_matrix = sm.confusion_matrix(total_c_labels, total_c_predictions)
    report += write_confusion_matrix(cf_matrix, get_target_names(total_c_labels, total_c_predictions, cm_targets)) + '\n\n'
    ct_cr_txt = sm.classification_report(total_labels, total_predictions, labels=ct_targets)
    report += 'Cloud type eval:\n\n' + ct_cr_txt + '\n\n'
    cf_matrix = sm.confusion_matrix(total_labels, total_predictions)
    report += write_confusion_matrix(cf_matrix, get_target_names(total_labels, total_predictions, ct_targets))

    if FLAGS.o_path is None:
        o_path = FLAGS.path
    else:
        if not os.path.exists(FLAGS.o_path):
            os.makedirs(FLAGS.o_path)
        o_path = FLAGS.o_path

    with open(os.path.join(o_path, 'report.txt'), 'w') as f:
        f.write(report)


if __name__ == '__main__':
    app.run(main)
