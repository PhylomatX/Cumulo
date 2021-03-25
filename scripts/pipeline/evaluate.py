import os
import glob
from tqdm import tqdm
import numpy as np
from absl import app
from absl import flags
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from cumulo.utils.pipeline import include_cloud_mask

flags.DEFINE_string('path', None, help='Location of predictions')
flags.DEFINE_string('o_path', None, help='Save location, defaults to above path')
flags.DEFINE_bool('all', True, help='Include file-wise evaluation in report.')
flags.DEFINE_bool('mask', False, help='Include file-wise cloud mask eval.')
flags.DEFINE_bool('mask_total', False, help='Include total cloud mask eval.')
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
    mask_targets = [0, 1]  # cloud mask targets ('no cloud', 'cloud')
    class_targets = list(range(9))  # cloud class targets ('no cloud' + 8 different cloud types)
    files = glob.glob(os.path.join(FLAGS.path, "*.npz"))
    total_class_labels = np.array([])
    total_class_predictions = np.array([])
    total_mask_labels = np.array([])
    total_mask_predictions = np.array([])
    total_mask_predictions_raw = np.array([])

    if FLAGS.o_path is None:
        o_path = FLAGS.path
    else:
        if not os.path.exists(FLAGS.o_path):
            os.makedirs(FLAGS.o_path)
        o_path = FLAGS.o_path

    for file in tqdm(files):
        data = np.load(file)
        labels = data['labels']
        cloud_mask = data['cloud_mask'].reshape(-1)
        prediction = data['prediction']
        mask_prediction_raw = None

        if prediction.ndim == 3:
            mask_prediction_raw = prediction[0].copy().reshape(-1)
            prediction[0][prediction[0] < 0.5] = 0
            prediction[0][prediction[0] >= 0.5] = 1
            flat = np.argmax(prediction[1:, ...], 0)
            prediction = include_cloud_mask(flat, prediction[0]).astype(np.int)

        if FLAGS.all:
            report += f"#### {file.replace(FLAGS.path, '')} ####\n\n"

        if FLAGS.mask:
            # --- cloud mask evaluation ---
            mask_prediction = prediction.copy()
            mask_prediction[prediction != 0] = 1
            mask_prediction = mask_prediction.reshape(-1)
            if FLAGS.mask_total:
                total_mask_labels = np.append(total_mask_labels, cloud_mask)
                total_mask_predictions = np.append(total_mask_predictions, mask_prediction)
                if mask_prediction_raw is not None:
                    total_mask_predictions_raw = np.append(total_mask_predictions_raw, mask_prediction_raw)
            if FLAGS.all:
                # --- Cloud mask classification report ---
                mask_report = sm.classification_report(cloud_mask, mask_prediction, labels=mask_targets)
                report += 'Cloud mask eval:\n\n' + mask_report + '\n\n'

                # --- Cloud mask ROC curve ---
                fpr, tpr, _ = sm.roc_curve(cloud_mask, mask_prediction_raw)
                mask_auc = sm.auc(fpr, tpr)
                mask_roc_disp = sm.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=mask_auc)
                mask_roc_disp.plot()
                plt.savefig(os.path.join(o_path, f"{file.replace(FLAGS.path, '')}_mask_roc.png"))
                plt.close()

                # --- Cloud mask PR curve ---
                prec, recall, _ = sm.precision_recall_curve(cloud_mask, mask_prediction_raw)
                mask_pr_disp = sm.PrecisionRecallDisplay(precision=prec, recall=recall)
                mask_pr_disp.plot()
                plt.savefig(os.path.join(o_path, f"{file.replace(FLAGS.path, '')}_mask_pr.png"))
                plt.close()

                # --- Cloud mask histograms (bin number determined with Freedman-Diaconis) ---
                cloudy = mask_prediction_raw[cloud_mask == 1]
                not_cloudy = mask_prediction_raw[cloud_mask == 0]

                n = len(cloudy)
                sorted_cloudy = np.sort(cloudy)
                q1 = np.median(sorted_cloudy[:int(n/2)])
                q3 = np.median(sorted_cloudy[-int(n/2):])
                bin_width = 2 * (q3 - q1) / np.cbrt(n)
                bin_num = int(1 / bin_width)
                cloudy_n, _, _ = plt.hist(cloudy, density=True, bins=bin_num, histtype='step')

                n = len(not_cloudy)
                sorted_not_cloudy = np.sort(not_cloudy)
                q1 = np.median(sorted_not_cloudy[:int(n/2)])
                q3 = np.median(sorted_not_cloudy[-int(n/2):])
                bin_width = 2 * (q3 - q1) / np.cbrt(n)
                bin_num = int(1 / bin_width)
                not_cloudy_n, _, _ = plt.hist(not_cloudy, density=True, bins=bin_num, histtype='step')

                plt.xlabel('Network output')
                plt.ylabel('Normalized counts')
                plt.tight_layout()
                plt.savefig(os.path.join(o_path, f"{file.replace(FLAGS.path, '')}_mask_histo.png"))
                plt.close()

        # --- cloud class evaluation ---
        labels = include_cloud_mask(labels.reshape(-1), cloud_mask)
        cloudy_label_mask = np.logical_and(labels != -1, labels != 0)
        cloudy_labels = labels[cloudy_label_mask]
        cloudy_predictions = prediction.reshape(-1)[cloudy_label_mask]
        total_class_labels = np.append(total_class_labels, cloudy_labels)
        total_class_predictions = np.append(total_class_predictions, cloudy_predictions)
        if FLAGS.all:
            class_report = sm.classification_report(cloudy_labels, cloudy_predictions, labels=class_targets, zero_division=0)
            report += 'Cloud class eval:\n\n' + class_report + '\n\n'
            class_matrix = sm.confusion_matrix(cloudy_labels, cloudy_predictions)
            report += write_confusion_matrix(class_matrix, get_target_names(cloudy_labels, cloudy_predictions, class_targets)) + '\n\n\n\n'

        # --- Save intermediate report ---
        with open(os.path.join(o_path, 'report.txt'), 'w') as f:
            f.write(report)

    report += '#### TOTAL ####\n\n'

    if FLAGS.mask and FLAGS.mask_total:
        # --- Total cloud mask evaluation ---
        mask_report = sm.classification_report(total_mask_labels, total_mask_predictions, labels=mask_targets)
        report += 'Cloud mask eval:\n\n' + mask_report + '\n\n'

    # --- Total cloud class evaluation ---
    class_report = sm.classification_report(total_class_labels, total_class_predictions, labels=class_targets, zero_division=0)
    report += 'Cloud class eval:\n\n' + class_report + '\n\n'
    class_matrix_normalized = sm.confusion_matrix(total_class_labels, total_class_predictions, normalize='true')
    class_matrix_disp = sm.ConfusionMatrixDisplay(class_matrix_normalized, display_labels=class_targets)
    class_matrix_disp.plot(include_values=False, cmap='Reds')
    plt.savefig(os.path.join(o_path, 'class_matrix_normalized.png'))

    class_matrix = sm.confusion_matrix(total_class_labels, total_class_predictions)
    class_matrix_disp = sm.ConfusionMatrixDisplay(class_matrix, display_labels=class_targets)
    class_matrix_disp.plot(cmap='Reds')
    plt.savefig(os.path.join(o_path, 'class_matrix.png'))

    report += write_confusion_matrix(class_matrix, get_target_names(total_class_labels, total_class_predictions, class_targets))

    with open(os.path.join(o_path, 'report.txt'), 'w') as f:
        f.write(report)


if __name__ == '__main__':
    app.run(main)
