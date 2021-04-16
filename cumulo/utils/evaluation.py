import numpy as np
import os
import matplotlib
import pickle as pkl
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn import metrics as sm
from cumulo.utils.basics import probabilities_from_outputs


def divide_into_tiles(tile_size, offset, radiances):
    img_width = radiances.shape[1]
    img_height = radiances.shape[2]

    output_size = tile_size - 2 * offset
    nb_outputs_row = (img_width - 2 * offset) // output_size
    nb_outputs_col = (img_height - 2 * offset) // output_size

    tiles = []
    locations = []

    # --- gather tiles from within swath ---
    for i in range(nb_outputs_row):
        for j in range(nb_outputs_col):
            tiles.append(radiances[:, i * output_size: 2 * offset + (i + 1) * output_size, j * output_size: 2 * offset + (j + 1) * output_size])
            locations.append(((offset + i * output_size, offset + (i + 1) * output_size),
                              (offset + j * output_size, offset + (j + 1) * output_size)))

    # --- gather tiles from bottom row ---
    for i in range(nb_outputs_row):
        tiles.append(radiances[:, i * output_size: 2 * offset + (i + 1) * output_size, img_height - tile_size:img_height])
        locations.append(((offset + i * output_size, offset + (i + 1) * output_size),
                          (offset + img_height - tile_size, img_height - offset)))

    # --- gather tiles from most right column ---
    for j in range(nb_outputs_col):
        tiles.append(radiances[:, img_width - tile_size:img_width, j * output_size: 2 * offset + (j + 1) * output_size])
        locations.append(((offset + img_width - tile_size, img_width - offset),
                          (offset + j * output_size, offset + (j + 1) * output_size)))

    # --- gather tile from lower right corner ---
    tiles.append(radiances[:, img_width - tile_size:img_width, img_height - tile_size:img_height])
    locations.append(((offset + img_width - tile_size, img_width - offset),
                      (offset + img_height - tile_size, img_height - offset)))

    tiles = np.stack(tiles)
    locations = np.stack(locations)

    return tiles, locations


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


def create_histogram(class_predictions, file, labels):
    for ix, class_prediction in enumerate(class_predictions):
        # --- determine bin number with Freedman-Diaconis rule ---
        n = len(class_prediction)
        sorted_cloudy = np.sort(class_prediction)
        q1 = np.median(sorted_cloudy[:int(n / 2)])
        q3 = np.median(sorted_cloudy[-int(n / 2):])
        bin_width = 2 * (q3 - q1) / np.cbrt(n)
        bin_num = int(1 / bin_width)
        cloudy_n, _, _ = plt.hist(class_prediction, density=True, bins=bin_num, histtype='step', label=labels[ix])

    plt.xlabel('Network output after softmax')
    plt.ylabel('Normalized counts')
    plt.legend()
    plt.tight_layout()
    plt.savefig(file)
    plt.close()


def create_class_histograms(outputs, labels, path):
    outputs = outputs.copy()
    outputs = outputs.transpose()
    for label in range(8):
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        axs = axs.reshape(-1)
        for i in range(8):
            _ = axs[i].hist(outputs[labels == label][:, i], bins=100, alpha=0.8, label=i, range=(-20, 20))
            axs[i].set_title(i)
        plt.savefig(os.path.join(path, f'{label}.png'))


def evaluate_cloud_mask(mask_predictions, mask, mask_names, npz_file, detailed=False):
    mask_predictions = mask_predictions.reshape(-1)
    hard_mask_predictions = mask_predictions.copy()
    hard_mask_predictions[mask_predictions < 0.5] = 0
    hard_mask_predictions[mask_predictions >= 0.5] = 1
    report = sm.classification_report(mask, hard_mask_predictions, labels=mask_names)

    if detailed:
        fpr, tpr, _ = sm.roc_curve(mask, mask_predictions)
        auc = sm.auc(fpr, tpr)
        mask_roc_disp = sm.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
        mask_roc_disp.plot()
        plt.savefig(npz_file.replace('.npz', '_mask_roc.png'))
        plt.close()

        precision, recall, _ = sm.precision_recall_curve(mask, mask_predictions)
        mask_pr_disp = sm.PrecisionRecallDisplay(precision=precision, recall=recall)
        mask_pr_disp.plot()
        plt.savefig(npz_file.replace('.npz', '_mask_pr.png'))
        plt.close()

        cloudy = mask_predictions[mask == 1]
        not_cloudy = mask_predictions[mask == 0]
        create_histogram([cloudy, not_cloudy], npz_file.replace('.npz', f'_mask_hist.png'), ['cloud', 'no cloud'])
    return report


def evaluate_clouds(cloudy_probabilities, cloudy_labels, label_names, npz_file, detailed=False):
    hard_cloudy_predictions = np.argmax(cloudy_probabilities, 0).reshape(-1)

    report = sm.classification_report(cloudy_labels, hard_cloudy_predictions, labels=label_names, zero_division=0)
    matrix = sm.confusion_matrix(cloudy_labels, hard_cloudy_predictions, labels=label_names)
    matrix_string = write_confusion_matrix(matrix, get_target_names(cloudy_labels, hard_cloudy_predictions, label_names))

    if detailed:
        report_dict = sm.classification_report(cloudy_labels, hard_cloudy_predictions, labels=label_names, zero_division=0, output_dict=True)
        with open(npz_file.replace('.npz', '_report.pkl'), 'wb') as f:
            pkl.dump(report_dict, f)
        with open(npz_file.replace('.npz', '_matrix.pkl'), 'wb') as f:
            pkl.dump(matrix, f)

        class_matrix_disp = sm.ConfusionMatrixDisplay(matrix, display_labels=label_names)
        class_matrix_disp.plot(cmap='Reds')
        plt.savefig(npz_file.replace('.npz', '_matrix.png'))
        plt.close()

        class_matrix_normalized = sm.confusion_matrix(cloudy_labels, hard_cloudy_predictions, normalize='true', labels=label_names)
        class_matrix_disp = sm.ConfusionMatrixDisplay(class_matrix_normalized, display_labels=label_names)
        class_matrix_disp.plot(include_values=False, cmap='Reds')
        plt.savefig(npz_file.replace('.npz', '_matrix_normalized.png'))
        plt.close()

        for ix in range(len(label_names)):
            if np.all(cloudy_labels != ix):
                continue
            ix_labels = np.zeros_like(cloudy_labels)
            ix_labels[cloudy_labels == ix] = 1
            ix_predictions = cloudy_probabilities[ix].reshape(-1)

            fpr, tpr, _ = sm.roc_curve(ix_labels, ix_predictions)
            auc = sm.auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{label_names[ix]} - AUC: {round(auc, 2)}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.legend()
        plt.savefig(npz_file.replace('.npz', f'_roc.pdf'))
        plt.close()

        for ix in range(len(label_names)):
            if np.all(cloudy_labels != ix):
                continue
            ix_labels = np.zeros_like(cloudy_labels)
            ix_labels[cloudy_labels == ix] = 1
            ix_predictions = cloudy_probabilities[ix].reshape(-1)
            precision, recall, _ = sm.precision_recall_curve(ix_labels, ix_predictions)
            plt.plot(recall, precision, label=label_names[ix])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.legend()
        plt.savefig(npz_file.replace('.npz', f'_pr.pdf'))
        plt.close()
    return report, matrix_string


# noinspection PyUnboundLocalVariable
def evaluate_file(file, outputs, labels, cloud_mask, label_names, mask_names, no_cloud_mask):
    labels = labels.reshape(-1)
    cloud_mask = cloud_mask.reshape(-1)
    cloudy_labels_mask = labels != -1  # use all existing labels (also non-cloudy ones)
    cloudy_labels = labels[cloudy_labels_mask]
    probabilities = probabilities_from_outputs(outputs, no_cloud_mask)
    if no_cloud_mask:
        cloudy_class_probabilities = probabilities[0:8].reshape(8, -1)[:, cloudy_labels_mask]
        outputs = outputs[0:8].reshape(8, -1)[:, cloudy_labels_mask]
    else:
        mask_probabilities = probabilities[0].copy().reshape(-1)
        cloudy_class_probabilities = probabilities[1:9].reshape(8, -1)[:, cloudy_labels_mask]
        outputs = outputs[1:9].reshape(8, -1)[:, cloudy_labels_mask]

    # --- Generate file-wise evaluation ---
    report = f"#### {file} ####\n\n"
    if not no_cloud_mask:
        mask_report = evaluate_cloud_mask(mask_probabilities, cloud_mask, mask_names, file)
        report += 'Cloud mask eval:\n\n' + mask_report + '\n\n'
    class_report, class_matrix = evaluate_clouds(cloudy_class_probabilities, cloudy_labels, label_names, file)
    report += 'Cloud class eval:\n\n' + class_report + '\n\n'
    report += class_matrix + '\n\n\n\n'

    return report, cloudy_class_probabilities, cloudy_labels, outputs
