import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics as sm

from cumulo.utils.basics import probabilities_from_outputs
from scripts.pipeline.evaluate import FLAGS


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


def create_histogram(class_predictions, file):
    for class_prediction in class_predictions:
        # --- determine bin number with Freedman-Diaconis rule ---
        n = len(class_prediction)
        sorted_cloudy = np.sort(class_prediction)
        q1 = np.median(sorted_cloudy[:int(n / 2)])
        q3 = np.median(sorted_cloudy[-int(n / 2):])
        bin_width = 2 * (q3 - q1) / np.cbrt(n)
        bin_num = int(1 / bin_width)
        cloudy_n, _, _ = plt.hist(class_prediction, density=True, bins=bin_num, histtype='step')

    plt.xlabel('Network output after softmax')
    plt.ylabel('Normalized counts')
    plt.tight_layout()
    plt.savefig(file)
    plt.close()


def evaluate_cloud_mask(mask_predictions, mask, mask_names, npz_file):
    mask_predictions = mask_predictions.reshape(-1)
    hard_mask_predictions = mask_predictions.copy()
    hard_mask_predictions[mask_predictions < 0.5] = 0
    hard_mask_predictions[mask_predictions >= 0.5] = 1
    report = sm.classification_report(mask, hard_mask_predictions, labels=mask_names)

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
    create_histogram([cloudy, not_cloudy], npz_file.replace('.npz', f'_mask_hist.png'))
    return report


def evaluate_clouds(cloudy_probabilities, cloudy_labels, label_names, npz_file):
    hard_cloudy_predictions = np.argmax(cloudy_probabilities, 0).reshape(-1)

    report = sm.classification_report(cloudy_labels, hard_cloudy_predictions, labels=label_names, zero_division=0)
    matrix = sm.confusion_matrix(cloudy_labels, hard_cloudy_predictions)
    matrix = write_confusion_matrix(matrix, get_target_names(cloudy_labels, hard_cloudy_predictions, label_names))
    class_matrix_disp = sm.ConfusionMatrixDisplay(matrix, display_labels=label_names)
    class_matrix_disp.plot(cmap='Reds')
    plt.savefig(npz_file.replace('.npz', '_matrix.png'))

    class_matrix_normalized = sm.confusion_matrix(cloudy_labels, hard_cloudy_predictions, normalize='true')
    class_matrix_disp = sm.ConfusionMatrixDisplay(class_matrix_normalized, display_labels=label_names)
    class_matrix_disp.plot(include_values=False, cmap='Reds')
    plt.savefig(npz_file.replace('.npz', '_matrix_normalized.png'))

    histogram_predictions = []
    for ix in range(len(label_names)):
        if np.all(cloudy_labels != ix):
            continue
        ix_labels = np.zeros_like(cloudy_labels)
        ix_labels[cloudy_labels == ix] = 1
        ix_predictions = cloudy_probabilities[ix].reshape(-1)
        histogram_predictions.append(ix_predictions)

        fpr, tpr, _ = sm.roc_curve(ix_labels, ix_predictions)
        auc = sm.auc(fpr, tpr)
        mask_roc_disp = sm.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
        mask_roc_disp.plot()
        plt.savefig(npz_file.replace('.npz', f'_{label_names[ix]}_roc.png'))
        plt.close()

        precision, recall, _ = sm.precision_recall_curve(ix_labels, ix_predictions)
        mask_pr_disp = sm.PrecisionRecallDisplay(precision=precision, recall=recall)
        mask_pr_disp.plot()
        plt.savefig(npz_file.replace('.npz', f'_{label_names[ix]}_pr.png'))
        plt.close()

        create_histogram([ix_predictions[ix_labels.astype(bool)], ix_predictions[~ix_labels.astype(bool)]],
                         npz_file.replace('.npz', f'_{label_names[ix]}_hist.png'))
    create_histogram(histogram_predictions, npz_file.replace('.npz', f'_predictions_hist.png'))
    return report, matrix


def evaluate_file(file, outputs, labels, cloud_mask, label_names, mask_names, report, total_probabilities, total_labels):
    labels = labels.reshape(-1)
    cloud_mask = cloud_mask.reshape(-1)
    cloudy_labels = labels[cloud_mask == 1]
    probabilities = probabilities_from_outputs(outputs)
    mask_probabilities = probabilities[0].copy().reshape(-1)
    cloudy_class_probabilities = probabilities[1:9].reshape(8, -1)[:, cloud_mask == 1]

    # --- Generate file-wise evaluation and save intermediate report ---
    report += f"#### {file.replace(FLAGS.path, '')} ####\n\n"
    mask_report = evaluate_cloud_mask(mask_probabilities, cloud_mask, mask_names, file)
    report += 'Cloud mask eval:\n\n' + mask_report + '\n\n'
    class_report, class_matrix = evaluate_clouds(cloudy_class_probabilities, cloudy_labels, label_names, file)
    report += 'Cloud class eval:\n\n' + class_report + '\n\n'
    report += class_matrix + '\n\n\n\n'

    # --- Save probabilities and labels for total evaluation ---
    total_labels = np.append(total_labels, cloudy_labels)
    if total_probabilities is None:
        total_probabilities = cloudy_class_probabilities
    else:
        total_probabilities = np.vstack(total_probabilities, cloudy_class_probabilities)
    return report, total_labels, total_probabilities