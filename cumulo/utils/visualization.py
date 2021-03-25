import numpy as np
from cumulo.utils.basics import include_cloud_mask, probabilities_from_outputs
import matplotlib.pyplot as plt
import imageio

COLORS = np.array([[247., 22., 232.],  # pink
                   [229., 51., 51.],  # red
                   [232., 232., 21.],  # yellow
                   [16., 204., 204.],  # turquoise
                   [14., 49., 156.],  # blue
                   [127., 25., 229.],  # purple
                   [219., 146., 0.],  # orange
                   [12., 171., 3.]]) / 255  # green

BORDER = np.array([255., 255., 255.]) / 255  # white
NO_CLOUD = np.array([5., 5., 5.]) / 255  # black
NO_LABEL = np.array([250., 250., 250.]) / 255  # white


def prediction_to_continuous_rgb(prediction, cloud_mask_is_binary=True):
    clouds = np.matmul(prediction[1:9, ...].transpose(), COLORS ** 2)
    clouds = np.swapaxes(clouds, 0, 1)
    if cloud_mask_is_binary:
        prediction[0][prediction[0] < 0.5] = 0
        prediction[0][prediction[0] >= 0.5] = 1
    cloud_mask_prediction = np.expand_dims(prediction[0], -1)
    prediction = clouds * cloud_mask_prediction + (1 - cloud_mask_prediction) * NO_CLOUD ** 2
    prediction = np.sqrt(prediction)
    prediction[np.all(prediction == 0, 2)] = NO_CLOUD
    return prediction


def prediction_to_discrete_rgb(prediction):
    prediction[0][prediction[0] < 0.5] = 0
    prediction[0][prediction[0] >= 0.5] = 1
    flat = np.argmax(prediction[1:9, ...], 0)
    prediction = include_cloud_mask(flat, prediction[0]).astype(np.int)
    colors_merged = np.vstack([NO_CLOUD, COLORS])
    prediction = colors_merged[prediction.reshape(-1)].reshape(*prediction.shape, 3)
    return prediction


def labels_and_cloud_mask_to_rgb(labels, cloud_mask):
    labels = include_cloud_mask(labels, cloud_mask)
    colors_merged = np.vstack([NO_CLOUD, COLORS, NO_LABEL])
    flat_labels = labels.reshape(-1)
    return colors_merged[flat_labels].reshape(*labels.shape, 3)


def labels_to_rgb(labels):
    labels = labels.copy()
    labels[labels >= 0] += 1
    colors_merged = np.vstack([NO_CLOUD, COLORS, BORDER, NO_LABEL])
    flat_labels = labels.reshape(-1)
    return colors_merged[flat_labels].reshape(*labels.shape, 3)


def prediction_to_figure(prediction, ground_truth, cloud_mask_as_binary=True):
    if cloud_mask_as_binary:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].set_title('Predicted cloud classes')
    axs[0].imshow(prediction)
    axs[1].set_title('Ground Truth')
    axs[1].imshow(ground_truth)
    if cloud_mask_as_binary:
        prediction[np.any(prediction != NO_CLOUD[0], 2)] = NO_LABEL
        axs[2].set_title('Predicted cloud mask')
        axs[2].imshow(prediction)
    return fig


def prediction_to_file(npz_file, rgb_predictions, rgb_ground_truth, rgb_labels, cloud_mask_as_binary=True):
    imageio.imwrite(npz_file.replace('.npz', '_classpred.png'), (rgb_predictions * 255).astype(np.uint8))
    imageio.imwrite(npz_file.replace('.npz', '_gt.png'), (rgb_ground_truth * 255).astype(np.uint8))
    # --- project labeled pixels into predictions ---
    rgb_predictions_cache = rgb_predictions.copy()
    label_mask = np.all(rgb_labels != NO_LABEL, -1)
    rgb_predictions[label_mask, :] = rgb_labels[label_mask, :]
    imageio.imwrite(npz_file.replace('.npz', '_classpred_labels.png'), (rgb_predictions * 255).astype(np.uint8))
    # --- save binary cloud mask prediction ---
    if cloud_mask_as_binary:
        rgb_predictions = rgb_predictions_cache
        rgb_predictions[np.any(rgb_predictions != NO_CLOUD[0], 2)] = NO_LABEL
        imageio.imwrite(npz_file.replace('.npz', '_maskpred.png'), (rgb_predictions * 255).astype(np.uint8))


def outputs_to_figure_or_file(outputs, labels, cloud_mask, use_continuous_colors=True, cloud_mask_as_binary=True,
                              to_file=True, npz_file='', label_dilation=10, border_dilation=2):
    prediction = probabilities_from_outputs(outputs)
    if use_continuous_colors:
        rgb_prediction = prediction_to_continuous_rgb(prediction, cloud_mask_as_binary)
    else:
        rgb_prediction = prediction_to_discrete_rgb(prediction)

    # --- dilate labeled pixels for better visualization ---
    labels_cache = labels
    labeled_pixels = labels != -1
    labeled_pixels = np.array(list(zip(*np.where(labeled_pixels == 1))))
    for pixel in labeled_pixels:
        for y in range(max(pixel[0] - label_dilation, 0), min(pixel[0] + label_dilation, labels.shape[0])):
            for x in range(max(pixel[1] - label_dilation, 0), min(pixel[1] + label_dilation, labels.shape[1])):
                if y < pixel[0] - label_dilation + border_dilation or y > pixel[0] + label_dilation - border_dilation * 2:
                    labels[y, x] = 8
                else:
                    labels[y, x] = labels_cache[pixel[0], pixel[1]]

    rgb_labels = labels_to_rgb(labels)
    rgb_ground_truth = labels_and_cloud_mask_to_rgb(labels, cloud_mask)

    if to_file:
        prediction_to_file(npz_file, rgb_prediction, rgb_ground_truth, rgb_labels, cloud_mask_as_binary)
    else:
        prediction_to_figure(rgb_prediction, rgb_ground_truth, cloud_mask_as_binary)
        plt.show()
        plt.close()
