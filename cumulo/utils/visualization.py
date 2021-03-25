import numpy as np
from cumulo.utils.basics import include_cloud_mask, probabilities_from_outputs
import matplotlib.pyplot as plt
import imageio

COLORS = np.array([[153., 153., 153.],  # grey
                   [229., 51., 51.],  # red
                   [232., 232., 21.],  # yellow
                   [16., 204., 204.],  # turquoise
                   [14., 49., 156.],  # blue
                   [127., 25., 229.],  # purple
                   [219., 146., 0.],  # orange
                   [12., 171., 3.]]) / 255  # green

NO_CLOUD = np.array([5., 5., 5.]) / 255
NO_LABEL = np.array([250., 250., 250.]) / 255


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


def prediction_to_file(npz_file, prediction, ground_truth, cloud_mask_as_binary=True):
    imageio.imwrite(npz_file.replace('.npz', '_classpred.png'), (prediction * 255).astype(np.uint8))
    imageio.imwrite(npz_file.replace('.npz', '_gt.png'), (ground_truth * 255).astype(np.uint8))
    if cloud_mask_as_binary:
        prediction[np.any(prediction != NO_CLOUD[0], 2)] = NO_LABEL
        imageio.imwrite(npz_file.replace('.npz', '_maskpred.png'), (prediction * 255).astype(np.uint8))


def outputs_to_figure_or_file(outputs, labels, cloud_mask, use_continuous_colors=True, cloud_mask_as_binary=True, to_file=True, file=''):
    prediction = probabilities_from_outputs(outputs)
    if use_continuous_colors:
        prediction = prediction_to_continuous_rgb(prediction, cloud_mask_as_binary)
    else:
        prediction = prediction_to_discrete_rgb(prediction)

    ground_truth = labels_and_cloud_mask_to_rgb(labels, cloud_mask)

    if to_file:
        prediction_to_file(file, prediction, ground_truth, cloud_mask_as_binary)
    else:
        prediction_to_figure(prediction, ground_truth, cloud_mask_as_binary)
        plt.show()
        plt.close()
