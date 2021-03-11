import os
import numpy as np
import matplotlib.pyplot as plt
from absl import app
from absl import flags
from cumulo.utils.utils import include_cloud_mask

flags.DEFINE_string('path', None, help='Location of predictions')
flags.DEFINE_boolean('save', False, help='Save predictions as images')
flags.DEFINE_boolean('continuous', True, help='Use predictions for color weighting')
flags.DEFINE_boolean('binary_mask', True, help='Make cloud mask binary')
flags.DEFINE_integer('offset', 46, help='Offset of predictions to border to exclude in view')
FLAGS = flags.FLAGS

colors = np.array([[153., 153., 153.],  # grey
                   [229., 51., 51.],  # red
                   [232., 232., 21.],  # yellow
                   [16., 204., 204.],  # turquoise
                   [14., 49., 156.],  # blue
                   [127., 25., 229.],  # purple
                   [219., 146., 0.],  # orange
                   [12., 171., 3.]]) / 255  # green

no_cloud = np.array([5., 5., 5.]) / 255
no_label = np.array([250., 250., 250.]) / 255


def main(_):
    files = os.listdir(FLAGS.path)
    for file in files:
        if 'npz' not in file:
            continue
        print(file)
        data = np.load(os.path.join(FLAGS.path, file))
        prediction = data['prediction']
        if prediction.ndim == 3:
            if FLAGS.continuous:
                clouds = np.matmul(prediction[1:, ...].transpose(), colors ** 2)
                clouds = np.swapaxes(clouds, 0, 1)
                if FLAGS.binary_mask:
                    prediction[0][prediction[0] < 0.5] = 0
                    prediction[0][prediction[0] >= 0.5] = 1
                cloud_mask_pred = np.expand_dims(prediction[0], -1)
                prediction = clouds * cloud_mask_pred + (1 - cloud_mask_pred) * no_cloud ** 2
                prediction = np.sqrt(prediction)
                prediction[np.all(prediction == 0, 2)] = no_cloud
            else:
                prediction[0][prediction[0] < 0.5] = 0
                prediction[0][prediction[0] >= 0.5] = 1
                flat = np.argmax(prediction[1:, ...], 0)
                prediction = include_cloud_mask(flat, prediction[0]).astype(np.int)
                colors_merged = np.vstack([no_cloud, colors])
                prediction = colors_merged[prediction.reshape(-1)].reshape(*prediction.shape, 3)
        else:
            prediction = np.expand_dims(prediction, -1)
            colors_merged = np.vstack([no_cloud, colors])
            prediction = colors_merged[prediction.reshape(-1)].reshape(*prediction.shape, 3)
        prediction = prediction[FLAGS.offset:-1 - FLAGS.offset, FLAGS.offset:-1 - FLAGS.offset, :]
        labels = data['labels'][FLAGS.offset:-1 - FLAGS.offset, FLAGS.offset:-1 - FLAGS.offset]
        cloud_mask = data['cloud_mask'][FLAGS.offset:-1 - FLAGS.offset, FLAGS.offset:-1 - FLAGS.offset]
        labels = include_cloud_mask(labels, cloud_mask)
        colors_merged = np.vstack([no_cloud, colors, no_label])
        flat_labels = labels.reshape(-1)
        labels = colors_merged[flat_labels].reshape(*labels.shape, 3)
        if not FLAGS.save:
            if FLAGS.binary_mask:
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            else:
                fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            axs[0].set_title('Predicted cloud classes')
            axs[0].imshow(prediction)
            axs[1].set_title('Ground Truth')
            axs[1].imshow(labels)
            if FLAGS.binary_mask:
                prediction[np.any(prediction != no_cloud[0], 2)] = no_label
                axs[2].set_title('Predicted cloud mask')
                axs[2].imshow(prediction)
            plt.show()
            plt.close()
        else:
            figsize = (15, 12)
            fig, axs = plt.subplots(1, 1, figsize=figsize)
            plt.title('Predicted cloud classes')
            plt.imshow(prediction)
            plt.tight_layout()
            plt.savefig(os.path.join(FLAGS.path, file.replace('.npz', '_classpred.png')))
            plt.close()
            fig, axs = plt.subplots(1, 1, figsize=figsize)
            plt.title('Ground Truth')
            plt.imshow(labels)
            plt.tight_layout()
            plt.savefig(os.path.join(FLAGS.path, file.replace('.npz', '_gt.png')))
            plt.close()
            if FLAGS.binary_mask:
                fig, axs = plt.subplots(1, 1, figsize=figsize)
                prediction[np.any(prediction != no_cloud[0], 2)] = no_label
                plt.title('Predicted cloud mask')
                plt.imshow(prediction)
                plt.tight_layout()
                plt.savefig(os.path.join(FLAGS.path, file.replace('.npz', '_cloudmaskpred.png')))
                plt.close()


if __name__ == '__main__':
    app.run(main)
