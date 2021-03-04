import os
import numpy as np
import matplotlib.pyplot as plt
from absl import app
from absl import flags
from cumulo.utils.utils import include_cloud_mask

flags.DEFINE_string('path', None, help='Location of predictions')
flags.DEFINE_boolean('save', False, help='Save predictions as images')
flags.DEFINE_integer('offset', 0, help='Offset of predictions to border to exclude in view')
FLAGS = flags.FLAGS


def main(_):
    files = os.listdir(FLAGS.path)
    for file in files:
        if 'npz' not in file:
            continue
        print(file)
        data = np.load(os.path.join(FLAGS.path, file))
        prediction = data['prediction']
        prediction = prediction[FLAGS.offset:-1-FLAGS.offset, FLAGS.offset:-1-FLAGS.offset]
        labels = data['labels'][FLAGS.offset:-1-FLAGS.offset, FLAGS.offset:-1-FLAGS.offset]
        cloud_mask = data['cloud_mask'][FLAGS.offset:-1-FLAGS.offset, FLAGS.offset:-1-FLAGS.offset]
        labels = include_cloud_mask(labels, cloud_mask)
        if not FLAGS.save:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].set_title('Predicted cloud classes')
            axs[0].imshow(prediction)
            axs[1].set_title('Ground Truth')
            axs[1].imshow(labels)
            prediction[prediction != 0] = 1
            axs[2].set_title('Predicted cloud mask')
            axs[2].imshow(prediction)
            plt.show()
            plt.close()
        else:
            figsize = (15, 12)
            fig, axs = plt.subplots(1, 1, figsize=figsize)
            plt.title('Predicted cloud classes')
            plt.imshow(prediction)
            plt.axis('off')
            plt.savefig(os.path.join(FLAGS.path, file.replace('.npz', '_classpred.png')))
            plt.close()
            fig, axs = plt.subplots(1, 1, figsize=figsize)
            plt.title('Cloud mask (GT)')
            plt.imshow(cloud_mask)
            plt.axis('off')
            plt.savefig(os.path.join(FLAGS.path, file.replace('.npz', '_cloudmaskgt.png')))
            plt.close()
            fig, axs = plt.subplots(1, 1, figsize=figsize)
            prediction[prediction != 0] = 1
            plt.title('Predicted cloud mask')
            plt.imshow(prediction)
            plt.axis('off')
            plt.savefig(os.path.join(FLAGS.path, file.replace('.npz', '_cloudmaskpred.png')))
            plt.close()


if __name__ == '__main__':
    app.run(main)
