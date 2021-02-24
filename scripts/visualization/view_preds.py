import os
import numpy as np
import matplotlib.pyplot as plt
from absl import app
from absl import flags
from cumulo.utils.utils import include_cloud_mask

flags.DEFINE_string('path', None, help='Location of predictions')
FLAGS = flags.FLAGS


def main(_):
    files = os.listdir(FLAGS.path)
    for file in files:
        if 'npz' not in file:
            continue
        print(file)
        data = np.load(os.path.join(FLAGS.path, file))
        prediction = data['prediction']
        height = len(prediction[:, 0][prediction[:, 1] != -1])
        width = len(prediction[0, :][prediction[0, :] != -1])
        prediction = prediction[:height, :width]
        labels = data['labels'][:height, :width]
        cloud_mask = data['cloud_mask'][:height, :width]
        labels = include_cloud_mask(labels, cloud_mask)
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


if __name__ == '__main__':
    app.run(main)
