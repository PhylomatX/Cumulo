from absl import app
from absl import flags
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

flags.DEFINE_string('path', None, help='Directory where npz files are located.')
FLAGS = flags.FLAGS


def main(_):
    with open(FLAGS.path, 'rb') as f:
        metrics = pkl.load(f)
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    axs[0].plot(np.arange(len(metrics['train_loss'])), metrics['train_loss'], label='train')
    if len(metrics['val_loss']) > 0:
        axs[0].plot(np.arange(len(metrics['val_loss'])), metrics['val_loss'], label='val')
    axs[0].set_title('Losses')
    axs[0].legend()
    axs[1].plot(np.arange(len(metrics['train_acc'])), metrics['train_acc'], label='train')
    if len(metrics['val_acc']) > 0:
        axs[1].plot(np.arange(len(metrics['val_acc'])), metrics['val_acc'], label='val')
    axs[1].set_title('Accuracies')
    axs[0].legend()
    plt.savefig(FLAGS.path.replace('.pkl', '.png'))


if __name__ == '__main__':
    app.run(main)
