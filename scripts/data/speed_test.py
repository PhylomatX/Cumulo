import time
from absl import app
from absl import flags
from cumulo.data.loader import CumuloDataset

flags.DEFINE_string('npz_path', None, help='Directory where npz files are located.')
flags.DEFINE_string('h5_path', None, help='Directory where h5 files are located.')
FLAGS = flags.FLAGS


def main(_):
    dataset = CumuloDataset(FLAGS.npz_path, ext="npz")
    print(len(dataset))
    total = 0
    mean = 0
    for sample in range(len(dataset)):
        start = time.time()
        radiances, labels = dataset[sample]
        mean += radiances.mean()
        total += time.time() - start
    print(f'npz: {total / len(dataset)}s')
    print(f'mean: {mean / len(dataset)}')

    dataset = CumuloDataset(FLAGS.h5_path, ext="h5", file_size=1000)
    print(len(dataset))
    total = 0
    mean = 0
    for sample in range(len(dataset)):
        start = time.time()
        radiances, labels = dataset[sample]
        mean += radiances.mean()
        total += time.time() - start
    print(f'h5: {total / len(dataset)}s')
    print(f'mean: {mean / len(dataset)}')


if __name__ == '__main__':
    app.run(main)
