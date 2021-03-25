from absl import app
from absl import flags
import time
from cumulo.data.loader import CumuloDataset
import random
import torch
import numpy as np

flags.DEFINE_string('path', None, help='Directory where nc files are located.')

FLAGS = flags.FLAGS


def main(_):
    dataset = CumuloDataset(FLAGS.path, indices=np.arange(53), batch_size=64, tile_size=128)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1, num_workers=8)
    total_time = 0
    r_seed = 6

    torch.manual_seed(r_seed)
    torch.cuda.manual_seed(r_seed)
    np.random.seed(r_seed)
    random.seed(r_seed)
    torch.backends.cudnn.deterministic = True

    # idcs = [1012, 1011, 1010, 1009]

    for epoch in range(5):
        for ix, res in enumerate(dataloader):
            print(f'{ix} - {res}')
            # fig, axs = plt.subplots(1, 2, figsize=(15, 8))
            # axs[0].imshow(rads[0][0])
            # axs[1].imshow(labs[0])
            # plt.title(f'Iteration {epoch}')
            # plt.show()
            # plt.close()
        dataset.next_epoch()

    print(total_time / len(dataset))


if __name__ == '__main__':
    app.run(main)
