import numpy as np
import torch
import math
import os
from cumulo.data.loader import CumuloDataset
from cumulo.utils.utils import Normalizer, TileExtractor
from cumulo.models.unet_weak import UNet_weak
from cumulo.models.unet_equi import UNet_equi
from absl import app
from absl import flags

flags.DEFINE_string('m_path', None, help='Location of model')
flags.DEFINE_string('nc_path', None, help='Location of nc files')
flags.DEFINE_string('stat_path', None, help='Location of dataset statistics')
flags.DEFINE_string('m_name', 'train_best.pth', help='Model name')
flags.DEFINE_string('m_type', 'weak', help='Model type')
flags.DEFINE_integer('nb_classes', 9, help='Number of classes')
flags.DEFINE_string('o_path', None, help='Location for output')
FLAGS = flags.FLAGS


def load_model(model_dir, use_cuda):
    model = None
    if FLAGS.m_type == 'weak':
        model = UNet_weak(in_channels=13, out_channels=FLAGS.nb_classes, starting_filters=32)
    elif FLAGS.m_type == 'equi':
        model = UNet_equi(in_channels=13, out_channels=FLAGS.nb_classes, starting_filters=32)
    print('Model initialized!')
    model_path = os.path.join(model_dir, FLAGS.m_name)
    if model is None:
        raise ValueError('Model type not known.')
    state_dict = torch.load(model_path)['model_state_dict']
    model.load_state_dict(state_dict)
    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, range(torch.cuda.device_count()))
        torch.backends.cudnn.enabled = False
    model.eval()
    return model


def predict_tiles(model, tiles, use_cuda, batch_size: int = 64):
    b_num = math.ceil(tiles.shape[0] / batch_size)
    ix = 0
    predictions = np.zeros((tiles.shape[0], *tiles.shape[2:]))
    remaining = 0
    for b in range(b_num):
        batch = np.zeros((batch_size, *tiles.shape[1:]))
        upper = ix + batch_size
        if upper > tiles.shape[0]:
            # fill batch with tiles from the beginning to avoid artefacts due to zero
            # tiles at the end
            remaining = tiles.shape[0] - ix
            batch[:remaining] = tiles[ix:]
            batch[remaining:] = tiles[:upper - tiles.shape[0]]
        else:
            batch[:] = tiles[ix:ix+batch_size]
        inputs = torch.from_numpy(batch).float()
        if use_cuda:
            inputs = inputs.cuda()
        outputs = model(inputs)
        outputs = torch.argmax(outputs, 1).cpu().detach().numpy()
        if upper > tiles.shape[0]:
            predictions[ix:] = outputs[:remaining]
        else:
            predictions[ix:ix+batch_size] = outputs
        ix += batch_size
    return predictions


def include_cloud_mask(labels, cloud_mask):
    labels[labels >= 0] += 1
    return labels * cloud_mask


def main(_):
    m = np.load(os.path.join(FLAGS.stat_path, "mean.npy"))
    s = np.load(os.path.join(FLAGS.stat_path, "std.npy"))

    # dataset loader
    tile_extr = TileExtractor()
    normalizer = Normalizer(m, s)
    dataset = CumuloDataset(d_path=FLAGS.nc_path, ext="nc", normalizer=normalizer, tiler=tile_extr, pred=True)

    use_cuda = torch.cuda.is_available()
    print("using GPUs?", use_cuda)
    model = load_model(FLAGS.m_path, use_cuda)

    if not os.path.exists(FLAGS.o_path):
        os.makedirs(FLAGS.o_path)

    for swath in dataset:
        filename, tiles, locations, cloud_mask, labels = swath
        labels = include_cloud_mask(labels.data, cloud_mask.data)
        merged = np.ones(cloud_mask.squeeze().shape) * -1
        predictions = predict_tiles(model, tiles, use_cuda)
        for ix, loc in enumerate(locations):
            merged[loc[0][0]:loc[0][1], loc[1][0]:loc[1][1]] = predictions[ix]
        np.savez(os.path.join(FLAGS.o_path, filename.replace(FLAGS.nc_path, '').replace('.nc', '')),
                 prediction=merged, labels=labels.squeeze())


if __name__ == '__main__':
    app.run(main)
