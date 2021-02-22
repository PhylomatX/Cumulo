import numpy as np
import torch
import math
import os
import sys
from cumulo.data.loader import CumuloDataset
from cumulo.utils.utils import GlobalNormalizer, TileExtractor
from cumulo.models.unet_weak import UNet_weak
from cumulo.models.unet_equi import UNet_equi
from absl import app
from absl import flags

flags.DEFINE_string('m_name', 'val_best', help='Model name')
flags.DEFINE_integer('nb_classes', 9, help='Number of classes')
flags.DEFINE_integer('pred_num', None, help='Number of prediction files')
flags.DEFINE_string('o_path', None, help='Location for output')
flags.DEFINE_string('d_path', None, help='Data path')
flags.DEFINE_string('m_path', None, help='Model path')
flags.DEFINE_string('filetype', 'nc', help='File type for dataset')
flags.DEFINE_integer('r_seed', 1, help='Random seed')
flags.DEFINE_integer('nb_epochs', 100, help='Number of epochs')
flags.DEFINE_integer('num_workers', 4, help='Number of workers for the dataloader.')
flags.DEFINE_integer('bs', 32, help='Batch size for training and validation.')
flags.DEFINE_integer('dataset_bs', 32, help='Batch size for training and validation.')
flags.DEFINE_integer('tile_num', None, help='Tile number / data set size.')
flags.DEFINE_integer('tile_size', 128, help='Tile size.')
flags.DEFINE_integer('center_distance', None, help='Distance between base points of tile extraction.')
flags.DEFINE_bool('val', False, help='Flag for validation after each epoch.')
flags.DEFINE_string('model', 'weak', help='Option for choosing between UNets.')
flags.DEFINE_bool('merged', False, help='Flag for indicating use of merged dataset')
flags.DEFINE_bool('examples', False, help='Save some training examples in each epoch')
flags.DEFINE_bool('local_norm', True, help='Standardize each image channel-wise. If False the statistics of a data subset will be used.')
flags.DEFINE_integer('examples_num', None, help='How many samples should get saved as example?')
flags.DEFINE_integer('analysis_freq', 1, help='Validation and example save frequency')
flags.DEFINE_integer('rot', 2, help='Number of elements in rotation group')
flags.DEFINE_float('augment_prob', 0, help='Augmentation probability')
FLAGS = flags.FLAGS
# add arg of form --flagfile 'PATH_TO_FLAGFILE' at the beginning


def load_model(model_dir, use_cuda):
    model = None
    if FLAGS.model == 'weak':
        model = UNet_weak(in_channels=13, out_channels=FLAGS.nb_classes, starting_filters=32)
    elif FLAGS.model == 'equi':
        model = UNet_equi(in_channels=13, out_channels=FLAGS.nb_classes, starting_filters=32, rot=FLAGS.rot)
    if model is None:
        raise ValueError('Model type not known.')

    model_path = os.path.join(model_dir, FLAGS.m_name)
    saved = torch.load(model_path)
    model.load_state_dict(saved)
    print('Model loaded!')
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
    m = np.load(os.path.join(FLAGS.d_path, "mean.npy"))
    s = np.load(os.path.join(FLAGS.d_path, "std.npy"))

    # dataset loader
    tile_extr = TileExtractor()
    normalizer = GlobalNormalizer(m, s)
    try:
        test_idx = np.load(os.path.join(FLAGS.m_path, 'test_idx.npy'))
        print(f"Found test set with {len(test_idx)} files.")
    except FileNotFoundError:
        test_idx = None
    if FLAGS.pred_num is not None:
        test_idx = test_idx[:FLAGS.pred_num]
    dataset = CumuloDataset(d_path=FLAGS.d_path, ext="nc", normalizer=normalizer, tiler=tile_extr, pred=True,
                            indices=test_idx, tile_size=FLAGS.tile_size)

    print(f"Predicting {len(dataset)} files.")
    use_cuda = torch.cuda.is_available()
    print("using GPUs?", use_cuda)
    model = load_model(FLAGS.m_path, use_cuda)

    if not os.path.exists(FLAGS.o_path):
        os.makedirs(FLAGS.o_path)

    for swath in dataset:
        filename, tiles, locations, cloud_mask, labels = swath
        labels = include_cloud_mask(labels.data, cloud_mask.data)
        merged = np.ones(cloud_mask.squeeze().shape) * -1
        predictions = predict_tiles(model, tiles, use_cuda, batch_size=FLAGS.bs)
        for ix, loc in enumerate(locations):
            merged[loc[0][0]:loc[0][1], loc[1][0]:loc[1][1]] = predictions[ix]
        np.savez(os.path.join(FLAGS.o_path, filename.replace(FLAGS.d_path, '').replace('.nc', '')),
                 prediction=merged, labels=labels.squeeze())


if __name__ == '__main__':
    FLAGS.read_flags_from_files(sys.argv[1:])
    app.run(main)
