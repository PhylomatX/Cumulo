import numpy as np
import torch
import math
import os
import sys
from tqdm import tqdm
from cumulo.data.loader import CumuloDataset
from cumulo.utils.utils import GlobalNormalizer, LocalNormalizer, TileExtractor, include_cloud_mask
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
flags.DEFINE_string('norm', 'bn', help='Type of normalization, one of [bn, gn, none]')
flags.DEFINE_bool('merged', False, help='Flag for indicating use of merged dataset')
flags.DEFINE_bool('examples', False, help='Save some training examples in each epoch')
flags.DEFINE_bool('local_norm', True, help='Standardize each image channel-wise. If False the statistics of a data subset will be used.')
flags.DEFINE_integer('examples_num', None, help='How many samples should get saved as example?')
flags.DEFINE_integer('analysis_freq', 1, help='Validation and example save frequency')
flags.DEFINE_integer('rot', 2, help='Number of elements in rotation group')
flags.DEFINE_integer('offset', 0, help='Cropping offset for labels in case of valid convolutions')
flags.DEFINE_integer('padding', 0, help='Padding for convolutions')
flags.DEFINE_float('augment_prob', 0, help='Augmentation probability')
flags.DEFINE_bool('raw_predictions', True, help='Save network outputs for later visualization')
FLAGS = flags.FLAGS

# add arg of form --flagfile 'PATH_TO_FLAGFILE' at the beginning and add --o_path and --pred_num


def load_model(model_dir):
    if FLAGS.model == 'weak':
        model = UNet_weak(in_channels=13, out_channels=FLAGS.nb_classes, starting_filters=32, padding=FLAGS.padding)
    elif FLAGS.model == 'equi':
        model = UNet_equi(in_channels=13, out_channels=FLAGS.nb_classes, starting_filters=32, rot=FLAGS.rot)
    else:
        raise ValueError('Model type not known.')
    model_path = os.path.join(model_dir, FLAGS.m_name)
    model.load_state_dict(torch.load(model_path))
    return model.eval()


@torch.no_grad()
def predict_tiles(model, tiles, device, batch_size):
    b_num = math.ceil(tiles.shape[0] / batch_size)
    ix = 0
    output_size = FLAGS.tile_size - 2 * FLAGS.offset
    if FLAGS.raw_predictions:
        predictions = np.zeros((tiles.shape[0], FLAGS.nb_classes, output_size, output_size))
    else:
        predictions = np.zeros((tiles.shape[0], output_size, output_size))
    remaining = 0

    for b in range(b_num):
        # --- batch building ---
        batch = np.zeros((batch_size, *tiles.shape[1:]))
        upper = ix + batch_size
        if upper > tiles.shape[0]:
            # --- fill batch with tiles from the beginning to avoid artefacts due to zero tiles at the end ---
            remaining = tiles.shape[0] - ix
            batch[:remaining] = tiles[ix:]
            batch[remaining:] = tiles[:upper - tiles.shape[0]]
        else:
            batch[:] = tiles[ix:ix+batch_size]

        # --- inference ---
        inputs = torch.from_numpy(batch).float()
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = outputs.cpu().detach()

        # --- postprocessing ---
        if FLAGS.raw_predictions:
            outputs[:, 1:, ...] = torch.softmax(outputs[:, 1:, ...], dim=1)  # last 8 channels are trained with LogSoftMax and CrossEntropy
            outputs[:, 0, ...] = torch.sigmoid(outputs[:, 0, ...])  # first channel is trained with Sigmoid and BinaryCrossEntropy
            outputs = outputs.numpy()
        else:
            outputs = outputs.numpy()
            cloud_mask_pred = outputs[:, 0, ...]
            cloud_mask_pred[cloud_mask_pred < 0.5] = 0
            cloud_mask_pred[cloud_mask_pred >= 0.5] = 1
            cloud_class_pred = np.argmax(outputs[:, 1:, ...], axis=1)
            outputs = include_cloud_mask(cloud_class_pred, cloud_mask_pred)
        if upper > tiles.shape[0]:
            predictions[ix:] = outputs[:remaining]
        else:
            predictions[ix:ix+batch_size] = outputs
        ix += batch_size
    return predictions


def main(_):
    if not os.path.exists(FLAGS.o_path):
        os.makedirs(FLAGS.o_path)
    with open(os.path.join(FLAGS.o_path, 'eval_flagfile.txt'), 'w') as f:
        f.writelines(FLAGS.flags_into_string())

    if FLAGS.local_norm:
        normalizer = LocalNormalizer()
    else:
        m = np.load(os.path.join(FLAGS.d_path, "mean.npy"))
        s = np.load(os.path.join(FLAGS.d_path, "std.npy"))
        normalizer = GlobalNormalizer(m, s)
    try:
        test_idx = np.load(os.path.join(FLAGS.m_path, 'test_idx.npy'))
        print(f"Found test set with {len(test_idx)} files.")
    except FileNotFoundError:
        test_idx = None
    if FLAGS.pred_num is not None:
        test_idx = test_idx[:FLAGS.pred_num]
    tile_extr = TileExtractor(FLAGS.tile_size, offset=FLAGS.offset)
    dataset = CumuloDataset(d_path=FLAGS.d_path, ext="nc", normalizer=normalizer, tiler=tile_extr, pred=True,
                            indices=test_idx, tile_size=FLAGS.tile_size)
    print(f"Predicting {len(dataset)} files.")

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    model = load_model(FLAGS.m_path)
    model.to(device)

    for swath in tqdm(dataset):
        filename, tiles, locations, label_tiles, cloud_mask, labels = swath
        predictions = predict_tiles(model, tiles, device, FLAGS.dataset_bs)
        merged = np.ones((FLAGS.nb_classes if FLAGS.raw_predictions else 1, *cloud_mask.squeeze().shape)) * -1
        for ix, loc in enumerate(locations):
            merged[:, loc[0][0]:loc[0][1], loc[1][0]:loc[1][1]] = predictions[ix]

        # Remove unpredicted border region possibly caused by offset / valid convolutions
        merged = merged[:, FLAGS.offset:-1 - FLAGS.offset, FLAGS.offset:-1 - FLAGS.offset]
        labels = labels.squeeze()[FLAGS.offset:-1 - FLAGS.offset, FLAGS.offset:-1 - FLAGS.offset]
        cloud_mask = cloud_mask.squeeze()[FLAGS.offset:-1 - FLAGS.offset, FLAGS.offset:-1 - FLAGS.offset]

        np.savez(os.path.join(FLAGS.o_path, filename.replace(FLAGS.d_path, '').replace('.nc', '')),
                 prediction=merged, labels=labels, cloud_mask=cloud_mask)


if __name__ == '__main__':
    FLAGS.read_flags_from_files(sys.argv[1:])
    app.run(main)
