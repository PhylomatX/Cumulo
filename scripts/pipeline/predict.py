import numpy as np
import torch
import math
import os
import sys
from tqdm import tqdm
from cumulo.data.loader import CumuloDataset
from cumulo.utils.training import GlobalNormalizer, LocalNormalizer
from cumulo.models.unet_weak import UNet_weak
from cumulo.models.unet_equi import UNet_equi
from absl import app
from .flags import FLAGS

# add arg of form --flagfile 'PATH_TO_FLAGFILE' at the beginning and add --o_path and --pred_num


def load_model(model_dir):
    if FLAGS.model == 'weak':
        model = UNet_weak(in_channels=13, out_channels=FLAGS.nb_classes, starting_filters=32, padding=FLAGS.padding)
    elif FLAGS.model == 'equi':
        model = UNet_equi(in_channels=13, out_channels=FLAGS.nb_classes, starting_filters=32, rot=FLAGS.rot)
    else:
        raise ValueError('Model type not known.')
    model_path = os.path.join(model_dir, FLAGS.model_name)
    model.load_state_dict(torch.load(model_path))
    return model.eval()


@torch.no_grad()
def predict_tiles(model, tiles, device, batch_size):
    b_num = math.ceil(tiles.shape[0] / batch_size)
    ix = 0
    output_size = FLAGS.tile_size - 2 * FLAGS.valid_convolution_offset
    predictions = np.zeros((tiles.shape[0], FLAGS.nb_classes, output_size, output_size))
    remaining = 0

    for b in range(b_num):
        # --- batch building ---
        batch = np.zeros((batch_size, *tiles.shape[1:]))
        upper = ix + batch_size
        if upper > tiles.shape[0]:
            # fill batch with tiles from the beginning to avoid artefacts due to zero tiles at the end
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

        if upper > tiles.shape[0]:
            predictions[ix:] = outputs[:remaining]
        else:
            predictions[ix:ix+batch_size] = outputs
        ix += batch_size
    return predictions


def main(_):
    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)
    with open(os.path.join(FLAGS.output_path, 'eval_flagfile.txt'), 'w') as f:
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
    if FLAGS.prediction_number is not None:
        test_idx = test_idx[:FLAGS.prediction_number]
    dataset = CumuloDataset(d_path=FLAGS.d_path, normalizer=normalizer, indices=test_idx, prediction_mode=True,
                            tile_size=FLAGS.tile_size, valid_convolution_offset=FLAGS.valid_convolution_offset)
    print(f"Predicting {len(dataset)} files.")

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    model = load_model(FLAGS.m_path)
    model.to(device)

    for swath in tqdm(dataset):
        filename, tiles, locations, cloud_mask, labels = swath
        predictions = predict_tiles(model, tiles, device, FLAGS.dataset_bs)
        outputs = np.ones((FLAGS.nb_classes if FLAGS.raw_predictions else 1, *cloud_mask.squeeze().shape)) * -1
        for ix, loc in enumerate(locations):
            outputs[:, loc[0][0]:loc[0][1], loc[1][0]:loc[1][1]] = predictions[ix]

        # Remove unpredicted border region possibly caused by offset / valid convolutions
        outputs = outputs[:, FLAGS.valid_convolution_offset:-1 - FLAGS.valid_convolution_offset, FLAGS.valid_convolution_offset:-1 - FLAGS.valid_convolution_offset]
        labels = labels.squeeze()[FLAGS.valid_convolution_offset:-1 - FLAGS.valid_convolution_offset, FLAGS.valid_convolution_offset:-1 - FLAGS.valid_convolution_offset]
        cloud_mask = cloud_mask.squeeze()[FLAGS.valid_convolution_offset:-1 - FLAGS.valid_convolution_offset, FLAGS.valid_convolution_offset:-1 - FLAGS.valid_convolution_offset]

        np.savez(os.path.join(FLAGS.output_path, filename.replace(FLAGS.d_path, '').replace('.nc', '')),
                 outputs=outputs, labels=labels, cloud_mask=cloud_mask)


if __name__ == '__main__':
    FLAGS.read_flags_from_files(sys.argv[1:])
    app.run(main)
