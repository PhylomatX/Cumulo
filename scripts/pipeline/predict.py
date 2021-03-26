import numpy as np
import torch
import math
import os
import sys
from tqdm import tqdm
from cumulo.data.loader import CumuloDataset
from cumulo.utils.visualization import outputs_to_figure_or_file
from cumulo.utils.training import GlobalNormalizer, LocalNormalizer
from cumulo.utils.evaluation import evaluate_file, evaluate_clouds
from cumulo.models.unet_weak import UNet_weak
from cumulo.models.unet_equi import UNet_equi
from absl import app
from flags import FLAGS

# add arg of form --flagfile 'PATH_TO_FLAGFILE' at the beginning and add --output_path and --prediction_number


def load_model(model_dir):
    if FLAGS.model == 'weak':
        model = UNet_weak(in_channels=13, out_channels=FLAGS.nb_classes, starting_filters=32, padding=FLAGS.padding, norm=FLAGS.norm)
    elif FLAGS.model == 'equi':
        model = UNet_equi(in_channels=13, out_channels=FLAGS.nb_classes, starting_filters=32, padding=FLAGS.padding, norm=FLAGS.norm, rot=FLAGS.rot)
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


# noinspection PyUnboundLocalVariable
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

    if FLAGS.immediate_evaluation:
        total_report = ''
        mask_names = [0, 1]  # cloud mask targets ('no cloud', 'cloud')
        label_names = list(range(8))  # cloud class targets (8 different cloud types)
        total_labels = np.array([])
        total_probabilities = None

    for swath in tqdm(dataset):
        filename, radiances, locations, cloud_mask, labels = swath
        predictions = predict_tiles(model, radiances, device, FLAGS.dataset_bs)
        outputs = np.ones((FLAGS.nb_classes if FLAGS.raw_predictions else 1, *cloud_mask.squeeze().shape)) * -1
        for ix, loc in enumerate(locations):
            outputs[:, loc[0][0]:loc[0][1], loc[1][0]:loc[1][1]] = predictions[ix]

        # Remove unpredicted border region possibly caused by offset / valid convolutions
        outputs = outputs[:, FLAGS.valid_convolution_offset:-1 - FLAGS.valid_convolution_offset, FLAGS.valid_convolution_offset:-1 - FLAGS.valid_convolution_offset]
        labels = labels.squeeze()[FLAGS.valid_convolution_offset:-1 - FLAGS.valid_convolution_offset, FLAGS.valid_convolution_offset:-1 - FLAGS.valid_convolution_offset]
        cloud_mask = cloud_mask.squeeze()[FLAGS.valid_convolution_offset:-1 - FLAGS.valid_convolution_offset, FLAGS.valid_convolution_offset:-1 - FLAGS.valid_convolution_offset]

        if FLAGS.immediate_evaluation:
            pure_filename = filename.replace(FLAGS.d_path, '').replace('.nc', '')
            if not os.path.exists(os.path.join(FLAGS.output_path, pure_filename)):
                os.makedirs(os.path.join(FLAGS.output_path, pure_filename))
            filename = filename.replace(FLAGS.d_path, FLAGS.output_path + f'/{pure_filename}/').replace('.nc', '.npz')
            report, probabilities, cloudy_labels = evaluate_file(filename, outputs.copy(), labels.copy(), cloud_mask.copy(), label_names, mask_names)
            # --- Save intermediate report and merge probabilities and labels for total evaluation ---
            total_report += report
            with open(os.path.join(FLAGS.output_path, 'report.txt'), 'w') as f:
                f.write(total_report)
            total_labels = np.append(total_labels, cloudy_labels)
            if total_probabilities is None:
                total_probabilities = probabilities
            else:
                total_probabilities = np.hstack((total_probabilities, probabilities))
            outputs_to_figure_or_file(outputs, labels, cloud_mask, FLAGS.use_continuous_colors,
                                      FLAGS.cloud_mask_as_binary, FLAGS.to_file, filename)
        else:
            np.savez(os.path.join(FLAGS.output_path, filename.replace(FLAGS.d_path, '').replace('.nc', '')),
                     outputs=outputs, labels=labels, cloud_mask=cloud_mask)

    if FLAGS.immediate_evaluation:
        # --- Generate total evaluation and save final report ---
        print('Performing total evaluation...')
        total_report += '#### TOTAL ####\n\n'
        if not os.path.exists(os.path.join(FLAGS.output_path, 'total')):
            os.makedirs(os.path.join(FLAGS.output_path, 'total'))
        total_file = os.path.join(FLAGS.output_path, 'total/total.npz')
        report, matrix = evaluate_clouds(total_probabilities, total_labels, label_names, total_file, detailed=True)
        total_report += 'Cloud class eval:\n\n' + report + '\n\n'
        total_report += matrix
        with open(os.path.join(FLAGS.output_path, 'report.txt'), 'w') as f:
            f.write(total_report)


if __name__ == '__main__':
    FLAGS.read_flags_from_files(sys.argv[1:])
    app.run(main)
