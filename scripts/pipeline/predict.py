import numpy as np
import torch
import math
import os
import sys
import pickle as pkl
from tqdm import tqdm
from cumulo.data.loader import CumuloDataset
from cumulo.utils.visualization import outputs_to_figure_or_file
from cumulo.utils.training import GlobalNormalizer, LocalNormalizer
from cumulo.utils.evaluation import evaluate_file, evaluate_clouds, create_class_histograms
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
            # fill up last batch with tiles from the beginning to avoid artefacts due to zero tiles
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
    if not os.path.exists(os.path.join(FLAGS.output_path, 'total')):
        os.makedirs(os.path.join(FLAGS.output_path, 'total'))

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
    dataset = CumuloDataset(d_path=FLAGS.d_path, normalizer=normalizer, indices=test_idx, prediction_mode=True,
                            tile_size=FLAGS.tile_size, valid_convolution_offset=FLAGS.valid_convolution_offset,
                            most_frequent_clouds_as_GT=FLAGS.most_frequent_clouds_as_GT)
    print(f"Predicting {len(dataset)} files.")

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    model = load_model(FLAGS.m_path)
    model.to(device)

    no_cloud_mask = False
    if FLAGS.mask_weight == 0:
        no_cloud_mask = True

    if FLAGS.immediate_evaluation:
        total_report = ''
        mask_names = [0, 1]  # cloud mask targets ('no cloud', 'cloud')
        label_names = list(range(8))  # cloud class targets (8 different cloud types)
        total_labels = np.array([])
        total_probabilities = None
        total_outputs = None

    swath_ix = 0
    used = 0
    prediction_number = len(dataset)
    if FLAGS.prediction_number is not None:
        prediction_number = FLAGS.prediction_number
    while used < prediction_number:
        swath = dataset[swath_ix]
        filename, radiances, locations, cloud_mask, labels = swath
        if np.all(labels == -1):
            swath_ix += 1
            continue
        else:
            print(used)
            used += 1
            swath_ix += 1
        # --- Mix tiles for potential improvement in BatchNorm ---
        random_permutation = torch.randperm(radiances.shape[0])
        radiances = radiances[random_permutation]
        locations = locations[random_permutation]

        # --- Generate tile predictions and insert them into the swath ---
        predictions = predict_tiles(model, radiances, device, FLAGS.dataset_bs)
        outputs = np.ones((FLAGS.nb_classes, *cloud_mask.squeeze().shape)) * -1
        for ix, loc in enumerate(locations):
            outputs[:, loc[0][0]:loc[0][1], loc[1][0]:loc[1][1]] = predictions[ix]

        # --- Remove unpredicted border region possibly caused by offset / valid convolutions ---
        outputs = outputs[:, FLAGS.valid_convolution_offset:-1 - FLAGS.valid_convolution_offset, FLAGS.valid_convolution_offset:-1 - FLAGS.valid_convolution_offset]
        labels = labels.squeeze()[FLAGS.valid_convolution_offset:-1 - FLAGS.valid_convolution_offset, FLAGS.valid_convolution_offset:-1 - FLAGS.valid_convolution_offset]
        cloud_mask = cloud_mask.squeeze()[FLAGS.valid_convolution_offset:-1 - FLAGS.valid_convolution_offset, FLAGS.valid_convolution_offset:-1 - FLAGS.valid_convolution_offset]

        outputs[3] = outputs[3] + 2

        if FLAGS.immediate_evaluation:
            filename = filename.replace(FLAGS.d_path, FLAGS.output_path + f'/').replace('.nc', '.npz')
            report, probabilities, cloudy_labels, eval_outputs = evaluate_file(filename, outputs.copy(), labels.copy(), cloud_mask.copy(), label_names, mask_names, no_cloud_mask)
            # --- Save intermediate report and merge probabilities and labels for total evaluation ---
            total_report += report
            with open(os.path.join(FLAGS.output_path, 'total/total_report.txt'), 'w') as f:
                f.write(total_report)
            total_labels = np.append(total_labels, cloudy_labels)
            if total_probabilities is None:
                total_probabilities = probabilities
            else:
                total_probabilities = np.hstack((total_probabilities, probabilities))
            if total_outputs is None:
                total_outputs = eval_outputs
            else:
                total_outputs = np.hstack((total_outputs, eval_outputs))
            outputs_to_figure_or_file(outputs, labels, cloud_mask, cloud_mask_as_binary=FLAGS.cloud_mask_as_binary,
                                      to_file=FLAGS.to_file, npz_file=filename, no_cloud_mask_prediction=no_cloud_mask)
        else:
            np.savez(os.path.join(FLAGS.output_path, filename.replace(FLAGS.d_path, '').replace('.nc', '')),
                     outputs=outputs, labels=labels, cloud_mask=cloud_mask)

    if FLAGS.immediate_evaluation:
        # --- Generate total evaluation and save final report ---
        with open(os.path.join(FLAGS.output_path, 'total/total_outputs.pkl'), 'wb') as f:
            pkl.dump(total_outputs, f)
        with open(os.path.join(FLAGS.output_path, 'total/total_labels.pkl'), 'wb') as f:
            pkl.dump(total_labels, f)
        create_class_histograms(total_outputs, total_labels, os.path.join(FLAGS.output_path, 'total/'))

        print('Performing total evaluation...')
        total_report += '#### TOTAL ####\n\n'
        total_file = os.path.join(FLAGS.output_path, 'total/total.npz')
        report, matrix = evaluate_clouds(total_probabilities, total_labels, label_names, total_file, detailed=True)
        total_report += 'Cloud class eval:\n\n' + report + '\n\n'
        total_report += matrix
        with open(os.path.join(FLAGS.output_path, 'total/total_report.txt'), 'w') as f:
            f.write(total_report)


if __name__ == '__main__':
    FLAGS.read_flags_from_files(sys.argv[1:])
    app.run(main)
