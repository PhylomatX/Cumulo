import numpy as np
import torch
import os
from cumulo.data.loader import CumuloDataset
from cumulo.utils.utils import Normalizer, TileExtractor
from absl import app
from absl import flags

flags.DEFINE_string('m_path', None, help='Location of model')
flags.DEFINE_string('nc_path', None, help='Location of nc files')
flags.DEFINE_string('stat_path', None, help='Location of dataset statistics')
FLAGS = flags.FLAGS


def load_model(model_dir, use_cuda):
    model_path = os.path.join(model_dir, "model.pth")
    model = torch.load(model_path)["model"]
    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, range(torch.cuda.device_count()))
        torch.backends.cudnn.enabled = False
    model.eval()
    return model


def predict_tiles(model, tiles, use_cuda):
    inputs = torch.from_numpy(tiles).float()
    if use_cuda:
        inputs = inputs.cuda()
    outputs = model(inputs)
    predictions = torch.argmax(outputs, 1)
    return predictions.cpu().detach().numpy()


def main(_):
    m = np.load(os.path.join(FLAGS.stat_path, "mean.npy"))
    s = np.load(os.path.join(FLAGS.stat_path, "std.npy"))

    # dataset loader
    tile_extr = TileExtractor()
    normalizer = Normalizer(m, s)
    dataset = CumuloDataset(d_path=FLAGS.nc_path, ext="nc", label_preproc=None,
                            normalizer=normalizer, tiler=tile_extr)

    use_cuda = torch.cuda.is_available()
    print("using GPUs?", use_cuda)
    model = load_model(FLAGS.m_path, use_cuda)

    for swath in dataset:
        filename, tiles, locations, cloud_mask, labels = swath
        predictions = predict_tiles(model, tiles, use_cuda)


if __name__ == '__main__':
    app.run(main)
