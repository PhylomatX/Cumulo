import os
import time
import torch
import random
import glob
import pickle as pkl
from tqdm import tqdm
import numpy as np
from absl import app
import torch.nn as nn
from absl import flags
import torch.optim as optim
import faulthandler
import shutil
from cumulo import __file__ as arch_src
from cumulo.data.loader import CumuloDataset
from cumulo.models.unet_weak import UNet_weak
from cumulo.models.unet_equi import UNet_equi
from cumulo.utils.utils import GlobalNormalizer, LocalNormalizer, include_cloud_mask

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
flags.DEFINE_integer('nb_classes', 9, help='Number of classes.')
flags.DEFINE_integer('center_distance', None, help='Distance between base points of tile extraction.')
flags.DEFINE_bool('val', False, help='Flag for validation after each epoch.')
flags.DEFINE_string('model', 'weak', help='Option for choosing between UNets.')
flags.DEFINE_bool('merged', False, help='Flag for indicating use of merged dataset')
flags.DEFINE_bool('examples', False, help='Save some training examples in each epoch')
flags.DEFINE_bool('local_norm', False, help='Standardize each image channel-wise. If False the statistics of a data subset will be used.')
flags.DEFINE_integer('examples_num', None, help='How many samples should get saved as example?')
flags.DEFINE_integer('analysis_freq', 1, help='Validation and example save frequency')
flags.DEFINE_integer('rot', 2, help='Number of elements in rotation group')
flags.DEFINE_integer('offset', 0, help='Cropping offset for labels in case of valid convolutions')
flags.DEFINE_integer('padding', 0, help='Padding for convolutions')
flags.DEFINE_float('augment_prob', 0, help='Augmentation probability')
FLAGS = flags.FLAGS


def main(_):
    # Initialize parameters and prepare data
    nb_epochs = FLAGS.nb_epochs
    lr = 1e-3
    weight_decay = 0.5e-4

    torch.manual_seed(FLAGS.r_seed)
    torch.cuda.manual_seed(FLAGS.r_seed)
    np.random.seed(FLAGS.r_seed)
    random.seed(FLAGS.r_seed)
    torch.backends.cudnn.deterministic = True
    faulthandler.enable()

    if not os.path.exists(FLAGS.m_path):
        os.makedirs(FLAGS.m_path)
        os.makedirs(os.path.join(FLAGS.m_path, 'examples'))

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print("using GPUs?", torch.cuda.is_available())

    try:
        class_weights = np.load(os.path.join(FLAGS.d_path, "class-weights.npy"))
        m = np.load(os.path.join(FLAGS.d_path, "mean.npy"))
        std = np.load(os.path.join(FLAGS.d_path, "std.npy"))
    except FileNotFoundError:
        print("Statistics were not found!")

    if FLAGS.local_norm:
        print("Using local normalization.")
        normalizer = LocalNormalizer()
    else:
        print("Using global normalization.")
        normalizer = GlobalNormalizer(m, std)
    class_weights = torch.from_numpy(class_weights).float()

    if FLAGS.tile_num is None:
        tile_num = len(glob.glob(os.path.join(FLAGS.d_path, "*." + FLAGS.filetype)))
    else:
        tile_num = FLAGS.tile_num
    idx = np.arange(tile_num)
    np.random.shuffle(idx)

    try:
        train_idx = np.load(os.path.join(FLAGS.m_path, 'train_idx.npy'))
        val_idx = np.load(os.path.join(FLAGS.m_path, 'val_idx.npy'))
    except FileNotFoundError:
        train_idx, val_idx, test_idx = np.split(idx, [int(.85 * tile_num), int(.9 * tile_num)])
        np.save(os.path.join(FLAGS.m_path, 'train_idx.npy'), train_idx)
        np.save(os.path.join(FLAGS.m_path, 'val_idx.npy'), val_idx)
        np.save(os.path.join(FLAGS.m_path, 'test_idx.npy'), test_idx)

    train_dataset = CumuloDataset(FLAGS.d_path, normalizer=normalizer, indices=train_idx, batch_size=FLAGS.dataset_bs,
                                  tile_size=FLAGS.tile_size, center_distance=FLAGS.center_distance, ext=FLAGS.filetype,
                                  augment_prob=FLAGS.augment_prob, offset=FLAGS.offset)

    if FLAGS.val:
        print("Training with validation!")
        val_dataset = CumuloDataset(FLAGS.d_path, normalizer=normalizer, indices=val_idx, batch_size=FLAGS.dataset_bs,
                                    tile_size=FLAGS.tile_size, center_distance=FLAGS.center_distance, ext=FLAGS.filetype,
                                    augment_prob=FLAGS.augment_prob, offset=FLAGS.offset)
        datasets = {'train': train_dataset, 'val': val_dataset}
    else:
        print("Training without validation!")
        datasets = {'train': train_dataset}

    # Prepare model
    if FLAGS.model == 'weak':
        model = UNet_weak(in_channels=13, out_channels=FLAGS.nb_classes, starting_filters=32, padding=FLAGS.padding)
    elif FLAGS.model == 'equi':
        model = UNet_equi(in_channels=13, out_channels=FLAGS.nb_classes, starting_filters=32, rot=FLAGS.rot)
    print('Model initialized!')
    model = model.to(device)

    # Prepare training environment
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # lr_sched = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)

    # Begin with a very small lr and double it every 100 steps.
    # for grp in optimizer.param_groups:
    #     grp['lr'] = 1e-7
    # lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, 100, 2)

    lr_sched = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1.2e-5,  # 1e-6 for weak, 1.2e-5 for equi
        max_lr=2e-4,  # 1e-4 for weak, 2e-4 for equi
        cycle_momentum=True if 'momentum' in optimizer.defaults else False
    )

    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # backup training script and src folder
    shutil.copyfile(__file__, FLAGS.m_path + '/0-' + os.path.basename(__file__))
    os.chmod(FLAGS.m_path + '/0-' + os.path.basename(__file__), 0o755)
    pkg_path = os.path.dirname(arch_src)
    backup_path = os.path.join(FLAGS.m_path, 'src_backup')
    shutil.make_archive(backup_path, 'gztar', pkg_path)

    # Start training
    train(model, FLAGS.m_path, datasets, criterion1, criterion2, optimizer, lr_sched, num_epochs=nb_epochs, device=device)


def train(model, m_path, datasets, criterion1, criterion2, optimizer, scheduler, num_epochs=1000, device='cuda'):
    """
    Trains a model for all epochs using the provided dataloader.
    """
    t0 = time.time()

    best_acc = 0.0
    best_loss = None
    offset = FLAGS.offset

    with open(os.path.join(FLAGS.m_path, 'flagfile.txt'), 'w') as f:
        f.writelines(FLAGS.flags_into_string())

    training_info = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
                     'train_running_accuracy': [], 'running_lr': []}

    for epoch in range(num_epochs):
        dataloaders = {}
        for phase in datasets:
            dataloaders[phase] = torch.utils.data.DataLoader(datasets[phase], shuffle=True, batch_size=FLAGS.bs,
                                                             num_workers=FLAGS.num_workers)

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        torch.manual_seed(epoch)
        torch.cuda.manual_seed(epoch)
        np.random.seed(epoch)
        random.seed(epoch)
        torch.backends.cudnn.deterministic = True

        # Each epoch has a training and validation phase
        for phase in dataloaders:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                if epoch % FLAGS.analysis_freq != 0:
                    continue
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_accuracy = 0

            # Iterate over data.
            for sample_ix, sample in enumerate(tqdm(dataloaders[phase])):
                inputs, labels, cloud_mask = sample

                if FLAGS.merged:
                    inputs = inputs.reshape(-1, *tuple(inputs.shape[2:]))
                    labels = labels.reshape(-1, *tuple(inputs.shape[2:]))
                    cloud_mask = cloud_mask.reshape(-1, *tuple(inputs.shape[2:]))

                inputs = inputs.to(device)
                labels = labels.to(device)
                cloud_mask = cloud_mask.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs.type(torch.float32))
                    if offset > 0:
                        labels = labels[:, offset:-offset, offset:-offset]
                        cloud_mask = cloud_mask[:, offset:-offset, offset:-offset]
                    mask = labels >= 0  # get labeled pixels (this includes labels in non-cloudy regions as well!)
                    loss = 0
                    for ix in range(mask.shape[0]):
                        bmask = mask[ix]
                        blabels = labels[ix][bmask]
                        cmask = cloud_mask[ix]
                        bmask = bmask.unsqueeze(0).expand_as(outputs[ix][1:])
                        bouts = outputs[ix][1:][bmask].reshape(outputs.shape[1]-1, -1).transpose(0, 1)
                        loss += (criterion1(outputs[ix][0], cmask) + 2 * criterion2(bouts, blabels.long())) / 3  # BCEWithLogitsLoss for cloud mask, CE for labels
                    loss /= mask.shape[0]

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                outputs = outputs.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                inputs = inputs.cpu().detach().numpy()
                cloud_mask = cloud_mask.cpu().detach().numpy()
                cloud_mask_pred = outputs[:, 0, ...]
                cloud_mask_pred[cloud_mask_pred < 0.5] = 0
                cloud_mask_pred[cloud_mask_pred >= 0.5] = 1
                cloud_class_pred = np.argmax(outputs[:, 1:, ...], axis=1)
                output = include_cloud_mask(cloud_class_pred, cloud_mask_pred)
                merged = include_cloud_mask(labels, cloud_mask)
                mask = merged > 0

                # statistics
                running_loss += loss.item()
                print(f"Epoch: {epoch} - Loss: {running_loss / (sample_ix + 1)}")
                accuracy = float(np.sum(output[mask] == merged[mask]) / output[mask].shape)  # use only labeled pixels in cloudy regions for accuracies
                running_accuracy += accuracy

                if phase == 'train' and epoch < 2:
                    training_info[phase + '_running_accuracy'].append(accuracy)
                    training_info['running_lr'].append(scheduler.get_lr())
                with open(os.path.join(FLAGS.m_path, 'metrics.pkl'), 'wb') as f:
                    pkl.dump(training_info, f)

                # save training examples
                if FLAGS.examples and epoch % FLAGS.analysis_freq == 0:
                    if sample_ix == 0:
                        examples_num = FLAGS.examples_num
                        if examples_num is None:
                            examples_num = inputs.shape[0]
                        np.savez(os.path.join(FLAGS.m_path, f'examples/{epoch}_{phase}'), inputs=inputs[:examples_num],
                                 labels=labels[:examples_num], outputs=output[:examples_num], cloud_mask=cloud_mask[:examples_num])

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_accuracy / len(datasets[phase])

            if best_loss is None:
                best_loss = epoch_loss

            training_info[phase + '_loss'].append(epoch_loss)
            training_info[phase + '_acc'].append(float(epoch_acc))

            print('{} loss: {:.4f}, single pixel accuracy: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # save models
            if phase == 'val' and best_acc < epoch_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), os.path.join(m_path, f'val_best'))
                torch.save(model.state_dict(), os.path.join(m_path, f'val_best_backup'))
            elif phase == 'train' and epoch_loss < best_loss:
                best_loss = epoch_loss
                model.eval()
                torch.save(model.state_dict(), os.path.join(m_path, f'train_best'))
                torch.save(model.state_dict(), os.path.join(m_path, f'train_best_backup'))
                model.train()
            if phase == 'train':
                model.eval()
                torch.save(model.state_dict(), os.path.join(m_path, f'last_model'))
                torch.save(model.state_dict(), os.path.join(m_path, f'last_model_backup'))
                model.train()

        with open(os.path.join(FLAGS.m_path, 'metrics.pkl'), 'wb') as f:
            pkl.dump(training_info, f)

    time_elapsed = time.time() - t0
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val single pixel accuracy: {:4f}'.format(best_acc))


if __name__ == '__main__':
    app.run(main)
