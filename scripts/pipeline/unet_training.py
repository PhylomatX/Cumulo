import os
import time
import copy
import torch
import random
import glob
import numpy as np
from absl import app
import torch.nn as nn
from absl import flags
from tqdm import tqdm
import torch.optim as optim
from cumulo.data.loader import CumuloDataset
from cumulo.models.unet_weak import UNet_weak
from cumulo.models.unet_equi import UNet_equi
from cumulo.utils.utils import Normalizer, get_dataset_statistics

flags.DEFINE_string('d_path', None, help='Data path')
flags.DEFINE_string('m_path', None, help='Model path')
flags.DEFINE_integer('r_seed', 1, help='Random seed')
flags.DEFINE_integer('nb_epochs', 100, help='Number of epochs')
FLAGS = flags.FLAGS


def main(_):
    # Initialize parameters and prepare data
    nb_epochs = FLAGS.nb_epochs
    nb_classes = 9
    batch_size = 32
    lr = 0.001
    weight_decay = 0.0

    torch.manual_seed(FLAGS.r_seed)
    torch.cuda.manual_seed(FLAGS.r_seed)
    np.random.seed(FLAGS.r_seed)
    random.seed(FLAGS.r_seed)
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(FLAGS.m_path):
        os.makedirs(FLAGS.m_path)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print("using GPUs?", torch.cuda.is_available())

    try:
        class_weights = np.load(os.path.join(FLAGS.d_path, "class-weights.npy"))
        m = np.load(os.path.join(FLAGS.d_path, "mean.npy"))
        std = np.load(os.path.join(FLAGS.d_path, "std.npy"))
    except FileNotFoundError:
        print("Computing dataset mean, standard deviation and class ratios")
        dataset = CumuloDataset(FLAGS.d_path)
        weights, class_weights, m, std = get_dataset_statistics(dataset, nb_classes, tile_size=128)
        np.save(os.path.join(FLAGS.d_path, "class-weights.npy"), class_weights)
        np.save(os.path.join(FLAGS.d_path, "mean.npy"), m)
        np.save(os.path.join(FLAGS.d_path, "std.npy"), std)

    normalizer = Normalizer(m, std)
    class_weights = torch.from_numpy(class_weights).float()

    tile_num = len(glob.glob(os.path.join(FLAGS.d_path, "*.npz")))
    idx = np.arange(tile_num)
    np.random.shuffle(idx)
    # 10 % for validation, 20 % for testing, 70 % for training
    train_idx, val_idx = np.split(idx, [int(.9 * tile_num)])

    train_dataset = CumuloDataset(FLAGS.d_path, normalizer=normalizer, indices=train_idx)
    val_dataset = CumuloDataset(FLAGS.d_path, "npz", normalizer=normalizer, indices=val_idx)

    dataloaders = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8),
                   'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)}

    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    # Prepare model
    model = UNet_weak(in_channels=13, out_channels=nb_classes, starting_filters=32)
    print('Model initialized!')
    model = model.to(device)

    # Prepare training environment
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.9)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Start training
    metrics = train(model, FLAGS.m_path, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, num_epochs=nb_epochs, device=device)


def train(model, m_path, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=1000, device='cuda'):
    """
    Trains a model for all epochs using the provided dataloader.
    """
    t0 = time.time()

    best_acc = 0.0
    best_loss = None

    metrics = {'train_loss': [], 'train_acc': [], 'train_segacc': [],
               'val_loss': [], 'val_acc': [], 'val_segacc': []}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            i = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs.type(torch.float32))
                    mask = labels != -1
                    loss = 0
                    for ix in range(mask.shape[0]):
                        bmask = mask[ix]
                        blabels = labels[ix][bmask]
                        bmask = bmask.unsqueeze(0).expand_as(outputs[ix])
                        bouts = outputs[ix][bmask].reshape(outputs.shape[1], -1).transpose(0, 1)
                        loss += criterion(bouts, blabels.long())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                output = np.argmax(outputs.cpu().detach().numpy(), axis=1)
                labels = labels.cpu().detach().numpy()
                mask = mask.cpu().detach().numpy()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += np.sum(output[mask] == labels[mask])
                i += 1

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            if best_loss is None:
                best_loss = epoch_loss

            metrics[phase + '_loss'].append(epoch_loss)
            metrics[phase + '_acc'].append(float(epoch_acc))

            print('{} loss: {:.4f}, single pixel accuracy: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save({'model_state_dict': model.state_dict()}, os.path.join(m_path, f'val_best.pth'))

            if phase == 'train' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save({'model_state_dict': model.state_dict()}, os.path.join(m_path, f'train_best.pth'))

    time_elapsed = time.time() - t0
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val single pixel accuracy: {:4f}'.format(best_acc))

    return metrics


if __name__ == '__main__':
    app.run(main)
