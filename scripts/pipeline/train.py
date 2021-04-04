import os
import torch
import random
import glob
import pickle as pkl
from tqdm import tqdm
import numpy as np
from absl import app
import torch.nn as nn
import torch.optim as optim
import faulthandler
import shutil
from cumulo import __file__ as arch_src
from cumulo.data.loader import CumuloDataset
from cumulo.models.unet_weak import UNet_weak
from cumulo.models.unet_equi import UNet_equi
from cumulo.utils.training import GlobalNormalizer, LocalNormalizer
from flags import FLAGS


def main(_):
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

    class_weights = np.load(os.path.join(FLAGS.d_path, "class-weights.npy"))
    m = np.load(os.path.join(FLAGS.d_path, "mean.npy"))
    std = np.load(os.path.join(FLAGS.d_path, "std.npy"))

    custom_class_weights = np.ones_like(class_weights)
    custom_class_weights[3] *= 20
    # custom_class_weights[4] /= 10
    class_weights *= custom_class_weights

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
        train_idx, val_idx, test_idx = np.split(idx, [int(.92 * tile_num), int(.97 * tile_num)])
        np.save(os.path.join(FLAGS.m_path, 'train_idx.npy'), train_idx)
        np.save(os.path.join(FLAGS.m_path, 'val_idx.npy'), val_idx)
        np.save(os.path.join(FLAGS.m_path, 'test_idx.npy'), test_idx)

    train_dataset = CumuloDataset(FLAGS.d_path, normalizer=normalizer, indices=train_idx, batch_size=FLAGS.dataset_bs,
                                  tile_size=FLAGS.tile_size, rotation_probability=FLAGS.rotation_probability,
                                  valid_convolution_offset=FLAGS.valid_convolution_offset, most_frequent_clouds_as_GT=FLAGS.most_frequent_clouds_as_GT)

    if FLAGS.val:
        print("Training with validation!")
        val_dataset = CumuloDataset(FLAGS.d_path, normalizer=normalizer, indices=val_idx, batch_size=FLAGS.dataset_bs,
                                    tile_size=FLAGS.tile_size, rotation_probability=FLAGS.rotation_probability,
                                    valid_convolution_offset=FLAGS.valid_convolution_offset,
                                    most_frequent_clouds_as_GT=FLAGS.most_frequent_clouds_as_GT)
        datasets = {'train': train_dataset, 'val': val_dataset}
    else:
        print("Training without validation!")
        datasets = {'train': train_dataset}

    # --- prepare model ---
    if FLAGS.model == 'weak':
        model = UNet_weak(in_channels=13, out_channels=FLAGS.nb_classes, starting_filters=32, padding=FLAGS.padding, norm=FLAGS.norm)
    elif FLAGS.model == 'equi':
        model = UNet_equi(in_channels=13, out_channels=FLAGS.nb_classes, starting_filters=32, padding=FLAGS.padding, norm=FLAGS.norm, rot=FLAGS.rot)
    else:
        raise NotImplementedError()
    print('Model initialized!')
    model = model.to(device)

    # Prepare training environment
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # lr_sched = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)

    # Begin with a very small lr and double it every 100 steps.
    # for grp in optimizer.param_groups:
    #     grp['lr'] = 1e-7
    # lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, 100, 2)

    # base_lr and max_lr were found with the experimental procedure from the CyclicLR paper (see above)
    if FLAGS.model == 'weak':
        lr_sched = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=1e-6,
            max_lr=1e-4,
            cycle_momentum=True if 'momentum' in optimizer.defaults else False
        )
    elif FLAGS.model == 'equi':
        lr_sched = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=1.2e-5,
            max_lr=2e-4,
            cycle_momentum=True if 'momentum' in optimizer.defaults else False
        )

    bce = nn.BCEWithLogitsLoss()
    class_loss = nn.CrossEntropyLoss(weight=class_weights.to(device))
    auto_loss = nn.MSELoss()

    # backup training script and src folder
    shutil.copyfile(__file__, FLAGS.m_path + '/0-' + os.path.basename(__file__))
    os.chmod(FLAGS.m_path + '/0-' + os.path.basename(__file__), 0o755)
    pkg_path = os.path.dirname(arch_src)
    backup_path = os.path.join(FLAGS.m_path, 'src_backup')
    shutil.make_archive(backup_path, 'gztar', pkg_path)

    # Start training
    train(model, FLAGS.m_path, datasets, bce, class_loss, auto_loss, optimizer, lr_sched, num_epochs=nb_epochs, device=device)


def train(model, m_path, datasets, bce_fn, class_loss_fn, auto_loss_fn, optimizer, scheduler, num_epochs=1000, device='cuda'):
    best_acc = 0.0
    best_loss = None
    offset = FLAGS.valid_convolution_offset

    with open(os.path.join(FLAGS.m_path, 'flagfile.txt'), 'w') as f:
        f.writelines(FLAGS.flags_into_string())

    metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        dataloaders = {}
        for phase in datasets:
            dataloaders[phase] = torch.utils.data.DataLoader(datasets[phase], shuffle=True, batch_size=FLAGS.bs,
                                                             num_workers=FLAGS.num_workers)

        torch.manual_seed(epoch)
        torch.cuda.manual_seed(epoch)
        np.random.seed(epoch)
        random.seed(epoch)
        torch.backends.cudnn.deterministic = True

        for phase in dataloaders:
            if phase == 'train':
                model.train()
            else:
                if epoch % FLAGS.analysis_freq != 0:
                    continue
                model.eval()

            running_loss = 0.0
            running_accuracy = 0.0

            for sample_ix, sample in enumerate(tqdm(dataloaders[phase])):
                radiances, labels, cloud_mask = sample

                # --- unpack PyTorch batching if dataloader delivers batches already
                if FLAGS.merged:
                    radiances = radiances.reshape(-1, *tuple(radiances.shape[2:]))
                    labels = labels.reshape(-1, *tuple(radiances.shape[2:]))
                    cloud_mask = cloud_mask.reshape(-1, *tuple(radiances.shape[2:]))

                radiances = radiances.to(device)
                labels = labels.to(device)
                cloud_mask = cloud_mask.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(radiances.type(torch.float32))
                    if offset > 0:
                        labels = labels[:, offset:-offset, offset:-offset]
                        cloud_mask = cloud_mask[:, offset:-offset, offset:-offset]
                        radiances = radiances[..., offset:-offset, offset:-offset]
                    labeled_mask = labels >= 0  # get labeled pixels (this includes labels in non-cloudy regions as well!)
                    loss = 0
                    for ix in range(labeled_mask.shape[0]):
                        b_labeled_mask = labeled_mask[ix]
                        b_labels = labels[ix][b_labeled_mask]
                        b_labeled_mask = b_labeled_mask.unsqueeze(0).expand_as(outputs[ix][1:9])
                        b_cloud_mask = cloud_mask[ix]
                        b_outputs = outputs[ix][1:9][b_labeled_mask].reshape(8, -1).transpose(0, 1)

                        mask_loss = FLAGS.mask_weight * bce_fn(outputs[ix][0], b_cloud_mask.float())  # BCEWithLogitsLoss for cloud mask
                        class_loss = FLAGS.class_weight * class_loss_fn(b_outputs, b_labels.long())  # CrossEntropy for labels

                        specific_class_weight = 5
                        specific_class_labels = b_labels.clone()
                        specific_class_labels[b_labels != 3] = 0
                        specific_class_labels[b_labels == 3] = 1
                        specific_class_loss = specific_class_weight * bce_fn(outputs[ix][9][b_labeled_mask[0]], specific_class_labels.float())

                        auto_loss, auto_weight = 0, 0
                        if FLAGS.auto_weight > 0:
                            auto_loss = FLAGS.auto_weight * auto_loss_fn(outputs[ix][9:].float(), radiances[ix][:(FLAGS.nb_classes - 9)].float())  # MSE for autoencoder loss
                            auto_weight = FLAGS.auto_weight
                        loss += (mask_loss + class_loss + auto_loss + specific_class_loss) / (FLAGS.mask_weight + FLAGS.class_weight + auto_weight + specific_class_weight)
                    loss /= labeled_mask.shape[0]

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                outputs = outputs.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                cloud_mask = cloud_mask.cpu().detach().numpy()

                # --- save training examples ---
                if FLAGS.examples and epoch % FLAGS.analysis_freq == 0:
                    if sample_ix == 0:
                        np.savez(os.path.join(FLAGS.m_path, f'examples/{epoch}_{phase}'),
                                 labels=labels[0], outputs=outputs[0, ...],
                                 cloud_mask=cloud_mask[0])

                labeled_mask = labels != -1
                mask_prediction = outputs[:, 0, ...]
                mask_prediction[mask_prediction < 0.5] = 0
                mask_prediction[mask_prediction >= 0.5] = 1
                class_prediction = np.argmax(outputs[:, 1:9, ...], axis=1)
                specific_class_prediction = outputs[:, 9, ...]
                specific_class_prediction[specific_class_prediction < 0.5] = 0
                specific_class_prediction[specific_class_prediction >= 0.5] = 1
                specific_class_labels = labels.copy()
                specific_class_labels[specific_class_labels != 3] = 0
                specific_class_labels[specific_class_labels == 3] = 1

                # statistics
                running_loss += loss.item()
                mask_accuracy = float(np.sum(mask_prediction.reshape(-1) == cloud_mask.reshape(-1)) / cloud_mask.reshape(-1).shape[0])
                class_accuracy = float(np.sum(class_prediction[labeled_mask] == labels[labeled_mask]) / labels[labeled_mask].shape[0])
                specific_class_accuracy = float(np.sum(specific_class_prediction[labeled_mask] == specific_class_labels[labeled_mask]) / specific_class_labels[labeled_mask].shape[0])
                accuracy = (FLAGS.mask_weight * mask_accuracy + FLAGS.class_weight * class_accuracy) / (FLAGS.mask_weight + FLAGS.class_weight)
                running_accuracy += accuracy

                print(f"Epoch: {epoch} "
                      f"- Loss: {round(running_loss / (sample_ix + 1), 3)} "
                      f"- Class accuracy: {round(class_accuracy, 3)} "
                      f"- Mask accuracy: {round(mask_accuracy, 3)} "
                      f"- Specific class accuracy: {round(specific_class_accuracy, 3)} - {FLAGS.m_path}")

                with open(os.path.join(FLAGS.m_path, 'metrics.pkl'), 'wb') as f:
                    pkl.dump(metrics, f)

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_accuracy / len(datasets[phase])
            metrics[phase + '_loss'].append(epoch_loss)
            metrics[phase + '_acc'].append(float(epoch_acc))

            # --- save models ---
            if best_loss is None:
                best_loss = epoch_loss
            if phase == 'val' and best_acc < epoch_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), os.path.join(m_path, f'val_best'))
            if phase == 'train' and epoch_loss < best_loss:
                best_loss = epoch_loss
                model.eval()  # required by e2cnn
                torch.save(model.state_dict(), os.path.join(m_path, f'train_best'))
                model.train()
            if phase == 'train':
                model.eval()  # required by e2cnn
                torch.save(model.state_dict(), os.path.join(m_path, f'last_model'))
                model.train()

        with open(os.path.join(FLAGS.m_path, 'metrics.pkl'), 'wb') as f:
            pkl.dump(metrics, f)


if __name__ == '__main__':
    app.run(main)
