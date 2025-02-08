import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset_omdena import NPY_datasets, NPY_datasets2
from tensorboardX import SummaryWriter
from models.vmunet.vmunet_omdena import VMUNet

from engine import *  # Ensure the engine.py file is correctly imported
import os
import sys

from utils import *
from configs.config_setting_omdena import setting_config

import warnings
warnings.filterwarnings("ignore")


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(os.path.join(config.work_dir, 'summary'))

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_dataset2 = NPY_datasets2(config.data_path, config, train=True)
    train_dataset_new = torch.utils.data.ConcatDataset([train_dataset, train_dataset2])

    train_loader = DataLoader(train_dataset_new, batch_size=32, shuffle=True, pin_memory=True, num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=config.num_workers, drop_last=True)

    print('#----------Preparing Model----------#')
    model_cfg = config.model_config
    if config.network == 'vmunet':
        model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
        )
        model.load_from()
    else:
        raise ValueError('Invalid network specified!')

    model = model.cuda()
    cal_params_flops(model, 256, logger)

    print('#----------Preparing loss, optimizer, scheduler----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Setting other params----------#')
    min_loss = float('inf')
    start_epoch = 1
    val_loss_history = []  # Store last 5 validation loss values

    if os.path.exists(resume_model):
        print('#----------Resuming from checkpoint----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        min_loss = checkpoint['min_loss']

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()
        step = train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, step, logger, config, writer)

        val_loss = val_one_epoch(val_loader, model, criterion, epoch, logger, config, scheduler)

        # Store the last 5 loss values
        val_loss_history.append(val_loss)
        if len(val_loss_history) > 5:
            val_loss_history.pop(0)  # Keep only the last 5 values

        # Early stopping: Stop if last 5 losses are increasing
        if len(val_loss_history) == 5 and all(val_loss_history[i] > val_loss_history[i - 1] for i in range(1, 5)):
            print(f"Early stopping triggered at epoch {epoch} (5 consecutive loss increases)")
            break

        # Save best model
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))

        # Save latest model
        torch.save({
            'epoch': epoch,
            'min_loss': min_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(checkpoint_dir, 'latest.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(os.path.join(checkpoint_dir, 'best.pth'), map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        test_loss = test_one_epoch(val_loader, model, criterion, logger, config)
        os.rename(os.path.join(checkpoint_dir, 'best.pth'), os.path.join(checkpoint_dir, f'best-epoch{start_epoch}-loss{min_loss:.4f}.pth'))


if __name__ == '__main__':
    config = setting_config
    main(config)
