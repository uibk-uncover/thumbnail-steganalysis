
import argparse
import json
import numpy as np
import pathlib
import shutil
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary


from training.model import get_b0, seed_everything
from evaluate import evaluate_model
from training.metrics import AverageMeter, AccuracyMeter, PEMeter, PMD5FPMeter, ProgressMeter
from training.log import setup_custom_logger, create_run_name
from training.data.loader import get_data_loader

log = setup_custom_logger(pathlib.Path(__file__).name)


def train_one_epoch(
    tr_loader,
    model,
    criterion,
    optimizer,
    epoch,
    writer,
    device,
    args
):
    """"""
    # Create meters
    loss_meter = AverageMeter('Loss', ':.4e')
    acc_meter = AccuracyMeter('Acc', ':4.3f')
    pe_meter = PEMeter('P_E', ':4.3f')
    pmd5fp_meter = PMD5FPMeter('P_MD^5%%FP', ':4.3f')
    progress = ProgressMeter(
        len(tr_loader),
        [loss_meter, acc_meter, pe_meter, pmd5fp_meter],
        prefix='Epoch: [{}]'.format(epoch),
    )

    # switch to train mode
    model.train()

    for i, (images, targets) in enumerate(tr_loader):
        batch_size = images.size(0)

        # Move data to the same device as model
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Compute output
        output = model(images)
        loss = criterion(output, targets)

        # Skip softmax activation as we are only interested in the argmax
        pred = output.data
        _, y_pred = torch.max(pred, dim=1)

        # Record performance
        loss_meter.update(loss.item(), batch_size)
        pe_meter.update(
            targets.cpu().numpy(),
            pred[:, 1].detach().cpu().numpy(),
        )
        pmd5fp_meter.update(
            targets.cpu().numpy(),
            pred[:, 1].detach().cpu().numpy(),
        )
        acc_meter.update(
            targets.cpu().numpy(),
            y_pred.detach().cpu().numpy(),
        )

        # Compute gradients
        loss.backward()

        # Gradient descend
        optimizer.step()

        if i % args['print_freq'] == 0:
            log.info(progress.to_str(batch=i + 1))

    writer.add_scalar('train/' + loss_meter.name, loss_meter.avg, global_step=epoch)
    writer.add_scalar('train/' + pe_meter.name, pe_meter.avg, global_step=epoch)
    writer.add_scalar('train/' + acc_meter.name, acc_meter.avg, global_step=epoch)


def validate(va_loader, model, criterion, writer, device, epoch=None):
    """"""
    # Create meters
    loss_meter = AverageMeter('Loss', ':.4e')
    acc_meter = AccuracyMeter('Acc', ':4.3f')
    pe_meter = PEMeter('P_E', ':4.3f')
    pmd5fp_meter = PMD5FPMeter('P_MD^5%%FP', ':4.3f')
    progress = ProgressMeter(
        len(va_loader),
        [loss_meter, acc_meter, pe_meter, pmd5fp_meter],
        prefix='Val: ',
    )

    # Evaluate model
    evaluate_model(
        loader=va_loader,
        model=model,
        criterion=criterion,
        device=device,
        loss_meter=loss_meter,
        target_meters=(acc_meter,),
        score_meters=(pe_meter, pmd5fp_meter,),
    )

    # Log progress
    log.info(progress.to_str(batch=0))

    # Write score
    writer.add_scalar('val/' + loss_meter.name, loss_meter.avg, global_step=epoch)
    writer.add_scalar('val/' + pe_meter.name, pe_meter.avg, global_step=epoch)
    writer.add_scalar('val/' + acc_meter.name, acc_meter.avg, global_step=epoch)

    return loss_meter.avg


def train(args):
    # Set up directory name for output directory
    # experiment_dir_name = time.strftime('%Y_%m_%d_%H_%M_%S') + '-prototyping'
    experiment_dir_name = time.strftime('%y%m%d%H%M%S') + '-' + create_run_name(args)
    if args['experiment_dir_suffix']:
        experiment_dir_name = experiment_dir_name + '_' + args['experiment_dir_suffix']
    print(f'{experiment_dir_name=}')

    # Create output directory for this experiment
    experiment_dir = pathlib.Path(args['output_dir']) / args['stego_method'] / experiment_dir_name
    if not experiment_dir.exists():
        experiment_dir.mkdir(exist_ok=False, parents=True)

    # Create resume directory
    if args['resume_dir']:
        resume_dir = pathlib.Path(args['resume_dir'])
    else:
        resume_dir = pathlib.Path(args['output_dir'])
    resume_dir = resume_dir / args['stego_method']

    # Dump args to file
    args_file = experiment_dir / 'config.json'
    with open(args_file, 'w') as f:
        json.dump(args, f, indent=4, sort_keys=True)

    # Set up model and log directories
    # Concatenate path to one subdirectory for logging
    log_dir = experiment_dir / 'log'
    model_dir = experiment_dir / 'model'
    best_model_file = model_dir / 'best_model.pt.tar'
    latest_model_file = model_dir / 'latest_model.pt.tar'
    # Create subdirectories if they don't exist yet
    log_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(exist_ok=False)

    # Summary writer
    writer = SummaryWriter(log_dir=log_dir)

    # Decide whether to run on GPU or CPU
    if torch.cuda.is_available():
        log.info('Using GPU')
        device = torch.device('cuda')
    else:
        log.info('Using CPU, this will be slow')
        device = torch.device('cpu')

    # Seed if requested
    if args['seed']:
        log.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably! You may see unexpected behavior when restarting from checkpoints.')
        seed_everything(args['seed'])

    # Input channels
    in_chans = 2 if args['thumbnail'] else 1

    # Data loaders
    args['thumbnail_stego'] = False  # only allowed for evaluation
    args_val = args.copy()
    args_val['post_flip'] = args_val['post_rotate'] = False
    args_val['pre_rotate'] = True  # evaluate with pre-rotation
    tr_loader, tr_dataset = get_data_loader(args['tr_csv'], args, in_chans)
    va_loader, va_dataset = get_data_loader(args['va_csv'], args_val, in_chans)

    # Set up model
    model = get_b0(
        in_chans=in_chans,
        shape=args['shape'],
        device=device,
        drop_rate=args['drop_rate'],
    )
    summary(
        model,
        input_size=(args["batch_size"], in_chans, *args["shape"])
    )

    # Set up loss and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), args["learning_rate"])
    scheduler = None

    start_epoch = 0
    best_val_loss = np.inf
    patience = args['patience']

    if args['resume']:
        resume_model_file = resume_dir / args['resume'] / 'model' / 'best_model.pt.tar'
        print(resume_model_file)
        if resume_model_file.exists():
            log.info("=> loading checkpoint '{}'".format(args["resume"]))

            checkpoint = torch.load(resume_model_file, map_location=device)

            # start_epoch = checkpoint['epoch']
            # best_val_loss = checkpoint['best_val_loss']

            model.load_state_dict(checkpoint['state_dict'])
            log.info("=> loaded checkpoint '{}'".format(args["resume"]))
        else:
            raise Exception("no checkpoint found at '{}'".format(args["resume"]))

    # Training loop
    for epoch in range(start_epoch, args["num_epochs"]):
        # Reshuffle training dataset
        tr_dataset.reshuffle()
        # va_dataset.reshuffle()  # TO REMOVE

        # Train for one epoch
        train_one_epoch(
            tr_loader=tr_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            writer=writer,
            device=device,
            args=args
        )

        val_loss = validate(
            va_loader=va_loader,
            model=model,
            criterion=criterion,
            writer=writer,
            device=device,
            epoch=epoch
        )

        if scheduler:
            scheduler.step(val_loss)

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            'patience': patience,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
        }, latest_model_file)

        # Remember best validation loss
        is_best = val_loss < best_val_loss
        if is_best:
            patience = args['patience']
            shutil.copyfile(latest_model_file, best_model_file)
            print('best model!', val_loss, 'is better than', best_val_loss)
            best_val_loss = val_loss
        else:
            patience -= 1
            print('patience countdown:', patience)

        # Early stopping
        if patience <= 0:
            print('my patience is over, early stopping!')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--dataset', type=str, help='Path to image root', default='data/dataset')
    parser.add_argument('--tr_csv', type=str, help='Path to csv file containing the training images', default='split_tr.csv')
    parser.add_argument('--va_csv', type=str, help='Path to csv file containing the validation images', default='split_va.csv')
    # Data: Covers
    parser.add_argument('--shape', nargs='+', type=int, default=[512, 512], help='Dataset shape')
    parser.add_argument('--quality', type=int, default=75, help='Selected specific JPEG quality')
    # Data: Stegos
    parser.add_argument('--stego_method', type=str, default='UERD', help='Selected stego method')
    parser.add_argument('--alpha', type=float, default=.4, help='Selected embedding rate (alpha)')
    parser.add_argument('--rotation', type=int, help='Dataset rotation to use')
    # Data: Thumbnail
    parser.add_argument('--thumbnail', action='store_true', help='Use thumbnail side-information')
    parser.add_argument('--thumbnail_shape', nargs='+', type=int, default=[128, 128], help='Thumbnail shape')
    parser.add_argument('--thumbnail_implementation', type=str, default='textbook', help='Thumbnail implementation')
    parser.add_argument('--thumbnail_kernel', type=str, default='nearest', help='Thumbnail interpolation kernel')
    parser.add_argument('--thumbnail_antialiasing', action='store_true', help='Generating thumbnail without/with antialiasing')
    parser.add_argument('--thumbnail_quality', type=int, default=75, help='Selected specific JPEG thumbnail quality')
    parser.add_argument('--thumbnail_precompress', action='store_true', help='Generating thumbnail post-/pre-compress')

    # Training
    parser.add_argument('--num_workers', type=int, help='Number of workers', default=8)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=8)
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs', default=150)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('--drop_rate', type=float, help='Dropout rate', default=.25)
    parser.add_argument('--seed', type=int, help='Optionally seed everything for deterministic training.')
    parser.add_argument('--resume_dir', type=str, help='Path to checkpoints')
    parser.add_argument('--resume', type=str, help='Model from which to resume training')
    parser.add_argument('--patience', type=int, help="Stop training if validation loss has not improved for X epochs", default=5)

    # Output
    parser.add_argument('--output_dir', type=str, default='model')
    parser.add_argument('--experiment_dir_suffix', type=str, help='Suffix for output directory')
    parser.add_argument('--print_freq', default=100, type=int, help='print frequency')

    # Parse args
    args = vars(parser.parse_args())
    print(f'{json.dumps(args, indent=4)}')

    # fixed settings
    args['tr_csv'] = str(pathlib.Path(args['dataset']) / args['tr_csv'])
    args['va_csv'] = str(pathlib.Path(args['dataset']) / args['va_csv'])
    args['pre_rotate'] = True
    args['post_flip'] = True

    train(args)
