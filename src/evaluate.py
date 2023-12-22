
import argparse
import json
import pathlib
import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from training.data.loader import get_data_loader
from training.model import get_b0, seed_everything
from training.log import setup_custom_logger
from training.metrics import AverageMeter, AccuracyMeter, MisclassificationMeter, PrecisionMeter, RecallMeter
from training.metrics import PEMeter, PMD5FPMeter, RocAucMeter, wAUCMeter
from training.metrics import PredictionWriter

log = setup_custom_logger(pathlib.Path(__file__).name)

ARGS_COLS = [
    'model',
    'dataset',
    'te_csv',
    'quality',
    'stego_method',
    'alpha',
    'beta',
    'hshift',
    'thumbnail',
    'thumbnail_precompress',
    'thumbnail_stego',
    'thumbnail_diff',
    'thumbnail_upside',
    'thumbnail_antialiasing',
    'thumbnail_kernel',
    'thumbnail_size',
    'thumbnail_hshift',
]
SCORES_COLS = [
    'loss',
    'accuracy',
    'misclassification',
    'precision',
    'recall',
    'p_e',
    'p_md^5fp',
    'wauc',
    'wauc_alaska',
]


def evaluate_model(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion, device,
    loss_meter,
    target_meters,
    score_meters,
) -> float:

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            batch_size = images.size(0)

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, targets)

            # Skip softmax activation as we are only interested in the argmax
            pred = output.data
            _, y_pred = torch.max(pred, dim=1)

            # Record loss and performance
            loss_meter.update(loss.item(), batch_size)
            for meter in target_meters:
                meter.update(
                    targets.cpu().numpy(),
                    y_pred.detach().cpu().numpy(),
                )
            for meter in score_meters:
                meter.update(
                    targets.cpu().numpy(),
                    pred[:, 1].detach().cpu().numpy(),
                )


def evaluate_for_dataset(loader, model, criterion, device, suffix):

    # Create meters
    loss_meter = AverageMeter('Loss', ':.4e')
    acc_meter = AccuracyMeter('Misc', ':4.3f')
    misc_meter = MisclassificationMeter('Misc', ':4.3f')
    precision_meter = PrecisionMeter('Prec', ':4.3f')
    recall_meter = RecallMeter('Rec', ':4.3f')
    pe_meter = PEMeter('p_e', ':4.3f')
    pmd5fp_meter = PMD5FPMeter('p_md^5fp', ':4.3f')
    wauc_meter = RocAucMeter('wAUC', ':4.3f')
    wauc_alaska_meter = wAUCMeter('wAUC Alaska', ':4.3f')
    # Create writer
    pred_writer = PredictionWriter()

    print(f'evaluate_for_dataset with suffix {suffix}')

    # Evaluate model
    evaluate_model(
        loader=loader,
        model=model,
        criterion=criterion,
        device=device,
        loss_meter=loss_meter,
        target_meters=(
            acc_meter,
            misc_meter,
            precision_meter,
            recall_meter,
        ),
        score_meters=(
            pe_meter,
            pmd5fp_meter,
            wauc_meter,
            wauc_alaska_meter,
            pred_writer,
        )
    )

    # write predictions
    pred_writer.write(f'model/predictions{suffix}.csv')

    return {
        'loss': loss_meter.avg,
        'accuracy': acc_meter.avg,
        'misclassification': misc_meter.avg,
        'precision': precision_meter.avg,
        'recall': recall_meter.avg,
        'p_e': pe_meter.avg,
        'p_md^5fp': pmd5fp_meter.avg,
        'wauc': wauc_meter.avg,
        'wauc_alaska': wauc_alaska_meter.avg,
    }


def evaluate(args):
    # Set up directory name for output directory
    experiment_dir = args['model']
    result_file = experiment_dir / 'results.csv'

    # Get model and log directories
    # Concatenate path to one subdirectory for logging
    log_dir = experiment_dir / 'log'
    model_file = experiment_dir / 'model' / 'best_model.pt.tar'

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
    in_chans = 1 if args['grayscale'] else 3
    if args['thumbnail'] or args.get('thumbnail_diff', False):
        in_chans = 2 * in_chans

    # Data loaders
    args['post_flip'] = args['post_rotate'] = args['pre_rotate'] = False
    te_loader, _ = get_data_loader(args['te_csv'], args, in_chans)
    print(f'Evaluating on {len(te_loader)} batches')

    # Set up model
    model = get_b0(
        in_chans=in_chans,
        shape=args['shape'],
        device=device
    )
    summary(
        model,
        input_size=(args['batch_size'], in_chans, *args['shape'])
    )

    # Set up loss and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load best checkpoint
    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    log.info(f'=> loaded trained model {model_file} ({checkpoint["epoch"]} epochs)')

    # Evaluation
    scores = evaluate_for_dataset(
        loader=te_loader,
        model=model,
        criterion=criterion,
        device=device,
        suffix=('_stego' if args['thumbnail_stego'] else '_cover') if not args['thumbnail'] else ''
    )

    # write columns (on first write)
    if not pathlib.Path(result_file).exists():
        with open(result_file, 'a') as f:
            f.write(
                ','.join(ARGS_COLS + SCORES_COLS) + '\n'
            )

    # write results
    print(scores)
    with open(result_file, 'a') as f:
        s = [
            str(args[c])
            for c in ARGS_COLS
        ] + [
            str(scores[c])
            for c in SCORES_COLS
        ]
        f.write(','.join(s) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--dataset', type=pathlib.Path, help='Path to image root', default='/scratch/martin.benes/alaska_20230303')
    parser.add_argument('--te_csv', type=pathlib.Path, help='Path to csv file containing the test images', default='config/split_te.csv')
    # Data: Covers
    parser.add_argument('--shape', nargs='+', type=int, default=None, help='Dataset shape')
    parser.add_argument('--quality', type=int, default=None, help='Selected specific JPEG quality')
    # Data: Stegos
    parser.add_argument('--stego_method', type=str, default=None, help='Selected stego method')
    parser.add_argument('--alpha', type=float, default=None, help='Selected embedding rate (alpha)')
    parser.add_argument('--hshift', type=int, default=0, help='Horizontal shift of the image.')
    # Data: Thumbnail
    parser.add_argument('--thumbnail_shape', nargs='+', type=int, default=None, help='Thumbnail shape')
    parser.add_argument('--thumbnail_kernel', type=str, default=None, help='Select thumbnail interpolation kernel')
    parser.add_argument('--thumbnail_antialiasing', action='store_true', default=None, help='Use thumbnails with AA filter instead of without')
    parser.add_argument('--thumbnail_no_antialiasing', action='store_false', dest='thumbnail_antialiasing', help='Use thumbnails with AA filter instead of without')
    parser.add_argument('--thumbnail_quality', type=int, default=None, help='Selected specific JPEG thumbnail quality')
    parser.add_argument('--thumbnail_precompress', action='store_true', default=None, help='Use pre-compress thumbnails')
    parser.add_argument('--thumbnail_postcompress', action='store_false', dest='thumbnail_precompress', help='Use post-compress thumbnails')
    parser.add_argument('--thumbnail_stego', action='store_true', help='Use stego thumbnail')
    parser.add_argument('--thumbnail_hshift', type=int, default=0, help='Horizontal shift of the upsampled thumbnail.')

    # Evaluation
    parser.add_argument('--num_workers', type=int, help='Number of workers', default=0)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=None)
    parser.add_argument('--seed', type=int, help='Optionally seed everything for deterministic training.')

    # Parse args
    args = vars(parser.parse_args())
    # Update args
    with open(args['model'] / 'config.json') as f:
        config = json.load(f)
    config.update((k, v) for k, v in args.items() if v is not None)
    if 'thumbnail_diff' not in config:
        config['thumbnail_diff'] = False
    config['thumbnail_size'] = config['thumbnail_shape'][0]
    config['thumbnail_upside'] = False

    print(config)

    evaluate(config)
