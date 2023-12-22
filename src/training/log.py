
import logging
import sys


def setup_custom_logger(name):
    # Taken from https://stackoverflow.com/questions/7621897/python-logging-module-globally
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger


def create_run_name(args):
    run_name = ''
    # run_name += args['stego_method'] + '_'
    run_name += 'alpha_'
    run_name += str(args['alpha'])
    if args['thumbnail']:
        run_name += '_thumb_'
        if args['thumbnail_precompress']:
            run_name += 'pre_'
        if args['thumbnail_precompress']:
            run_name += 'post_'
        sf = args['thumbnail_shape'][0] / args['shape'][0]
        run_name += str(sf)
        run_name += '_'
        run_name += args['thumbnail_kernel']
        if args['thumbnail_antialiasing']:
            run_name += '_aa'
    if args['learning_rate']:
        run_name += '_lr_'
        run_name += str(args['learning_rate'])
    return run_name
