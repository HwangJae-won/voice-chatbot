# i add this line for log and args
import os
import argparse
import logging
from logging import Logger
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Train energy network')
    # general
    parser.add_argument('--kernel_size',default=16,type=int)
    parser.add_argument('--checkpoint', '-c', default=None)
    parser.add_argument('--name', '-n', required=True)
    parser.add_argument('--lr',default=0.001,type=float)
    parser.add_argument('--prenet_dropout', default=0.5,type=float)
#    parser.add_argument('--length', default=4000, type=int,help='input audio spectrogram Length(ms)')
#    parser.add_argument('--data_path',default='data')
#    parser.add_argument('--save_path',default='result')
#    parser.add_argument('--padding',default='repeat')
#    parser.add_argument('--augment',action='store_true',default=False)

    args = parser.parse_args()

    return args


def create_logger(name: str, save_dir: str = None, quiet: bool = False) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of `quiet`.
    One file handler (verbose.log) saves all logs, the other (quiet.log) only saves important info.

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e. print only important info).
    :return: The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, 'quiet.log'))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger