import torch
import numpy as np
from utils.class_utils import get_cls_from_path
import logging
from utils.hparams import set_hparams, hparams


if __name__ == '__main__':
    # use --config=<path to *.yaml> to control all training related settings
    # use --exp_name=<str> to set the path to store log and checkpoints
    logging.basicConfig(level=logging.INFO)
    from utils.hparams import set_hparams, hparams

    set_hparams()
    # hparams["buffer_capacity"] = 20000
    trainer = get_cls_from_path(hparams['trainer_path'])()
    trainer.run()
