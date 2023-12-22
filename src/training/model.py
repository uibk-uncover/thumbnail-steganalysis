
import numpy as np
import os
import random
import timm
import torch
from typing import Tuple


def get_b0(
    in_chans: int,
    shape: Tuple[int],
    device,
    num_classes: int = 2,
    strict: bool = False,
    **kw
) -> torch.nn.Module:

    model = timm.create_model(
        'efficientnet_b0',
        num_classes=num_classes,
        in_chans=in_chans,
        pretrained=True,
        **kw,
    )
    model.model_name = 'EfficientNetB0'
    fc_name = 'classifier'
    conv_stem_names = ['conv_stem']

    # load weights
    state_dict = torch.hub.load_state_dict_from_url(model.default_cfg['url'])

    # remove FC, if not compatible
    out_fc, _ = state_dict[fc_name + '.weight'].shape
    if out_fc != num_classes:
        del state_dict[fc_name + '.weight']
        del state_dict[fc_name + '.bias']

    # modify first convolution to match the input size
    for conv_stem_name in conv_stem_names:
        weight_name = conv_stem_name + '.weight'
        _, in_conv, _, _ = state_dict[weight_name].shape
        if in_conv != in_chans:
            state_dict[weight_name] = timm.models.adapt_input_conv(in_chans, state_dict[weight_name])
        _, in_conv2, _, _ = state_dict[weight_name].shape

    state_dict['input_size'] = (in_chans, *shape)
    state_dict['img_size'] = shape[0]
    state_dict['num_classes'] = num_classes

    # load weights
    model.load_state_dict(state_dict, strict=strict)
    del state_dict

    # return model on device
    return model.to(device)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
