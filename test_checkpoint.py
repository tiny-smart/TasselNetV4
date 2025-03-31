
import numpy as np
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import util.misc as misc

from datasets.datasets import build_dataloader
from util.misc import plot_learning_curves
from models.build_model import build_model


def main(config):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print(config)
    device = torch.device(config['Training']['device'])

    resume_path = config['Resume']['resume_path']

    seed = config['Training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    config['Dataset'].update({'validation':['val','test']})
    # define dataloader
    data_loader_val = build_dataloader('val',**config['Dataset'])
    model = build_model(**config['Model'])
    model.to(device)
    model_without_ddp = model
    name = config['Model']['name']
    print("Model = %s" % str(model_without_ddp))
    misc.load_model_check(resume=resume_path, model_without_ddp=model_without_ddp)
    from val_local import val_func_multihead

    # visualize
    vis_path = os.path.join(os.getcwd(),'visual',config['Model']['name'])
    vis_path = None

    mae, mse, wca, r2 = val_func_multihead(model=model,
                                                    dataloader=data_loader_val,
                                                    config=config,
                                                    vis=vis_path)
    print(f"{name}:{resume_path}")
    print('Val MAE: {:5.2f}, RMSE: {:5.2f}, WCA: {:5.2f}, R2: {:5.2f} '.format(mae, mse, wca, r2))


if __name__ == '__main__':
    config = misc.load_yaml('cfg_test/local_32_64_128_loose.yml', encoding="utf-8")

    main(config)
