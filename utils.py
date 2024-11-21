import os
import torch
from torch.utils.tensorboard import SummaryWriter

def create_dir_ifnot_exist(folder):
    if os.path.exists(folder):
        return
    
    os.mkdir(folder)

def save_checkpoint(checkpoints_dir, model, checkpoint_name, save_type):

    create_dir_ifnot_exist(checkpoints_dir)
    if torch.cuda.device_count() > 1:

        torch.save(model.module.state_dict(), os.path.join(checkpoints_dir, f'{checkpoint_name}-{save_type}.pth'))
    else:

        torch.save(model.state_dict(), os.path.join(checkpoints_dir, f'{checkpoint_name}-{save_type}.pth'))



def load_checkpoint(checkpoints_dir, checkpoint_name, save_type):
    return torch.load(os.path.join(checkpoints_dir, f'{checkpoint_name}-{save_type}.pth'))

def get_tensorboard_writer(log_dir, filename_suffix = ""):
    create_dir_ifnot_exist(log_dir)
    return SummaryWriter(log_dir=log_dir, filename_suffix=filename_suffix)

