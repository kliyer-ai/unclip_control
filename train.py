from share import *

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import DataLoader
from custom_dataset_cross import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
# resume_path = './models/control-base.ckpt'
resume_path = './models/testest.ckpt'
experiment_name = 'kin_hed_cross_multi_1'
config_path='./models/cldm_v15_cross_multi.yaml'

batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(config_path).cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset('kin_hed2')
validation_set = MyDataset('kin_hed_val')


logger = ImageLogger(batch_frequency=logger_freq, name=experiment_name)
# wandb_logger = WandbLogger(name='kin_hed_cross_2', project="ControlNet")
# tbl = TensorBoardLogger(save_dir='ControlNet', name='kin_hed_cross_2')

dataloader = DataLoader(dataset, num_workers=64, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_set, num_workers=64, batch_size=batch_size, shuffle=True)


trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger],
                    limit_val_batches=1,
                    val_check_interval=logger_freq,
                    num_sanity_val_steps=2,
                    default_root_dir='train_log/' + experiment_name
                    ) #, logger=[wandb_logger, tbl])



# Train!
trainer.fit(model, dataloader, validation_loader)
