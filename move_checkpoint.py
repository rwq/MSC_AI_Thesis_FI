import os
import shutil
import torch
import time

ckpt_folder_old = 'models'
ckpt_folder_new ='models_backup'

ckpt_filename = 'checkpoint_1594777175_seed_0_lr=0.001_lr2=0.0001_loss=l1_pretrain=1_kq_d_size=45_kq_d_scale=2_kq_size=45_optimizer=adamax_input_size=4_kl_size=51_kl_d_size=45_kl_d_scale=2'
ckpt_fp_old = os.path.join(ckpt_folder_old, ckpt_filename)
ckpt_fp_new = os.path.join(ckpt_folder_new, ckpt_filename)

while True:

    # check if file exists at filepath

    if os.path.exists(ckpt_fp_old):
        print('[{}] Moving and copying files.'.format(time.ctime()))
        # move to new folder
        shutil.move(ckpt_fp_old, ckpt_fp_new)

        checkpoint = torch.load(ckpt_fp_new)
        epoch_nr = checkpoint['epoch']
        
        shutil.copyfile(ckpt_fp_new, os.path.join(ckpt_folder_new, '{}_{}'.format(epoch_nr, ckpt_filename)))

    else:
        print('[{}] does not exist.'.format(time.ctime()))

    time.sleep(60)
