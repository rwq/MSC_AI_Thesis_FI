from pytube import YouTube
from tqdm import tqdm
import os
import shutil
from pytorch_msssim import ssim
import time
import imageio
import argparse
import torch
from torchvision.utils import save_image
from torchvision import transforms
import math
from PIL import Image
from utilities import EdgeMap
from torch.nn.functional import interpolate
# from code.pytorch_pwc2.PWCNetnew import PWCNet

# DATASET_FOLDER = 'large_motion_dataset2'


# model = PWCNet()
# model.load_state_dict(torch.load('code/pytorch_pwc2/network-default.pytorch'))
# model = model.cuda().eval()

parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_dir", type=str, default='C:/ffmpeg/bin', help='path to ffmpeg.exe')
parser.add_argument("--dataset_folder", type=str, default='large_motion_dataset2', help='path to the output dataset folder')
parser.add_argument("--crop_height", type=int, default=256)
parser.add_argument("--crop_width", type=int, default=256)
parser.add_argument("--crop_height_test", type=int, default=720)
parser.add_argument("--crop_width_test", type=int, default=1280)
# parser.add_argument("--img_width", type=int, default=640, help="output image width")
# parser.add_argument("--img_height", type=int, default=360, help="output image height")
# parser.add_argument("--train_test_split", type=tuple, default=(90, 10), help="train test split for custom dataset")
args = parser.parse_args()



# for train/valid/test folds:
# download videos

# train/valid: highest quality, test: 720p progressive

# extract to temp folder

# move from temp to fold (full)


# for train/valid:
# create new folds train/valid cropped using interesting region cropping


def move_to_sequences(in_folder, out_folder, video_id, downsample_fps=1, frames_between_factor=30, max_frames=40000):
    '''
    Move subset of files from in_folder to out_folder and rename
    '''
    filenames = os.listdir(in_folder)
    files = [os.path.join(in_folder, f) for f in filenames]
    
    i = 0
    frame_indices=[0,2,3,4,6]
    # if len(os.listdir(out_folder)) == 0:
    s=1

    n_frames = len(files)
    while i + 6*downsample_fps <= min(n_frames-1, max_frames*downsample_fps):
        sequence_dir = os.path.join(out_folder, f'{video_id}_{s}')
        os.makedirs(sequence_dir)
        
        files_to_move = [files[i+j*downsample_fps] for j in frame_indices]
        
        for j, f in enumerate(files_to_move):
            shutil.move(f, os.path.join(sequence_dir, f'{frame_indices[j]}.jpg'))
        
        i += 30*downsample_fps*frames_between_factor
        s += 1

    return s

def discard_frames(folder, video_id, min_avg_value=10, ssim_threshold=0.3, ssim_range=.5):
    '''
    Iterates over folders and removes frame sequences if it
    -contains a dark frame (avg pixel value < 5)
    -contains a frame boundary (detected by SSIM)
    '''

    
    
    seq_folders = [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith(video_id)]
    
    for seq_folder in tqdm(seq_folders, desc=f'Discarding frames {folder}'):
        imgs = [imageio.imread(os.path.join(seq_folder, f)) for f in os.listdir(seq_folder)]
        imgs = [torch.tensor(img).permute(2,0,1).unsqueeze(dim=0).cuda().float() for img in imgs]
        removed=False
        for i in range(5):
            if imgs[i].mean().item() < min_avg_value:
                print(f'removed {seq_folder}')
                removed=True
                # shutil.rmtree(seq_folder)
                break
        

        ssims = [ssim(imgs[i], imgs[i+1], win_size=11).item() for i in range(4)]
        if min(ssims) < ssim_threshold:
            print(seq_folder, 'min lower than .3')
            # shutil.rmtree(seq_folder)
            removed=True
        

        if max(ssims)-min(ssims) > ssim_range:
            print(f'deleting {seq_folder}')
            # shutil.rmtree(seq_folder)
            removed=True
        
        if removed:
            with open(os.path.join(args.dataset_folder, 'to_remove_lmd.txt'), 'a') as f:
                f.write(seq_folder+'\n')
        
        


def crop_fold(fold, video_id=None):
    crop_size = (args.crop_height, args.crop_width)
    input_folder = os.path.join(args.dataset_folder, fold)
    output_folder = os.path.join(args.dataset_folder, f'{fold}_cropped')

    os.makedirs(output_folder, exist_ok=True)
    t=transforms.ToTensor()
    
    # edge_map = EdgeMap(tau=2, quadratic=True).cuda()
    
    if video_id != None:
        folders = [f for f in os.listdir(input_folder) if f.startswith(video_id)]
    else:
        folders = [f for f in os.listdir(input_folder)]

    for folder in tqdm(folders, desc=f'cropping {fold} {video_id}'):
        folderpath = os.path.join(args.dataset_folder, fold, folder)
        cropped_out_folder = os.path.join(output_folder, folder)
        os.makedirs(cropped_out_folder)
        # inladen
        im1 = t(Image.open(os.path.join(folderpath, '0.jpg')))
        im2 = t(Image.open(os.path.join(folderpath, '2.jpg')))
        y   = t(Image.open(os.path.join(folderpath, '3.jpg')))
        im4 = t(Image.open(os.path.join(folderpath, '4.jpg')))
        im5 = t(Image.open(os.path.join(folderpath, '6.jpg')))
        
        X = torch.stack([im1, im2, im4, im5]).unsqueeze(0).cuda()
        y = y.unsqueeze(0).cuda()

        # downscale X y
        
        
        # crop region vinden
        # flow_intensity = model(y.cuda(), X[:,0].cuda()).abs()
        # print(X.shape, y.shape)
        # edge_intensity = edge_map(X,
        #     interpolate(X.squeeze(0), scale_factor=.25, recompute_scale_factor=True, mode='bilinear', align_corners=False).unsqueeze(0),
        #     interpolate(y, scale_factor=.25, recompute_scale_factor=True, mode='bilinear', align_corners=False),
        # ).unsqueeze(0)

        # edge_intensity = interpolate(
        #     edge_intensity,scale_factor=4, recompute_scale_factor=True, mode='bilinear', align_corners=False
        # )

         
        move1 = (y-X[:,1]).pow(2).mean(dim=(0,1))
        move2 = (y-X[:,2]).pow(2).mean(dim=(0,1))
        weight_map = (move1 + move2).detach().cpu()
        # weight_map = flow_intensity.mean(dim=(0,1)).detach().cpu()


        # upscale weightmap

        # croppen
        B, F, C, H, W = X.size()
            
        start_h = int(crop_size[0]/2)
        end_h = int(H-crop_size[0]/2)
        start_w = int(crop_size[1]/2)
        end_w = int(W-crop_size[1]/2)

        # crop interesting region
        weights = weight_map[start_h:end_h, start_w:end_w].flatten()**2
        # try:
        ind = torch.multinomial(weights.clamp(min=1e-20), 1).item()
        # except:
        #     print(folder)
        #     time.sleep(100)
        start_i = math.floor(ind/(end_w-start_w))
        start_j = ind % (end_w-start_w)

        X = X[0,:,:, start_i:start_i+crop_size[0], start_j:start_j+crop_size[1]].clamp(0,1)
        y = y[0,:, start_i:start_i+crop_size[0], start_j:start_j+crop_size[1]].clamp(0,1)
        
        
        
        # wegschrijven    
        save_image(X[0], os.path.join(output_folder, folder, '0.jpg'))
        save_image(X[1], os.path.join(output_folder, folder, '2.jpg'))
        save_image(y, os.path.join(output_folder, folder, '3.jpg'))
        save_image(X[2], os.path.join(output_folder, folder, '4.jpg'))
        save_image(X[3], os.path.join(output_folder, folder, '6.jpg'))
        
        # delete???





def get_urls(fold):
    filepath = os.path.join(args.dataset_folder, f'{fold}.txt')
    with open(filepath, 'rb') as f:
        lines = f.readlines()
        lines = [l.decode().strip() for l in lines]
        lines = [url for url in lines if not url.startswith('-')]
        lines = [url.split(' ') for url in lines]

    if len(lines) > 0:
        urls, factors, skips = list(zip(*lines))
        factors = map(int, factors)
        skips   = map(int, skips)
            
        
        return urls, factors, skips
    return [], [], []

if __name__ == '__main__':

    VIDEO_PATH = os.path.join(args.dataset_folder, 'videos')
    TEMP_FRAMES_PATH = os.path.join(args.dataset_folder, 'temp_extracted_frames')

    os.makedirs(VIDEO_PATH, exist_ok=True)
    os.makedirs(TEMP_FRAMES_PATH, exist_ok=True)

    # for fold in ['train', 'valid', 'test']:
    for fold in ['valid']:
    
        print(f'[{time.ctime()}] Creating {fold} fold')

        fold_path = os.path.join(args.dataset_folder, fold)
        os.makedirs(fold_path, exist_ok=True)

        # video_filenames = []

        # download videos
        for url, factor, skip in tqdm(list(zip(*get_urls(fold)))):
            
            video = YouTube(url)
            video_path = os.path.join(VIDEO_PATH, video.video_id+'.webm')

            if not os.path.exists(video_path):

                streams = video.streams.order_by('resolution').desc()
                
                i = 1 if streams[0].resolution == '2160p' else 0
                print(url, i)
                video_path = streams[i].download(
                    output_path = VIDEO_PATH, filename=video.video_id
                )
                print('downloaded')

            

            # continue
            temp_frames_path = os.path.join(TEMP_FRAMES_PATH, video.video_id)
            os.makedirs(temp_frames_path)
            
            # ret = os.system('{} -i {} -vf scale={}:{} -vsync 0 -qscale:v 1 {}/%06d.jpg'.format(
            #     os.path.join(args.ffmpeg_dir, "ffmpeg"),
            #     video_path, 
            #     1280, 720,
            #     temp_frames_path
            # ))
            
            
            ret = os.system('{} -i {} -vsync 0 -qscale:v 1 {}/%06d.jpg'.format(
                os.path.join(args.ffmpeg_dir, "ffmpeg"),
                video_path, 
                temp_frames_path
            ))

            if ret: 
                print('error mate')
                exit()

            # move subset of frames to fold
            move_to_sequences(temp_frames_path, fold_path, video.video_id, downsample_fps=factor, frames_between_factor=skip)

            # # delete temp folder
            shutil.rmtree(temp_frames_path)

            # discard all frame boundaries and dark images from fold
            discard_frames(fold_path, video.video_id)
    
    
    

            # create cropped version of train and validation set
            # train_path = os.path.join(args.dataset_folder, 'train') #
            # valid_path = os.path.join(args.dataset_folder, 'valid') #
            
            # create cropped dataset and remove original
            # print('crop_fold', fold)
            if fold in ['train', 'valid']:
                crop_fold(fold, video.video_id) #
            # shutil.rmtree(train_path) #
            
    # shutil.rmtree(valid_path) #

    # remove folder for temporary frames
    # shutil.rmtree(TEMP_FRAMES_PATH) #

    # crop_fold('train')
    # crop_fold('valid')

        

    

