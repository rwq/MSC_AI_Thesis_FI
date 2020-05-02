import torch
import cv2
import imageio
from tqdm import tqdm
import multiprocessing
import os
import random

OUTPUT_EXT = 'jpeg'
FILEPATH_UCF101 = 'datasets/UCF-101'
FILEPATH_ADOBE240 = 'datasets/Adobe240/original_high_fps_videos/'
FILEPATH_CREATED_DATASETS = 'created_datasets'
N_THREADS = 4

def _get_frames_by_indices(cap, indices):
    frames = []
    
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        _, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)
        
    return frames



# def _create_samples_from_file(file):
#     cap = cv2.VideoCapture(file)
#     n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     k=1
#     # determine subset
#     domain = [[i+j for j in range(k+2)] for i in range(n_frames-k-1)]
    
#     i = 0

#     while len(domain) > 0:
#         datapoint = random.choice(domain)
#         domain = [x for x in domain if sum([x.count(y) for y in datapoint]) == 0]

#         frames = _get_frames_by_indices(cap, datapoint)

#         for j, frame in enumerate(frames):
#             filename, ext = os.path.basename(file).split('.')
#             filepath_out = os.path.join(FILEPATH_CREATED_DATASETS, 'UCF101', f"{filename}__{i}_{j}.jpeg")
#             imageio.imsave(filepath_out, frame)


#         i += 1

def _create_samples_from_file(inp):
    file, dataset, k = inp
    cap = cv2.VideoCapture(file)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if dataset == 'ADOBE240':
        remove_last_frames = 360
    else:
        remove_last_frames = 0
    # determine subset
    domain = [[i+j for j in range(k+2)] for i in range(n_frames-k-2-remove_last_frames)] # laatste frame soms leeg
    
    output_dir = os.path.join(FILEPATH_CREATED_DATASETS, dataset)
    os.makedirs(output_dir, exist_ok=True)

    i = 0
    while len(domain) > 0:
        datapoint = random.choice(domain)
        domain = [x for x in domain if sum([x.count(y) for y in datapoint]) == 0]
        # print(datapoint)
        frames = _get_frames_by_indices(cap, datapoint)

        for j, frame in enumerate(frames):
            filename, ext = os.path.basename(file).split('.')
            filepath_out = os.path.join(output_dir, f"{filename}__{i}_{j}.jpeg")
            imageio.imsave(filepath_out, frame)

        i += 1

    cap.release()


def generate_adobe240(k=3):
    files = [f for f in os.listdir(FILEPATH_ADOBE240) if not f.endswith('.DS_Store')]
    files = [
        (os.path.join(FILEPATH_ADOBE240, f), 'ADOBE240', k) for f in files
    ]

    os.makedirs(os.path.join(FILEPATH_CREATED_DATASETS, 'ADOBE240'), exist_ok=True)

    with multiprocessing.Pool(N_THREADS) as P:
        files = files[127:]
        file_it = tqdm(
                iter(P.imap_unordered(_create_samples_from_file, files)),
                total=len(files),
                desc='Parsing all ADOBE240 files'
            )
        for _ in file_it:
            pass



def generate_ucf101(k=3):

    folders = os.listdir(FILEPATH_UCF101)
    files = [
        os.path.join(FILEPATH_UCF101, folder, f) for folder in folders for f in os.listdir(os.path.join(FILEPATH_UCF101, folder))
    ]

    files = [
        (f, 'UCF101', k) for f in files
    ]


    with multiprocessing.Pool(N_THREADS) as P:
        files = files[:100]
        file_it = tqdm(
                iter(P.imap_unordered(_create_samples_from_file, files)),
                total=len(files),
                desc='Parsing all UCF101 files'
            )
        for _ in file_it:
            pass

if __name__ == '__main__':
    generate_adobe240()