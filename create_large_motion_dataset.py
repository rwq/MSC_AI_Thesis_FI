from pytube import YouTube
from tqdm import tqdm
import os
import datacreator
import metrics
import torch
import cv2
from collections import deque
import argparse



def get_videos():
    os.makedirs(FLAGS.temp_folder, exist_ok=True)

    with open(FLAGS.youtube_urls, 'rb') as f:
        urls = f.readlines()
        urls = [l.decode().strip() for l in urls]
        
    for url in tqdm(urls):
        video = YouTube(url)
        video.streams.filter(file_extension='mp4', res='720p', progressive=True)[0].download(
            output_path = FLAGS.temp_folder,
            filename = video.video_id
        )

def extract_frames():
    files = os.listdir(FLAGS.temp_folder)
    filepaths = [os.path.join(FLAGS.temp_folder, file) for file in files]
    inputs = [(f, 'large_motion_dataset', 4) for f in filepaths]

    for inp in tqdm(inputs, leave=True, position=0):
        datacreator._create_samples_from_file(inp, remove_boundaries=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--youtube_urls', type='str', default='hd_large_motion_links.txt')
    parser.add_argument('--temp_folder', type='str', default='temp_video')

    FLAGS, unparsed = parser.parse_known_args()

    get_videos()
    extract_frames()