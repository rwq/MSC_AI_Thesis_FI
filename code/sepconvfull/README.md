
# pytorch-sepconv(Backward Implemented)
This is a reference implementation of Video Frame Interpolation via Adaptive Separable Convolution [1] using PyTorch. Given two frames, it will make use of [adaptive convolution](http://graphics.cs.pdx.edu/project/adaconv) [2] in a separable manner to interpolate the intermediate frame. Should you be making use of the work, please cite the paper [1].

This is a modified version of [original code](https://github.com/sniklaus/pytorch-sepconv).

<a href="https://arxiv.org/abs/1708.01692" rel="Paper"><img src="http://content.sniklaus.com/sepconv/paper.jpg" alt="Paper" width="100%"></a>

## Difference from the original code
1. This is a __backpropagation implemented__ version, therefore trainable.
2. run.py was devided into [model.py](./model.py), [train.py](./train.py) and [test.py](./test.py)
3. A module to read dataset([TorchDB.py](./TorchDB.py)) was added.
4. Test module([TestModule.py](./TestModule.py)) for the evaluation with Middlebury dataset was added.

## setup
The separable convolution layer is implemented in CUDA using CuPy, which is why CuPy is a required dependency. It can be installed using `pip install cupy` or alternatively using one of the provided binary packages as outlined in the CuPy repository.

## To Prepare Training Dataset
Two input frames and one output frame are in a folder and the input frames should be named as frame0.png, frame2.png and the output frame should be named as frame1.png. You can name each folder freely.

The training dataset is not provided. We prepared training dataset by cropping [UCF101 dataset](http://crcv.ucf.edu/data/UCF101.php). When creating training dataset, we measured Optical Flow of each frame to balance the motion magnitude of whole dataset.

An example of train dataset is in [db](./db) folder.

## Train
```
python train.py --train ./your/datset/dir --out_dir ./output/folder/tobe/created --test_input ./test/input/of/Middlebury/data --gt ./gt/of/Middlebury/data
```

## Test
```
python test.py --input ./test/input/of/Middlebury/data --gt ./gt/of/Middlebury/data --output ./output/folder/tobe/created --checkpoint --./dir/for/pytorch/checkpoint
```

## video
<a href="http://web.cecs.pdx.edu/~fliu/project/sepconv/demo.mp4" rel="Video"><img src="http://web.cecs.pdx.edu/~fliu/project/sepconv/screen.jpg" alt="Video" width="100%"></a>

## license
The provided implementation is strictly for academic purposes only. Should you be interested in using our technology for any commercial use, please feel free to contact us.

## references
```
[1]  @inproceedings{Niklaus_ICCV_2017,
         author = {Simon Niklaus and Long Mai and Feng Liu},
         title = {Video Frame Interpolation via Adaptive Separable Convolution},
         booktitle = {IEEE International Conference on Computer Vision},
         year = {2017}
     }
```

```
[2]  @inproceedings{Niklaus_CVPR_2017,
         author = {Simon Niklaus and Long Mai and Feng Liu},
         title = {Video Frame Interpolation via Adaptive Convolution},
         booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
         year = {2017}
     }
```

## acknowledgment
This work was supported by NSF IIS-1321119. The video above uses materials under a Creative Common license or with the owner's permission, as detailed at the end.