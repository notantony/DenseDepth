# Parent repository
[Covers 2019](https://github.com/notantony/covers2019) 

# API
## Image depthmap building
Build depth map for image. Generated depthmap resolution is constant, 240x320.

#### Input:
Address: `/colormap`, POST \
MIMEs: `applcation/json`, `image/jpeg`, `image/png`

Parameters when Json: \
`data`: base64-encoded image.

#### Output:
JSON: \
`depthmap`: base64-encoded bytes representation of numpy array containing depth of each pixel. Depths are normalized to (0, 1) range according to expected distance in range (10, 10000) cm. \
Serialized with [numpy.tobytes()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tobytes.html), can be desrialized with [numpy.frombuffer()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.frombuffer.html). \
`shape`: shape of the array, equals to `(240, 320)`. \
`dtype`: dtype of the array, equals to `float32`.

<details>
  <summary> <b>Sample: </b> </summary> 

  Request JSON:
  ```json
  {
      "data" : "/9j/4AAQSkZJRgABAQEASABIAAD//gATQ3JlYXRlZCB3a..."
  }
  ```
  
  Response:
  ```json
  {
      "depthmap":"mFCtPUgirj1FFa89XU+vPVgPrz0twa49uJK...",
      "dtype": "float32",
      "shape": "(240, 320)"
  }
  ```
</details>

# Running

Run with `run.sh`:
```
chmod +x run.sh
./run.sh
```

# Reference

See [origin repository](https://github.com/CSAILVision/semantic-segmentation-pytorch) for more info.

Semantic Understanding of Scenes through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, T. Xiao, S. Fidler, A. Barriuso and A. Torralba. International Journal on Computer Vision (IJCV), 2018. (https://arxiv.org/pdf/1608.05442.pdf)

    @article{zhou2018semantic,
      title={Semantic understanding of scenes through the ade20k dataset},
      author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Xiao, Tete and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
      journal={International Journal on Computer Vision},
      year={2018}
    }

Scene Parsing through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso and A. Torralba. Computer Vision and Pattern Recognition (CVPR), 2017. (http://people.csail.mit.edu/bzhou/publication/scene-parse-camera-ready.pdf)

    @inproceedings{zhou2017scene,
        title={Scene Parsing through ADE20K Dataset},
        author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        year={2017}
    }
    














## [High Quality Monocular Depth Estimation via Transfer Learning (arXiv 2018)](https://arxiv.org/abs/1812.11941)
**[Ibraheem Alhashim](https://ialhashim.github.io/)** and **Peter Wonka**

Offical Keras (TensorFlow) implementaiton. If you have any questions or need more help with the code, contact the **first author**.

**[Update]** Added a Colab notebook to try the method on the fly.

**[Update]** Experimental TensorFlow 2.0 implementation added.

**[Update]** Experimental PyTorch code added.

## Results

* KITTI
<p align="center"><img style="max-width:500px" src="https://s3-eu-west-1.amazonaws.com/densedepth/densedepth_results_01.jpg" alt="KITTI"></p>

* NYU Depth V2
<p align="center">
  <img style="max-width:500px" src="https://s3-eu-west-1.amazonaws.com/densedepth/densedepth_results_02.jpg" alt="NYU Depth v2">
  <img style="max-width:500px" src="https://s3-eu-west-1.amazonaws.com/densedepth/densedepth_results_03.jpg" alt="NYU Depth v2 table">
</p>

## Requirements
* This code is tested with Keras 2.2.4, Tensorflow 1.13, CUDA 10.0, on a machine with an NVIDIA Titan V and 16GB+ RAM running on Windows 10 or Ubuntu 16.
* Other packages needed `keras pillow matplotlib scikit-learn scikit-image opencv-python pydot` and `GraphViz` for the model graph visualization and `PyGLM PySide2 pyopengl` for the GUI demo.
* Minimum hardware tested on for inference NVIDIA GeForce 940MX (laptop) / NVIDIA GeForce GTX 950 (desktop).
* Training takes about 24 hours on a single NVIDIA TITAN RTX with batch size 8.

## Pre-trained Models
* [NYU Depth V2](https://s3-eu-west-1.amazonaws.com/densedepth/nyu.h5) (165 MB)
* [KITTI](https://s3-eu-west-1.amazonaws.com/densedepth/kitti.h5) (165 MB)

## Demos
* After downloading the pre-trained model (nyu.h5), run `python test.py`. You should see a montage of images with their estimated depth maps.
* **[Update]** A Qt demo showing 3D point clouds from the webcam or an image. Simply run `python demo.py`. It requires the packages `PyGLM PySide2 pyopengl`. 
<p align="center">
  <img style="max-width:500px" src="https://s3-eu-west-1.amazonaws.com/densedepth/densedepth_results_04.jpg" alt="RGBD Demo">
</p>

## Data
* [NYU Depth V2 (50K)](https://tinyurl.com/nyu-data-zip) (4.1 GB): You don't need to extract the dataset since the code loads the entire zip file into memory when training.
* [KITTI](http://www.cvlibs.net/datasets/kitti/): copy the raw data to a folder with the path '../kitti'. Our method expects dense input depth maps, therefore, you need to run a depth [inpainting method](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) on the Lidar data. For our experiments, we used our [Python re-implmentaiton](https://gist.github.com/ialhashim/be6235489a9c43c6d240e8331836586a) of the Matlab code provided with NYU Depth V2 toolbox. The entire 80K images took 2 hours on an 80 nodes cluster for inpainting. For our training, we used the subset defined [here](https://s3-eu-west-1.amazonaws.com/densedepth/kitti_train.csv).
* [Unreal-1k](https://github.com/ialhashim/DenseDepth): coming soon.

## Training
* Run `python train.py --data nyu --gpus 4 --bs 8`.

## Evaluation
* Download, but don't extract, the ground truth test data from [here](https://s3-eu-west-1.amazonaws.com/densedepth/nyu_test.zip) (1.4 GB). Then simply run `python evaluate.py`.

## Reference
Corresponding paper to cite:
```
@article{Alhashim2018,
  author    = {Ibraheem Alhashim and Peter Wonka},
  title     = {High Quality Monocular Depth Estimation via Transfer Learning},
  journal   = {arXiv e-prints},
  volume    = {abs/1812.11941},
  year      = {2018},
  url       = {https://arxiv.org/abs/1812.11941},
  eid       = {arXiv:1812.11941},
  eprint    = {1812.11941}
}
```
