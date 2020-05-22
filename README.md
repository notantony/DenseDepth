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
Serialized with [numpy.tobytes()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tobytes.html), can be deserialized with [numpy.frombuffer()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.frombuffer.html). \
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

See [origin repository](https://github.com/ialhashim/DenseDepth) for more info.

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
