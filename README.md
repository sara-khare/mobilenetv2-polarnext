# **PolarNeXt: Rethink Instance Segmentation with Polar Representation**

The code for implementing the **PolarNeXt**. 


## News
- Training code is uploaded. It supports training on dual NVIDIA RTX 4090D GPUs with about 16 hours. (2025.06.30)
- Validation code is updated. It supports inference at 49 FPS speed on a single NVIDIA RTX 4090D GPU. (2024.12.10)
- This work has been submitted to CVPR 2025. To ensure anonymity, all information regarding the authors and affiliations will remain undisclosed. (2024.11.15)


## Results
![Figure2](imgs/Figure2.jpg)

## Models

| Backbone | MS train | Lr schd | FPS  | AP<sup>val</sup> | AP<sup>test</sup> |        Weights         |
| :------: | :------: | :-----: | :--: | :--------------: | :---------------: | :--------------------: |
|   [R-50](projects/PolarNeXt/configs/polarnext_r50-torch_fpn_1x_coco.py)   |    N     |   1x    |  49  |       33.9       |         -         |           -            |
|   [R-50](projects/PolarNeXt/configs/polarnext_r50-torch_fpn_3x_ms_coco.py)   |    Y     |   3x    |  49  |       35.7       |       36.1        | [model](https://pan.baidu.com/s/1LShE7EbeBsuK77I0hcSHbw?pwd=lyrn) |
|  [R-101](projects/PolarNeXt/configs/polarnext_r101-torch_fpn_3x_ms_coco.py)   |    Y     |   3x    |  38  |       37.1       |       37.4        | [model](https://pan.baidu.com/s/1yXuZ01CQvnSeQNtzDOQYsA?pwd=lyrn) |

- All models are trained on MS-COCO *train2017*.
- Data augmentation only contains random flip and scale jitter.
- For all 3x schedule settings, we default to using SGD to maintain fair comparison with other models. AdamW is recommended for 1x schedule, as using SGD with small batch size may lead to AP fluctuations of 0.2–0.4.

## Installation

Our PolarNeXt is fully based on [MMDetection](https://github.com/open-mmlab/mmdetection) **v3.3.0**. Please check [INSTALL.md](https://mmdetection.readthedocs.io/en/latest/get_started.html) for installation instructions.

#### Option 1: Install as a standalone project

You can install this project just like any standard MMDetection-based repository:

```bash
git clone https://github.com/Sun15194/PolarNeXt.git
cd PolarNeXt
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install -v -e .
```

#### Option 2: Integrate into your own MMDetection codebase

You may also directly integrate our project into your existing MMDetection framework by following these steps:

1. Copy the entire `projects/` folder into your MMDetection root directory.
2. Overwrite the following files in your MMDetection repo with ours:
   - `mmdet/structures/mask/structures.py`
   - `setup.py` (from the root directory)
3. Run the following command in the root directory of your MMDetection repo to install the environment:

```bash
pip install -v -e .
```


To avoid potential errors, the following versions are recommanded:

```python
Python == 3.10
PyTorch == 2.0.0
CUDA == 11.8
numpy == 1.24.3
matplotlib == 3.5.0
```
> ⚠️ **Note**: Make sure your MMDetection version is exactly **v3.3.0** to avoid compatibility issues.



## Run

(1) Training command:

- ```bash tools/dist_train.sh projects/PolarNeXt/configs/polarnext_r50-torch_fpn_3x_ms_coco.py --auto-scale-lr```

(2) Validation command:

- ```python tools/test.py projects/PolarNeXt/configs/polarnext_r50-torch_fpn_3x_ms_coco.py $YOUR_PTH_FILE ```

(3) Inference Speed command:

- ```python tools/analysis_tools/benchmark.py projects/PolarNeXt/configs/polarnext_r50-torch_fpn_3x_ms_coco.py --checkpoint $YOUR_PTH_FILE --task 'inference' ```

Notes: Considering that mask AP calculation requires converting polygons to pixel-level mask format, to ensure a fair comparison of inference speed, please comment out lines #600–603 in [`projects/PolarNeXt/model/head.py`](projects/PolarNeXt/model/head.py).
