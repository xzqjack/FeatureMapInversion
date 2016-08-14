# Feature Map Inversion with MXnet

This code is used to reproduce the experiments in the paper with MXnet:
Zhiqiang Xia, Ce Zhu, Zhengtao Wang, Qi Guo, Yipeng Liu. "Every Filter Extract a Specific Texture in Convolutional Neural Networks-short".

## Installation
This code is written in Python and requires [MXnet](https://github.com/dmlc/mxnet). If you're on Ubuntu, install MXnet in your home directory as the link described:
* [install MXnet and its Python interface](http://mxnet.readthedocs.io/en/latest/how_to/build.html)
* [Some Python libraries are required and can be installed quickly via Anaconda](https://www.continuum.io/downloads)

## Usage
**Input content images:**

<img src="https://github.com/xzqjack/FeatureMapInversion/blob/master/input/the golden gate bridge.jpg" height="200px">
<img src="https://github.com/xzqjack/FeatureMapInversion/blob/master/input/the tubingen.jpg" height="200px">

**Input style images:**

<img src="https://github.com/xzqjack/FeatureMapInversion/blob/master/input/the frida kahlo.jpg" height="175px">
<img src="https://github.com/xzqjack/FeatureMapInversion/blob/master/input/the seated nude.jpg" height="175px">
<img src="https://github.com/xzqjack/FeatureMapInversion/blob/master/input/the starry night.png" height="175px">
<img src="https://github.com/xzqjack/FeatureMapInversion/blob/master/input/the scream.jpg" height="175px">

To visualize modified code, you can run
```python
python vis_invert.py [content-image] [style-image] [layer-name] [mod_type]
```
* layer-name must be str like `"[relu1_1, relu2_1, relu3_1]"`
* mod_type should be `original`, `feature_map`, `random`, or `purposeful`

**Feature Map Inversion:**

<img src="https://github.com/xzqjack/FeatureMapInversion/blob/master/output/feature map of the golden gate.png" height="200px">

**Randomly Modified Code Inversion:**

<img src="https://github.com/xzqjack/FeatureMapInversion/blob/master/output/random.png" height="200px">

To do style transfer, you can run
```
python vis_style.py [content-image] [style-image] [layer-name] [mod_type]
```
* layer-name must be str like `"[relu1_1, relu2_1, relu3_1]"`
* mod_type should be `original` or `purposeful_optimization`
* Content / style tradeoff, you can set parameters `[content-weight]` and `[style-weight]`

**Purposefully Modified Code Inversion:**

<img src="https://github.com/xzqjack/FeatureMapInversion/blob/master/output/style_transfer3.png" height="200px">
