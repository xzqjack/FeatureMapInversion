# Feature Map Inversion with MXnet

This code is used to reproduce the experiments in the paper with MXnet:
[Zhiqiang Xia, Ce Zhu, Zhengtao Wang, Qi Guo, Yipeng Liu. "Every Filter Extract a Specific Texture in Convolutional Neural Networks-short"](https://arxiv.org/abs/1608.04170).

## Installation
This code is written in Python and requires [MXnet](https://github.com/dmlc/mxnet). If you're on Ubuntu, install MXnet in your home directory as the link described:
* [install MXnet and its Python interface](http://mxnet.readthedocs.io/en/latest/how_to/build.html)
* [Some Python libraries are required and can be installed quickly via Anaconda](https://www.continuum.io/downloads)

## Usage
**Input content images:**

<img src="https://github.com/xzqjack/FeatureMapInversion/blob/master/input/the golden gate bridge.jpg" width="300px">
<img src="https://github.com/xzqjack/FeatureMapInversion/blob/master/input/the tubingen.jpg" width="300px">

**Input style images:**

<img src="https://github.com/xzqjack/FeatureMapInversion/blob/master/input/the frida kahlo.jpg" width="150px">
<img src="https://github.com/xzqjack/FeatureMapInversion/blob/master/input/the seated nude.jpg" width="150px">
<img src="https://github.com/xzqjack/FeatureMapInversion/blob/master/input/the starry night.png" width="150px">
<img src="https://github.com/xzqjack/FeatureMapInversion/blob/master/input/the scream.jpg" width="150px">

To visualize modified code, you can run
```python
python vis_invert.py [content-image] [style-image] [layer-name] [mod_type]
```
* layer-name must be str like `"[relu1_1, relu2_1, relu3_1]"`
* mod_type should be `original`, `feature_map`, `random`, or `purposeful`

**Feature Map Inversion:**

<img src="https://github.com/xzqjack/FeatureMapInversion/blob/master/output/feature map of the golden gate.png" width="600px">

**Randomly Modified Code Inversion:**

<img src="https://github.com/xzqjack/FeatureMapInversion/blob/master/output/random.png" width="600px">

To do style transfer, you can run
```
python vis_style.py [content-image] [style-image] [layer-name] [mod_type]
```
* layer-name must be str like `"[relu1_1, relu2_1, relu3_1]"`
* mod_type should be `original` or `purposeful_optimization`
* Content / style tradeoff, you can set parameters `[content-weight]` and `[style-weight]`

**Purposefully Modified Code Inversion:**

<img src="https://github.com/xzqjack/FeatureMapInversion/blob/master/output/style_transfer3.png" width="600px">

## Reference

This code referred https://github.com/dmlc/mxnet/tree/master/example/neural-style.

## Future work

To add "Activation Maximization" such as deepdream.