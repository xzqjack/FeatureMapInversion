#!/usr/bin/env python
'''
This code is to reproduce the result of "Feature Map Inversion" from the paper:

"Every Filter Extract a Specific Texture in Convolutional Neural Networks-short".
Zhiqiang Xia, Ce Zhu, Zhengtao Wang, Qi Guo, Yipeng Liu.
https://arxiv.org/abs/1608.04170

This code referred https://github.com/dmlc/mxnet/tree/master/example/neural-style

Feel free to email Zhiqiang Xia <xzqjack@hotmail.com> if you have questions.
'''

import os
import logging
import argparse

import mxnet as mx

import utils
import model_vgg19_invert

parser = argparse.ArgumentParser(description='Feature Map Inversion and Randomly Modified Code Inversion')

parser.add_argument('--content-image', type=str, default='input/golden_gate.jpg',
                    choices=["golden_gate.jpg", "tubingen.jpg"],
                    help='the content image')
parser.add_argument('--style-image', type=str, default='input/starry_night.jpg',
                    choices=["seated-nude.jpg","the_scream.jpg","starry_night.jpg","frida_kahlo.jpg"],
                    help='the style image')
parser.add_argument('--layer-name', type=str, default="[relu1_1]", 
                    help='assign the target layer you want to modify')
parser.add_argument('--mod_type', type=str, default='random',
                    choices=['original','feature_map', 'random', 'purposeful'],
                    help="choose how to modify code to visualize")
parser.add_argument('--stop-eps', type=float, default=.005,
                    help='stop if the relative chanage is less than eps')     
parser.add_argument('--max-num-epochs', type=int, default=1000,
                    help='the maximal number of training epochs')
parser.add_argument('--max-long-edge', type=int, default=512,
                    help='resize the content image')
parser.add_argument('--lr', type=float, default=.1,
                    help='the initial learning rate')
parser.add_argument('--gpu', type=int, default=0,
                    help='which gpu card to use, -1 means using cpu')
parser.add_argument('--save-epochs', type=int, default=50,
                    help='save the output every n epochs')
parser.add_argument('--output', type=str, default="output",
                    help='save the generated image in folder output')
parser.add_argument('--remove-noise', type=float, default=.2,
                    help='the magtitute to remove noise')

args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG)      

# input
dev = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()
content_img = utils.PreprocessContentImage(args.content_image, args.max_long_edge)
content_name = os.path.basename(args.content_image)[:-4]
style_np = utils.PreprocessStyleImage(args.style_image, shape=content_img.shape)
style_name = os.path.basename(args.style_image)[:-4]

#import model
model_executor = model_vgg19_invert.get_model(content_img.shape, dev, args.layer_name)

#get target style code
style = [mx.nd.zeros(arr.shape, ctx=dev) for arr in model_executor.content]
model_executor.data[:] = style_np
model_executor.executor.forward()
for i in xrange(len(model_executor.content)):
    model_executor.content[i].copyto(style[i])

# get target content code
content_array = [mx.nd.zeros(arr.shape, ctx=dev) for arr in model_executor.content]
content_grad  = [mx.nd.zeros(arr.shape, ctx=dev) for arr in model_executor.content]
model_executor.data[:] = content_img
model_executor.executor.forward()

#unit is feature map of interest
units = [1]
for unit in units:
    for i in xrange(len(model_executor.content)):
        model_executor.content[i].copyto(content_array[i])
        content_array[i] = utils.ModifyCode(content_array[i], args.mod_type, \
                        ctx=dev, style=style[i], feature_map_interest=unit)
    
    img = mx.nd.zeros(content_img.shape, ctx=dev)
    img[:] = mx.rnd.uniform(-0.1, 0.1, img.shape)
    old_img = img.copyto(dev)

    #define optimizer   
 
    lr = mx.lr_scheduler.FactorScheduler(step=10, factor=.9)
    optimizer = mx.optimizer.SGD(
        learning_rate = args.lr,
        momentum = 0.9,
        wd = 0.0005,
        lr_scheduler = lr,
        clip_gradient=10)
    optim_state = optimizer.create_state(0, img)
    
    #start training    
    logging.info('start training arguments %s', args)
    for e in range(args.max_num_epochs):
        img.copyto(model_executor.data)
        model_executor.executor.forward()
        
        # modified code gradient
        for i in xrange(len(model_executor.content)):
            content_grad[i][:] = (model_executor.content[i]-content_array[i])
        model_executor.executor.backward(content_grad)
       
        #update img gradient
        optimizer.update(0, img, model_executor.data_grad, optim_state)
        new_img = img
        eps = (mx.nd.norm(old_img - new_img) / mx.nd.norm(new_img)).asscalar()
        old_img = new_img.copyto(dev)
        logging.info('epoch %d, relative change %f', e, eps)
        if eps < args.stop_eps:
            logging.info('eps < args.stop_eps, training finished')
            break
        
        #save image every save_epochs during training
#        if (e+1)%args.save_epochs ==0:
#            utils.SaveImage(new_img.asnumpy(), args, e+1)
        
    utils.SaveImage(new_img.asnumpy(), args, e+1)