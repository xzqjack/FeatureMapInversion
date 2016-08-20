#!/usr/bin/env python
'''
This code is to reproduce the result of "Feature Map Inversion" from the paper:

"Every Filter Extract a Specific Texture in Convolutional Neural Networks-short".
Zhiqiang Xia, Ce Zhu, Zhengtao Wang, Qi Guo, Yipeng Liu.
https://arxiv.org/abs/1608.04170

This code referred https://github.com/dmlc/mxnet/tree/master/example/neural-style

Feel free to email Zhiqiang Xia <xzqjack@hotmail.com> if you have questions.
'''

import mxnet as mx
import numpy as np
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)

import utils
import model_vgg19_style

parser = argparse.ArgumentParser(description='neural style')

parser.add_argument('--content-image', type=str, default='input/tubingen.jpg',
                    choices=["golden_gate.jpg", "tubingen.jpg"],
                    help='the content image')
parser.add_argument('--style-image', type=str, default='input/starry_night.jpg',
                    choices=["seated-nude.jpg","the_scream.jpg","starry_night.jpg","frida_kahlo.jpg"],
                    help='the style image')
parser.add_argument('--layer-name', type=str, default="[relu1_1,relu2_1,relu3_1,relu4_1]",
                    help='assign the output layer')
parser.add_argument('--mod_type', type=str, default='original',
                    choices=["original", 'purposeful_optimization'],
                    help="choose how to modify code to visualize")                
parser.add_argument('--stop-eps', type=float, default=.005,
                    help='stop if the relative chanage is less than eps')
parser.add_argument('--content-weight', type=float, default=10,
                    help='the weight for the content image')
parser.add_argument('--style-weight', type=float, default=1,
                    help='the weight for the style image')                                         
parser.add_argument('--max-num-epochs', type=int, default=1000,
                    help='the maximal number of training epochs')
parser.add_argument('--max-long-edge', type=int, default=512,
                    help='resize the content image')
parser.add_argument('--lr', type=float, default=.1,
                    help='the initial learning rate')
parser.add_argument('--gpu', type=int, default=0,
                    help='which gpu card to use, -1 means using cpu')
parser.add_argument('--output', type=str, default="output",
                    help='save the generated image in folder output')
parser.add_argument('--save-epochs', type=int, default=50,
                    help='save the output every n epochs')
parser.add_argument('--remove-noise', type=float, default=.2,
                    help='the magtitute to remove noise')

args = parser.parse_args()   
    
# input
ctx = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()
content_np = utils.PreprocessContentImage(args.content_image, args.max_long_edge)
style_np = utils.PreprocessStyleImage(args.style_image, shape=content_np.shape)

#input model
model_executor = model_vgg19_style.get_model(content_np.shape, ctx,args.layer_name)
   
# get style representation
style_grad = [mx.nd.zeros(style.shape, ctx=ctx) for style in model_executor.style]
style_target = []
model_executor.data[:] = style_np
model_executor.executor.forward()
for i in range(len(model_executor.style)):
    style_target.append(utils.get_style_target(model_executor.style[i], ctx, args.mod_type))
    
# get content representation
content_array = mx.nd.zeros(model_executor.content.shape, ctx=ctx)
content_grad  = mx.nd.zeros(model_executor.content.shape, ctx=ctx)
model_executor.data[:] = content_np
model_executor.executor.forward()
model_executor.content.copyto(content_array)


img = mx.nd.zeros(content_np.shape, ctx=ctx)
img[:] = mx.rnd.uniform(-0.1, 0.1, img.shape)

#denife optimizer
lr = mx.lr_scheduler.FactorScheduler(step=10, factor=.9)
optimizer = mx.optimizer.SGD(
    learning_rate = args.lr,
    momentum = 0.9,
    wd = 0.005,
    lr_scheduler = lr,
    clip_gradient = 10)
optim_state = optimizer.create_state(0, img)

#start training
logging.info('start training arguments %s', args)
old_img = img.copyto(ctx)
	       
for e in range(args.max_num_epochs):
    img.copyto(model_executor.data)
    model_executor.executor.forward()

    # style gradient
    for i in range(len(model_executor.style)):
        style_grad[i][:]  = utils.get_style_grad(model_executor.style[i], style_target[i], ctx, args.mod_type)
        style_grad[i][:] *= args.style_weight

    # content gradient
    if args.mod_type == "original":
        content_grad[:] = (model_executor.content - content_array)
    elif args.mod_type == "purposeful_optimization":
        content_grad[:] = (model_executor.content - content_array) /np.prod(content_array.shape[2:])
    content_grad[:] *= args.content_weight
    
    grad_array = style_grad + [content_grad]
    model_executor.executor.backward(grad_array)
    
    #update img gradient
    optimizer.update(0, img, model_executor.data_grad, optim_state)        
    new_img = img
    eps = (mx.nd.norm(old_img - new_img) / mx.nd.norm(new_img)).asscalar()
    old_img = new_img.copyto(ctx)
    logging.info('epoch %d, relative change %f', e, eps)
    if eps < args.stop_eps:
        logging.info('eps < args.stop_eps, training finished')
        break

        #save image every save_epochs during training
#        if (e+1)%args.save_epochs ==0:
#            utils.SaveImage(new_img.asnumpy(), args, e+1)
        
utils.SaveImage(new_img.asnumpy(), args, e+1)