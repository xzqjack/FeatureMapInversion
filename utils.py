# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 16:16:08 2016

@author: gpu2
"""
import mxnet as mx
import numpy as np
import logging
from skimage import io, transform
from collections import namedtuple
import os
from skimage.restoration import denoise_tv_chambolle
#from skimage.filter import denoise_tv_chambolle 

def PreprocessContentImage(path, long_edge):
    """
    Preprocess target content image
    1. resize image;
    2. swap axis (b,g,r) to (r,g,b)
    3. subtract the images dataset mean 

    Parameters
    --------
    path : str, content image path
    long_edge : int, resize content image according to your gpu memory

    Returns
    out : ndarray (1x3xMxN), prepocessed content image

    """

    img = io.imread(path)
    logging.info("load the content image, size = %s", img.shape[:2])
    factor = float(long_edge) / max(img.shape[:2])
    new_size = (int(img.shape[0] * factor), int(img.shape[1] * factor))
    resized_img = transform.resize(img, new_size)
    sample = np.asarray(resized_img) * 256
    # swap axes to make image from (224, 224, 3) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean
    sample[0, :] -= 123.68
    sample[1, :] -= 116.779
    sample[2, :] -= 103.939
    logging.info("resize the content image to %s", new_size)
    return np.resize(sample, (1, 3, sample.shape[1], sample.shape[2]))
    
def PreprocessStyleImage(path, shape):
    """
    Preprocess target style image
    1. resize style image to make its size same as content image shape
    2. swap axis (b,g,r) to (r,g,b)
    3. subtract the images dataset mean 

    Parameters
    --------
    path : str, target image path
    shape : tupe (1x3xMxN), resize target style image according to target image shape

    Returns
    out : ndarray (1x3xMxN), prepocessed style image   
    """

    img = io.imread(path)
    resized_img = transform.resize(img, (shape[2], shape[3]))
    sample = np.asarray(resized_img) * 255
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)

    sample[0, :] -= 123.68
    sample[1, :] -= 116.779
    sample[2, :] -= 103.939
    return np.resize(sample, (1, 3, sample.shape[1], sample.shape[2]))
    
def PostprocessImage(img):
    """
    Postprocess target style image    
    1. add the images dataset mean to optimized image
    2. swap axis (b,g,r) to (r,g,b) and save it

    Parameters
    --------
    img: ndarray (1x3xMxN), optimized image

    Returns
    out : ndarray (3xMxN), Postprocessed image   
    """

    img = np.resize(img, (3, img.shape[2], img.shape[3]))
    img[0, :] += 123.68
    img[1, :] += 116.779
    img[2, :] += 103.939
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    img = np.clip(img, 0, 255)
    return img.astype('uint8')

def SaveImage(img, args, epoch):
    """
    Postprocess Image and use total tv-norm to denoise postprocessed image
    
    1. postprocess Image
    2. use total tv-norm to denoise postprocessed image
    
    Parameters
    --------
    img: ndarray (1x3xMxN), optimized image

    Returns
    """
    out = PostprocessImage(img)
    out = denoise_tv_chambolle(out, weight=args.remove_noise, multichannel=True)
    if args.mod_type == "purposeful":
        save_name = os.path.join(args.output,"{}_{}_{}_{}_{}.jpg".\
        format(args.layer_name, args.mod_type, os.path.basename(args.content_image)[:-4],\
                 os.path.basename(args.style_image)[:-4], epoch))
    else:
        save_name = os.path.join(args.output,"{}_{}_{}_{}.jpg".\
        format(args.layer_name, args.mod_type, os.path.basename(args.content_image)[:-4], epoch))
    logging.info('save output to %s', save_name)
    io.imsave(save_name, out)

def ModifyCode(code, mod_type, ctx, style=None, feature_map_interest=0):
    """
    how to modify code
    1. original: don't modify code
    2. random: reallocate the sum of every feature map randomly
    3. purposeful: reallocate the sum of every feature map randomly on 
       according to the sum of style code directly without optimization.
       This is not mentioned in paper,
    4. feature_map: enhance the feature map of interest and weaken the others

    Parameters
    --------
    code: ndarray (1xCxHxW), neurons activations at a layer in CNNs
    mod_type: str, how to modify code
    ctx: context of devices, mx.gpu(0) or mx.cpu()
    style: ndarray (1xCxHxW), style target code 
    feature_map_interest: int, choose what feature map to visualize if mod_type is feature_map

    Returns
    new_code: ndarray (1xCxHxW), modified code
    """   

    shape = code.shape
    if mod_type == "original":
        new_code = code
  
        
    if mod_type == "random":
        code = code.asnumpy().squeeze()
        code = code.reshape(shape[1],-1)
        rate = np.random.normal(3,1e3,size=(1,shape[1]))
        rate = np.tile(rate/np.sum(rate), (shape[2]*shape[3],1)).T
        new_code = np.sum(code)*rate*(code/np.tile(np.sum(code,axis=1),(shape[2]*shape[3],1)).T)
        new_code = mx.nd.array(new_code.reshape(shape),ctx=ctx)   
        
    if mod_type == "purposeful":
        style = style.asnumpy().squeeze()
        style = style.reshape(shape[1],-1)
        code = code.asnumpy().squeeze()
        code = code.reshape(shape[1],-1)
        style_rowsum = np.sum(style, axis=1)
        code_row_sum = np.sum(code, axis=1)
        style_index = np.argsort(style_rowsum)
        code_index = np.argsort(code_row_sum)
        sort_code = np.zeros_like(code)        
        for i in range(shape[1]):
            sort_code[style_index[i],:] = code[code_index[i],:]  
            
        sort_code_mean = np.tile(np.mean(sort_code,axis=1), (shape[2]*shape[3],1)).T 
        style_mean = np.tile(np.mean(style,axis=1), (shape[2]*shape[3],1)).T       
        new_code = sort_code-sort_code_mean+style_mean
#        new_code[new_code<0] = 0
        new_code = mx.nd.array(new_code.reshape(shape),ctx=ctx)                     
        
    if mod_type == "feature_map":
        code = code.asnumpy().squeeze()
        code = code.reshape(shape[1],-1)                
        new_code = np.zeros_like(code)
        code_mean = np.sum(code,axis=0) 
        new_code[feature_map_interest,:] = code_mean
        new_code = mx.nd.array(new_code.reshape(shape),ctx=ctx)   
              
    return new_code
    
Executor = namedtuple('Executor', ['executor', 'data', 'data_grad'])	
def StyleGramExecutor(input_shape, ctx):
    """
    calculate Gram matrix of input code and it's gradient
    1. original: don't modify code
    2. random: reallocate the sum of every feature map randomly
    3. purposeful: reallocate the sum of every feature map randomly on 
       according to the sum of style code directly without optimization.
       This is not mentioned in paper,
    4. feature_map: enhance the feature map of interest and weaken the others

    Parameters
    --------
    code: ndarray (1xCxHxW), neurons activations at a layer in CNNs
    mod_type: str, how to modify code
    ctx: context of devices, mx.gpu(0) or mx.cpu()
    style: ndarray (1xCxHxW), style target code 
    feature_map_interest: int, choose what feature map to visualize if mod_type is feature_map

    Returns
    namedtuple: inlcude mx.executor, input data and the gradient of input data
    """      
    # symbol
    data = mx.sym.Variable("conv")
    rs_data = mx.sym.Reshape(data=data, shape=(int(input_shape[1]), int(np.prod(input_shape[2:]))))
    weight = mx.sym.Variable("weight")
    rs_weight = mx.sym.Reshape(data=weight, shape=(int(input_shape[1]), int(np.prod(input_shape[2:]))))
    fc = mx.sym.FullyConnected(data=rs_data, weight=rs_weight, no_bias=True, num_hidden=input_shape[1])
    # executor
    conv = mx.nd.zeros(input_shape, ctx=ctx)
    grad = mx.nd.zeros(input_shape, ctx=ctx)
    args = {"conv" : conv, "weight" : conv}
    grad = {"conv" : grad}
    reqs = {"conv" : "write", "weight" : "null"}
    executor = fc.bind(ctx=ctx, args=args, args_grad=grad, grad_req=reqs)
    return Executor(executor=executor, data=conv, data_grad=grad["conv"])  

def get_style_target(style, ctx, mod_type="purposeful_optimization"):
    """
    how to modify style target code
    1. purposeful_optimization: sum of code along channel axis at a layer

    Parameters
    --------
    style: ndarray (1xCxHxW), neurons activations at a layer in CNNs
    ctx: context of devices, mx.gpu(0) or mx.cpu()
    mod_type: str, how to modify code.

    Returns
    new_code: ndarray (C,1), target style term
    """  

#    shape = style.shape
    if mod_type == "original":
        gram_executor = StyleGramExecutor(style.shape, ctx)
        # get style representation
        style.copyto(gram_executor.data)
        gram_executor.executor.forward()
        new_style = gram_executor.executor.outputs[0]       
        
    elif mod_type == "purposeful_optimization":
        new_style = mx.nd.expand_dims(mx.nd.sum(style, axis=(0,2,3)), axis=1)   

    return new_style  
    

def get_style_grad(style, target, ctx, mod_type):
    
    if mod_type == "original":
        gram_executor = StyleGramExecutor(style.shape, ctx)
        style.copyto(gram_executor.data)
        gram_executor.executor.forward()
        gram_executor.executor.backward([gram_executor.executor.outputs[0] - target])
        grad = gram_executor.data_grad[:]
        grad /= style.shape[1]**2*float(np.prod(style.shape[2:]))
        
    elif mod_type == "purposeful_optimization":
        shape = style.shape
        one_vector = mx.nd.ones((np.prod(shape[2:]), 1), ctx=ctx)
        style = style.reshape((shape[1], np.prod(shape[2:]))) 
        grad = mx.nd.dot( mx.nd.dot(style, one_vector)-target, mx.nd.transpose(one_vector))
        grad = grad.reshape(shape)
        grad /= shape[1]*float(np.prod(shape[2:]))
        
    return grad