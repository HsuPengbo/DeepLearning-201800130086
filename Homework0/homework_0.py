###################################

#请根据需求自己补充头文件、函数体输入参数。
import numpy as np
import torch
###################################
#2 Vectorization
###################################

def vectorize_sumproducts(x,y):
    """
     Takes two 1-dimensional arrays and sums the products of all the pairs.
    :return:
    """
    return np.dot(x,y)
    pass
def vectorize_Relu(x):
    """
    Takes one 2-dimensional array and apply the relu function on all the values of the array.
    :return:
    """
    return np.maximum(x, 0.0)
    pass
def vectorize_PrimeRelu(x):
    """
    Takes one 2-dimensional array and apply the derivative of relu function on all the values of the array.
    :return:
    """
    return np.where(x > 0, 1.0, 0.0)
    pass

######################################
#3 Variable length
######################################

#Slice

def Slice_fixed_point(x,leng,st):
    """
    Takes one 3-dimensional array with the starting position and the length of the output instances.
    Your task is to slice the instances from the same starting position for the given length.
    :return:
    """
    return [d[st:st+leng] for d in x]

    pass
def slice_last_point(x,leng):
    """
     Takes one 3-dimensional array with the length of the output instances.
     Your task is to keeping only the l last points for each instances in the dataset.
    :return:
    """
    return [d[-leng:] for d in x]
    pass
def slice_random_point(x,l):
    """
     Takes one 3-dimensional  array  with  the  length  of the output instances.
     Your task is to slice the instances from a random point in each of the utterances with the given length.
     Please use function numpy.random.randint for generating the starting position.
    :return:
    """
    return [d[np.random.randint(0,len(d)-l):][:l] for d in x]
    pass

#Padding

def pad_pattern_end(x):
    """
    Takes one 3-dimensional array.
    Your task is to pad the instances from the end position as shown in the example below.
    That is, you need to pad the reflection of the utterance mirrored along the edge of the array.
    :return:
    """
    L=max(map(len,x))
    z=[]
    for d in x:
        p=(d+(d[::-1]+d)*L)[:L]
        z.append(p)
    return z
    pass
def pad_constant_central(x,v):
    """
     Takes one 3-dimensional array with the constant value of padding.
     Your task is to pad the instances with the given constant value while maintaining the array at the center of the padding.
    :return:
    """
    L=max(map(len,x))
    M = [v]*L
    Len = max(map(len,x))
    z=[]
    for d in x:
        p=(M+d+M)[Len- np.math.ceil((Len-len(d))/2) :][:Len]
        z.append(p)
    return z
    pass

#######################################
#PyTorch
#######################################

# numpy&torch

def numpy2tensor():
    """
    Takes a numpy ndarray and converts it to a PyTorch tensor.
    Function torch.tensor is one of the simple ways to implement it but please do not use it this time.
    :return:
    """
    return torch.from_numpy(array)
    pass
def tensor2numpy():
    """
    Takes a PyTorch tensor and converts it to a numpy ndarray.
    :return:
    """
    return array.numpy()
    pass

#Tensor Sum-products

def Tensor_Sumproducts():
    """
    you are to implement the function tensor sumproducts that takes two tensors as input.
    returns the sum of the element-wise products of the two tensors.
    :return:
    """
    return torch.dot(A,B)
    pass

#Tensor ReLu and ReLu prime

def Tensor_Relu():
    """
    Takes one 2-dimensional tensor and apply the relu function on all the values of the tensor.
    :return:
    """
    return torch.max(torch,zeros(M.size()),M)
    pass
def Tensor_Relu_prime():
    """
    Takes one 2-dimensional tensor and apply the derivative of relu function on all the values of the tensor.
    :return:
    """
    return torch.clamp(M,min=0)*torch.reciprocal(torch.clamp(M,min=1e-8))
    pass
