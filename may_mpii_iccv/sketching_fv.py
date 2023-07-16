import torch
import numpy as np
import os

device = torch.device('cuda:0')

def sketch_batch2(inp, sd1, nSketchDimC, nFeatDimC, nIsFloatC):
    inp=inp.t()
    sd1_t = torch.from_numpy(sd1).to(device)
    sd1 = sd1_t
    # print('sd1: ', sd1.shape)
    # print(inp.shape)
    if nIsFloatC==1:
        inp=inp.type(torch.cuda.FloatTensor).to(device)
    else:
        inp=inp.type(torch.cuda.DoubleTensor).to(device)

    inp=inp.t()
    # print(inp.shape)
    c=nSketchDimC
    d=nFeatDimC
    out1 = inp.mm(sd1)

    res=out1
 
    if nIsFloatC!=1:
        res=res.type(torch.cuda.FloatTensor).to(device)

    # res=res.t()
    # print(res.shape)
    return res

def choose_h_sk_mat(nSketchDimC, nFeatDimC):
    nRep=int( np.ceil(nFeatDimC/nSketchDimC) )
    rand_array=np.array([]).astype(int)
    for i in range(nRep):
        rand_array_i=np.random.permutation(int(nSketchDimC))
        rand_array=np.concatenate( (rand_array, rand_array_i), axis=0 )

    return rand_array[0:nFeatDimC]

def choose_s_sk_mat(nSketchDimC, nFeatDimC):
    nRep=int( np.ceil(nFeatDimC/nSketchDimC) )
    rand_array=np.array([]).astype(int)
    for i in range(nRep):
        rand_array_i=np.array([-1, 1]).astype(int)
        rand_array=np.concatenate( (rand_array, rand_array_i), axis=0 )

    rand_array=np.random.permutation(rand_array)
    return rand_array[0:nFeatDimC]

def create_s_dense(hi,si):
    c = np.max(hi) + 1
    d = len(hi)
    out = np.zeros((d,c))  # in*out
    for i in range(d):
        out[i, hi[i]] = si[i]
    return out

