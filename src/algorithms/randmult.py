import time
import random
import numpy as np
import torch
import torch.utils
import threading
from copy import deepcopy
from time import time

import scipy
from scipy.special import softmax, log_softmax, rel_entr
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from concrete.ml.torch.compile import compile_brevitas_qat_model

from concrete import fhe
import brevitas.nn as qnn


import matplotlib.pyplot as plt

sm = nn.Softmax()

vocab_size = 27
num_files = 22
tbase ="../data/names/target/"
nbase ="../data/names/b6/"
words = open('../data/names/names.txt', 'r').read().splitlines()


dist_samples = int(vocab_size*1E2)
sim_samples = int(100)
sim_per_file = min(300,int(sim_samples/num_files)+1) #len(y_pred) 
fhe_samples = 10

logerr = 1E-6/vocab_size
rng = np.random.default_rng()

fi=1
target = np.load(tbase + f"target{fi}.npy")#[:sim_per_file]
y_pred = np.load(nbase + f"y_predsim{fi}.npy")#[:sim_per_file]
samples = sim_samples
for fi in range(2,num_files+1):
    target = np.concatenate((target,np.load(tbase + f"target{fi}.npy")))
    y_pred = np.concatenate((y_pred, np.load(nbase + f"y_predsim{fi}.npy")))

print(f"{tbase} {nbase} {vocab_size} {words[:10]}")


b_sim = True
b_run = True
compsize = 150 #max(min(10,len(y_pred)), int(len(y_pred)**0.5))
dist_samples = int(vocab_size*1E2)
sim_samples = 100 #1000 #len(y_pred) 
fhe_samples = 10

#Define Lambda
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}


def encrypted_cumsum(y_pred, rand, percision):
    for ni in range(1, vocab_size):
        y_pred[ni] = np.min(np.max(y_pred[ni]+3, 0), 6)
        y_pred[ni] += y_pred[ni-1]
    index = rand*y_pred[-1]
    one_encoded = np.less(y_pred*percision, index)
    return np.sum(one_encoded)

def encrypted_batchermult(y_pred, rand, ranks,  percision):
    args = y_pred
    #argmax
    k=0
    argcnt = percision-1 #4*((vocab_size+3)//4)
    #loop1 (pb)
    pb=1
    while pb<argcnt:
        #loop2 (pk)
        pk = pb
        while pk > 0:
            #loop3 (pj)
            pj = pk % pb;
            pj_max = (argcnt-1) - pk
            while pj < pj_max:
                #loop4 (pi)
                pi = 0
                pi_max = (pk - 1) if (2*(pk-1) < argcnt - pj) else argcnt - pj - (pk -1)
                while pi <= pi_max:
                    t5mp = (pi + pj) // (pb*2)
                    t6mp = (pi + pj + pk) // (pb*2)
                    if (t5mp == t6mp):
                        xind = pi + pj
                        yind = pi + pj + pk  
                        
                        #get encrypted values
                        enc_xval = y_pred[xind]
                        enc_xrank = ranks[xind]
                        enc_yval = y_pred[yind]
                        enc_yrank = ranks[yind]
                        #greater than for max to be index 0
                        enc_cond = enc_yval > enc_xval
                        #swap if enc_y > enc_x
                        y_pred[xind] = (enc_cond)*enc_yval + (1-enc_cond)*enc_xval
                        ranks[xind] = (enc_cond)*enc_yrank + (1-enc_cond)*enc_xrank
                        y_pred[yind] = (enc_cond)*enc_xval + (1-enc_cond)*enc_yval
                        ranks[yind] = (enc_cond)*enc_xrank + (1-enc_cond)*enc_yrank
                    pi+=1
                    #end loop4
                pj = pj + (2*pk)
                #end loop3
            pk = pk // 2
            #end loop2
        pb *=2
        #end loop1
    return y_pred,ranks

def encrypted_randmult128(y_pred, rand):
    y_pred = np.minimum(np.maximum(0, y_pred+3*scale), 6*scale)
    args = np.multiply(y_pred, rand)
    #argmax
    amax = np.arange(128)
    
    bvals = np.less(args[0:64], args[64:128])
    vmax = np.multiply(1-bvals, args[0:64]) + np.multiply(bvals, args[64:128])
    amax = np.multiply(1-bvals, amax[:64]) + np.multiply(bvals, amax[64:])

    bvals = np.less(vmax[:32], vmax[32:])
    vmax = np.multiply(1-bvals, vmax[0:32]) + np.multiply(bvals, vmax[32:64])
    amax = np.multiply(1-bvals, amax[:32]) + np.multiply(bvals, amax[32:])

 
    bvals = np.less(vmax[:16], vmax[16:])
    vmax = np.multiply(1-bvals, vmax[:16]) + np.multiply(bvals, vmax[16:])
    amax = np.multiply(1-bvals, amax[:16]) + np.multiply(bvals, amax[16:])

    
    bvals = np.less(vmax[:8], vmax[8:])
    vmax = np.multiply(1-bvals, vmax[:8]) + np.multiply(bvals, vmax[8:])
    amax = np.multiply(1-bvals, amax[:8]) + np.multiply(bvals, amax[8:])


    bvals = np.less(vmax[:4], vmax[4:])
    vmax = np.multiply(1-bvals, vmax[:4]) + np.multiply(bvals, vmax[4:])
    amax = np.multiply(1-bvals, amax[:4]) + np.multiply(bvals, amax[4:])

    bvals = np.less(vmax[:2], vmax[2:])
    vmax = np.multiply(1-bvals, vmax[:2]) + np.multiply(bvals, vmax[2:])
    amax = np.multiply(1-bvals, amax[:2]) + np.multiply(bvals, amax[2:])

 
    bvals = np.less(vmax[:1], vmax[1:])
    vmax = np.multiply(1-bvals, vmax[:1]) + np.multiply(bvals, vmax[1:])
    amax = np.multiply(1-bvals, amax[:1]) + np.multiply(bvals, amax[1:])
    return amax[0]


def encrypted_randmult(y_pred, rand):
    y_pred = np.minimum(np.maximum(0, y_pred+3*scale), 6*scale)
    args = np.multiply(y_pred, rand)
    #argmax
    argcnt = vocab_size
    amax = 0 
    vmax = args[0]
    for argi in range(1, argcnt):
        bval = np.less(vmax, args[argi])
        amax = argi*bval + amax*(1-bval)
        vmax = args[argi]*bval + vmax*(1-bval)
    return amax

    #for argi in range(argcnt-1):
        #dists.append(args[argi]  > args[argi+1:])
    #ranks = fhe.array([np.sum(dist) for dist in dists] + [0])
    #for argi in range(argcnt-1):
    #    ranks[argi+1:] += (1 - dists[argi+1:])
    #one_hot = np.equal(ranks, k)
    #for argi in range(argcnt):
    #   value -= one_hot[argi]
    #   result += value
    #return result #ranks

def run_circuit(my_circuit, my_ypred, my_randarr, my_target):
    homomorphic_evaluation = my_circuit.encrypt_run_decrypt(my_ypred.astype(int), my_randarr)
    print(f"Pred: {itos[homomorphic_evaluation]}, Target: {itos[my_target]}")

print("################### RandMulti Plus One  ##################################")
print(f"Compile Size: {compsize}")
bits = 3
scale = 1
for scale in range(11,12):
    st = time()
    compiler = fhe.Compiler(encrypted_randmult128, {"y_pred": "encrypted", "rand":"encrypted"})
    percision = 2**bits
    y_pred_round = (y_pred*scale).astype(int)
    
    inputset = []
    for isi in range(compsize):
        rand_arr = rng.random(size=vocab_size)  
        rand_round = np.concatenate([np.round(rand_arr*percision).astype(int), np.zeros(128-vocab_size).astype(int)])
        inputset.append((np.concatenate([y_pred_round[isi], np.zeros(128-vocab_size).astype(int)]), rand_round))
        if isi==0:
            encrypted_randmult(inputset[0][0], inputset[0][1])
    circuit = compiler.compile(inputset)
    print(f"Compile Time: {(time()-st)} for vector length {vocab_size}")
    st = time()
    
    if b_sim:    
        maxloss = 0
        sumloss = 0
        loss = nn.NLLLoss() #nn.CrossEntropyLoss()
        #print(f"Simulating")
        for hei in range(sim_samples):
            freq = np.zeros(vocab_size)
            sim_list = []
            for si in range(dist_samples):
                rand_dnuor = np.concatenate([np.round(rng.random(size=vocab_size)*percision).astype(int), np.zeros(128-vocab_size).astype(int)])
                #sim_data.append((y_pred[hei][:vocab_size], rand_dnuor))
                sim_list.append(circuit.simulate(np.concatenate([y_pred[hei], np.zeros(128-vocab_size)]), rand_dnuor))
            for si in range(len(sim_list)):
                freq[sim_list[si]] += 1 
            #print(circuit.simulate(y_pred[hei], rand_dnuor[si], percision))
            freq /= dist_samples
            altfreq = rng.multinomial(dist_samples, sm(torch.from_numpy(y_pred[hei]))) / dist_samples
            myloss = loss(torch.from_numpy(np.log(freq+logerr)),
                torch.from_numpy(np.array(target[hei])).long()).item()
            altloss = loss(torch.from_numpy(np.log(altfreq+logerr)),
                torch.from_numpy(np.array(target[hei])).long()).item()
            kl = np.average(rel_entr(freq+logerr, altfreq+logerr))
            if kl > maxloss:
                maxindex = hei
                if kl < 27:
                    maxloss = kl
                print(f"Index {hei}")
                print(f"Loss: {myloss} {altloss} {kl}")
                print(f"Freq {freq}")
                print(f"Sim {altfreq}")
            sumloss += myloss

        print(f"Bits: {bits} Scale: {scale} Loss: {sumloss / sim_samples}") 
        print(f"Avg Sim Time: {(time()-st)/(sim_samples*dist_samples)} for vector length {vocab_size}")
    st = time()

    if b_run:
        print(f"Running")
        print("Seed: ", end='')
        for hei in range(compsize-10, compsize):
            print(f"{itos[target[hei]]}", end='')
            #print(f"{target[hei]}", end='')
        print()
        tasks = []
        for hei in range(compsize, compsize+fhe_samples):
            rand_dnuor = np.concatenate([np.round(rng.random(size=vocab_size)*percision).astype(int), np.zeros(128-vocab_size).astype(int)])
            tasks.append(threading.Thread(target=run_circuit, args=(circuit, np.concatenate([y_pred_round[hei], np.zeros(128-vocab_size).astype(int)]), rand_dnuor, target[hei])))
        print("Starting Threads")
        ts = time()
        for t in tasks:
            t.start()
        for t in tasks:
            t.join()
        print(f"Bits {bits} Avg Time: {(time() - ts)/fhe_samples}")
    print()
