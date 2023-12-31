import time
import random
import numpy as np
import torch
import torch.utils
from copy import deepcopy
from time import time
import threading
import scipy
from scipy.special import softmax, log_softmax, kl_div, rel_entr
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torchvision.transforms import ToTensor, Lambda, Compose
import torchvision.datasets as datasets
from concrete.ml.torch.compile import compile_brevitas_qat_model

from concrete import fhe
import brevitas.nn as qnn
import matplotlib.pyplot as plt


ALG_NAME = ["Simulated Baseline", "RNG 1-10", "RNG 0-10", "CumSum"]

rng = np.random.default_rng()

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

fi=1
target = np.load(tbase + f"target{fi}.npy")#[:sim_per_file]
y_pred = np.load(nbase + f"y_predsim{fi}.npy")#[:sim_per_file]
samples = sim_samples
for fi in range(2,num_files+1):
    target = np.concatenate((target,np.load(tbase + f"target{fi}.npy")))
    y_pred = np.concatenate((y_pred, np.load(nbase + f"y_predsim{fi}.npy")))

print(f"{tbase} {nbase} {vocab_size} {words[:10]}")

b_sim = True #False #True
b_run = False

#Define Lambda
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(f"{target.shape}")
for i in range(100):
    print(f"{itos[target[i]]}", end ='')

def permutate(arr):
    perm = np.arange(len(arr))
    rng.shuffle(perm)
    shuffled = np.zeros(len(arr))
    for si in range(len(arr)):
        shuffled[si] = arr[perm[si]]
    return shuffled, perm

def encrypted_cumsum(y_pred, rand, perm, percision):
    y_pred = np.minimum(np.maximum(0, y_pred+1*scale),4*scale)
    for ni in range(1, vocab_size):
        y_pred[ni] += y_pred[ni-1]
    index = rand*y_pred[-1]
    one_encoded = np.less(y_pred*percision, index)

    result = perm[0]
    for ni in range(vocab_size-1):
        result = perm[ni+1]*one_encoded[ni] + result*(1-one_encoded[ni])
    return result

def run_circuit(my_circuit, my_ypred, my_randarr, percision, my_target):
    shuffled, perm = permutate(my_ypred)
    homomorphic_evaluation = my_circuit.encrypt_run_decrypt(shuffled.astype(int), my_randarr, perm.astype(int), percision)
    print(f"Pred: {itos[homomorphic_evaluation]}, Target: {itos[my_target]}")



#baseline
print("################### Baseline ##################################")
loss = nn.CrossEntropyLoss()
loss_net = loss(torch.from_numpy(y_pred[:sim_samples]), torch.from_numpy(target[:sim_samples]).long())
print(f"Real Baseline: {loss_net.item()}")
 
rng = np.random.default_rng()
if b_sim:    
    sumloss = 0
    loss = nn.NLLLoss() #nn.CrossEntropyLoss()
    sm = nn.Softmax()
    print(f"Simulating")
    for hei in range(sim_samples):
        #freq = np.zeros(vocab_size)
        freq = rng.multinomial(dist_samples, sm(torch.from_numpy(y_pred[hei]))) / dist_samples
        sumloss += loss(torch.from_numpy(np.log(freq+logerr)), 
                 torch.from_numpy(np.array(target[hei])).long()).item()
    print(f"Loss: {sumloss / sim_samples}") 

print("################### CumSum - Multiply  ##################################")
compsize = max(min(10,len(y_pred)), int(5*len(y_pred)**0.5))
rand_arr = rng.random(size=compsize+sim_samples+fhe_samples) 
print(f"Compile Size: {compsize}")
scale = 1
for bits in range(4,5):
    compiler = fhe.Compiler(encrypted_cumsum, {"y_pred": "encrypted", "rand": "clear", "perm": "clear", "percision": "clear"})
    percision = 2**bits
    y_pred_round = (y_pred*scale).astype(int)
    rand_round = np.round(rand_arr*percision-0.5).astype(int)
    
    print(f"Compiling")
    inputset = []
    for isi in range(compsize):
        shuffled, perm = permutate(y_pred_round[isi])
        inputset.append((shuffled.astype(int), rand_round[isi], perm.astype(int), percision))
    circuit = compiler.compile(inputset)
    
    if b_sim:    
        sumloss = 0
        loss = nn.NLLLoss() #nn.CrossEntropyLoss()
        print(f"Simulating")
        maxloss = 0
        for hei in tqdm(range(sim_samples)):
            print(y_pred[hei])
            freq = np.zeros(vocab_size)
            rand_dnuor = np.round(rng.random(size=dist_samples)*percision).astype(int)
            for si in range(dist_samples):
                shuffled, perm = permutate(y_pred[hei])
                freq[circuit.simulate(shuffled.astype(int), rand_dnuor[si], perm.astype(int),  percision)] += 1
            freq /= dist_samples
            #print(f"Freq {freq}")
            #print(f"Logerr {logerr}")
            #print(f"Target {target[hei]}")
            altfreq = rng.multinomial(dist_samples, sm(torch.from_numpy(y_pred[hei]))) / dist_samples
            myloss = loss(torch.from_numpy(np.log(freq+logerr)), 
                torch.from_numpy(np.array(target[hei])).long()).item()
            altloss = loss(torch.from_numpy(np.log(altfreq+logerr)), 
                torch.from_numpy(np.array(target[hei])).long()).item()
            kl = np.average(rel_entr(freq+logerr, altfreq+logerr))
            if kl > maxloss:
                maxindex = hei
                if kl < 27:
                    maxloss= kl
                print(f"Index {hei}")
                print(f"Loss: {myloss} {altloss} {kl}")
                print(f"Freq {freq}")
                print(f"Sim {altfreq}")
            sumloss += myloss
        print(f"Bits: {bits} Loss: {sumloss / sim_samples}") 

    if b_run:
        print(f"Running")
        print("Seed: ", end='')
        for hei in range(compsize-10, compsize):
            print(f"{itos[target[hei]]}", end='')
            #print(f"{target[hei]}", end='')
        print()
        tasks = []
        for hei in range(compsize, compsize+fhe_samples):
            tasks.append(threading.Thread(target=run_circuit, args=(circuit, y_pred_round[hei], rand_round[hei], percision, target[hei])))
        print("Starting Threads")
        ts = time()
        for t in tasks:
            t.start()
        for t in tasks:
            t.join()
        print(f"Bits {bits} Avg Time: {(time() - ts)/fhe_samples}")
