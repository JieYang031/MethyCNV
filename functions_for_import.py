from __future__ import print_function
from __future__ import division


import random
import h5py as h5
from collections import OrderedDict
import numpy as np
import pandas as pd
import six
from six.moves import range
import sys
sys.path.insert(0, "/Users/yangj25/PycharmProjects/revised_code/")
import warnings
import os
from collections import OrderedDict
import gzip as gz
import threading
import re
from glob import glob
from keras.callbacks import Callback
from time import time
import subprocess

from keras import initializers as ki
from keras import callbacks as kcbk
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras import layers as kl
from keras import regularizers as kr
from keras import models as km
from keras.layers import concatenate



###################################################################################
####################### READ METHYLATION FILES #######################
def read_methyl_profile(filename, chromos=None, nb_sample=None, round=False,
                     sort=True, nb_sample_chromo=None):
    if is_bedgraph(filename):
        usecols = [0, 1, 3]
        skiprows = 1
    else:
        usecols = [0, 1, 2]
        skiprows = 0
    dtype = {usecols[0]: np.str, usecols[1]: np.int32, usecols[2]: np.float32}
    nrows = None
    if chromos is None and nb_sample_chromo is None:
        nrows = nb_sample
    d = pd.read_table(filename, header=None, comment='#', nrows=nrows,
                      usecols=usecols, dtype=dtype, skiprows=skiprows)
    d.columns = ['chromo', 'pos', 'signal']
    if np.any(d["pos"] <0 ):
        raise ValueError("position must be non-negative value!")
    d['chromo'] = format_chromo(d['chromo'])
    if chromos is not None:
        if not isinstance(chromos, list):
            chromos = [str(chromos)]
        d = d.loc[d.chromo.isin(chromos)]
        if len(d) == 0:
            raise ValueError('No data available for selected chromosomes!')
    if nb_sample_chromo is not None:
        d = sample_from_chromo(d, nb_sample_chromo)
    if nb_sample is not None:
        d = d.iloc[:nb_sample]
    if sort:
        d.sort_values(['chromo', 'pos'], inplace=True)
    return d

def read_methyl_profiles(filenames, log=None, *args, **kwargs):
    methyl_profiles = OrderedDict() 
    for filename in filenames:
        if filename.find("TCGA") >=0:
            if log:
                log(filename)
            methyl_file = GzipFile(filename, 'r') 
            output_name = split_ext(filename) 
            methyl_profile = read_methyl_profile(methyl_file, sort=True, *args, **kwargs) 
            methyl_profiles[output_name] = methyl_profile 
            methyl_file.close()
    return methyl_profiles 

def split_ext(filename):
    """Remove file extension from `filename`."""
    return os.path.basename(filename).split(os.extsep)[0] #return file name


class GzipFile(object):
    def __init__(self, filename, mode='r', *args, **kwargs):
        self.is_gzip = filename.endswith('.gz')
        if self.is_gzip:
            self.fh = gz.open(filename, mode, *args, **kwargs)
        else:
            self.fh = open(filename, mode, *args, **kwargs)

    def __iter__(self):
        return self.fh.__iter__()

    def __next__(self):
        return self.fh.__next__()

    def read(self, *args, **kwargs):
        return self.fh.read(*args, **kwargs)

    def readline(self, *args, **kwargs):
        return self.fh.readline(*args, **kwargs)

    def readlines(self, *args, **kwargs):
        return self.fh.readlines(*args, **kwargs)

    def write(self, data):
        if self.is_gzip and isinstance(data, str):
            data = data.encode()
        self.fh.write(data)

    def writelines(self, *args, **kwargs):
        self.fh.writelines(*args, **kwargs)

    def tell(self, *args, **kwargs):
        return self.fh.tell(*args, **kwargs)

    def seek(self, *args, **kwargs):
        self.fh.seek(*args, **kwargs)

    def closed(self):
        return self.fh.closed()

    def close(self):
        self.fh.close()

    def __iter__(self):
        self.fh.__iter__()

    def iter(self):
        self.fh.iter()


def is_bedgraph(filename):
    if isinstance(filename, str):
        with open(filename) as f:
            line = f.readline()
    else:
        pos = filename.tell()
        line = filename.readline()
        if isinstance(line, bytes):
            line = line.decode()
        filename.seek(pos)
    return re.match(r'track\s+type=bedGraph', line) is not None




class KnnMethylFeatureExtractor(object):

    def __init__(self, k=1):
        self.k = k

    def extract(self, x, y, ys):
        n = len(x)
        m = len(y)
        k = self.k
        kk = 2 * self.k
        yc = self.__larger_equal(x, y)
        knn_cpg = np.empty((n, kk), dtype=np.int32)
        knn_cpg.fill(np.nan)
        knn_dist = np.empty((n, kk), dtype=np.int32)
        knn_dist.fill(np.nan)

        for i in range(n):
            # Left side
            yl = yc[i] - k
            yr = yc[i] - 1
            if yr >= 0:
                xl = 0
                xr = k - 1
                if yl < 0:
                    xl += np.abs(yl)
                    yl = 0
                xr += 1
                yr += 1
                knn_cpg[i, xl:xr] = ys[yl:yr]
                knn_dist[i, xl:xr] = np.abs(y[yl:yr] - x[i])
            # Right side
            yl = yc[i]
            if yl >= m:
                continue
            if x[i] == y[yl]:
                yl += 1
                if yl >= m:
                    continue
            yr = yl + k - 1
            xl = 0
            xr = k - 1
            if yr >= m:
                xr -= yr - m + 1
                yr = m - 1
            xl += k
            xr += k + 1
            yr += 1
            knn_cpg[i, xl:xr] = ys[yl:yr]
            knn_dist[i, xl:xr] = np.abs(y[yl:yr] - x[i])

        return (knn_cpg, knn_dist)

    def __larger_equal(self, x, y):
        n = len(x)
        m = len(y)
        rv = np.empty(n, dtype=np.int)
        i = 0
        j = 0
        while i < n and j < m:
            while j < m and x[i] > y[j]:
                j += 1
            rv[i] = j
            i += 1
        if i < n:
            # x[i] > y[m - 1]
            rv[i:] = m
        return rv


###################################################################################
####################### READ CNV FILES #######################
def read_CNV_profile(filename, sort = True):
    usecols = [0, 1, 2, 3]
    skiprows = 1
    dtype = {usecols[0]: np.str, usecols[1]:np.int32, usecols[2]: np.int32, usecols[3]: np.int16}
    nrows = None
    d = pd.read_table(filename, header = None, nrows = nrows, usecols = usecols,
                      dtype = dtype, skiprows = skiprows)
    d.columns = ["chromo", "start", "end", "segment"]
    if np.any(d['segment'] < 0 ):
        raise ValueError("Segment need to be positive value!")
    if np.any(d['start'] < 0):
        raise ValueError("Start position need to be positive value!")
    if np.any(d['end'] < 0):
        raise ValueError("End position need to be positive value!")
    d_start = d[['chromo', 'start']]
    d_start.columns = ['chromo',"pos"]
    d_end = d[["chromo", "end"]]
    d_end.columns = ["chromo", "pos"]
    d = d_start.append(d_end)
    d['chromo'] = format_chromo(d['chromo'])
    if sort:
        d.sort_values(['chromo', "pos"], inplace = True)
    return d

def read_CNV_profiles(filenames, log = None, *args, **kwargs):
    CNV_profiles = OrderedDict()
    for filename in filenames:
        if log:
            log(filename)
        CNV_file = open(filename, 'r')
        output_name = split_ext(filename)
        num_lines = subprocess.check_output("wc -l " + CNV_file, shell = True)
        num_lines2 = str(num_lines).split("'")[1].split(" ")[0]
        if int(num_lines2) > 0:
            CNV_profile = read_CNV_profile(CNV_file, sort = True, *args, **kwargs)
            CNV_profiles[output_name] = CNV_profile
            CNV_file.close()
        else:
            print(CNV_file + " is empty, skip it.")
    return CNV_profiles


def map_values(pos, target_pos):
    assert np.all(pos == np.sort(pos))
    assert np.all(target_pos == np.sort(target_pos))
    pos = pos.ravel()
    target_pos = target_pos.ravel()
    idx = np.in1d(pos, target_pos)
    pos = pos[idx]
    target_states = np.empty(len(target_pos), dtype = 'int8')
    target_states.fill(0)
    idx = np.in1d(target_pos, pos).nonzero()[0]
    assert len(idx) == len(pos)
    assert np.all(target_pos[idx] == pos)
    target_states[idx] = 1
    return target_states


###################################################################################
####################### READ FASTA INFORMATION  #######################
def format_chromo(chromo):
    return chromo.str.upper().str.replace('^CHR', '')

def read_chromo(filenames, chromo):
    if chromo == "23":
        chromo = "X"
    elif chromo == "24":
        chromo = "Y"
    else:
        chromo = chromo
    filename = select_file_by_chromo(filenames, chromo)
    if not filename:
        raise ValueError('DNA file for chromosome "%s" not found!' % chromo)
    fasta_seqs = read_file(filename)
    if len(fasta_seqs) != 1:
        raise ValueError('Single sequence expected in file "%s"!' % filename)
    return fasta_seqs[0].seq

def select_file_by_chromo(filenames, chromo):
    filenames = to_list(filenames)
    if len(filenames) == 1 and os.path.isdir(filenames[0]):
        filenames = glob(os.path.join(filenames[0],
                                      '*.dna.chromosome.%s.fa*' % chromo))

    for filename in filenames:
        if filename.find('chromosome.%s.fa' % chromo) >= 0:
            return filename

def read_file(filename, gzip=None):
    #print("Check file name")
    print(filename)
    list
    if gzip is None:
        gzip = filename.endswith('.gz')
    if gzip:
        lines = gz.open(filename, 'r').read().decode()
    else:
        lines = open(filename, 'r').read()
    lines = lines.splitlines()
    return parse_lines(lines)

def parse_lines(lines):
    seqs = []
    seq = None
    start = None
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if len(line) > 0]
    for i in range(len(lines)):
        if lines[i][0] == '>':
            if start is not None:
                head = lines[start]
                seq = ''.join(lines[start + 1: i])
                seqs.append(FastaSeq(head, seq))
            start = i
    if start is not None:
        head = lines[start]
        seq = ''.join(lines[start + 1:])
        seqs.append(FastaSeq(head, seq))
    return seqs

class FastaSeq(object):
    def __init__(self, head, seq):
        self.head = head
        self.seq = seq



def extract_seq_windows(seq, pos, wlen, seq_index=1, assert_cpg=False):
    delta = wlen // 2
    nb_win = len(pos)
    seq = seq.upper() #change to upper case
    seq_wins = np.zeros((nb_win, wlen), dtype='int8')

    for i in range(nb_win):
        p = pos[i] - seq_index
        if p < 0 or p >= len(seq):
            raise ValueError('Position %d not on chromosome!' % (p + seq_index))
        win = seq[max(0, p - delta): min(len(seq), p + delta + 1)]
        if len(win) < wlen:
            win = max(0, delta - p) * 'N' + win #add NNN to seq
            win += max(0, p + delta + 1 - len(seq)) * 'N' 
            assert len(win) == wlen #assert: used to catch bugs
        seq_wins[i] = char_to_int(win) 
    idx = seq_wins == CHAR_TO_INT['N'] 
    seq_wins[idx] = np.random.randint(0, 4, idx.sum())
    assert seq_wins.max() < 4 
    if assert_cpg:
        assert np.all(seq_wins[:, delta] == 3) 
        assert np.all(seq_wins[:, delta + 1] == 2)
    return seq_wins

def char_to_int(seq):
    return [CHAR_TO_INT[x] for x in seq.upper()]

CHAR_TO_INT = OrderedDict([('A', 0), ('T', 1), ('G', 2), ('C', 3), ('N', 4)])
###################################################################################
####################### UTILS #######################

def to_list(value):
    if not isinstance(value, list) and value is not None:
        value = [value]
    return value


def make_dir(dirname):
    if os.path.exists(dirname):
        return False
    else:
        os.makedirs(dirname)
        return True

###################################################################################metrics
####################### SET UP METRICS #######################

def contingency_table(y, z):
    """Compute contingency table."""
    y = K.round(y)
    z = K.round(z)

    def count_matches(a, b):
        tmp = K.concatenate([a, b])
        return K.sum(K.cast(K.all(tmp, -1), K.floatx()))

    ones = K.ones_like(y)
    zeros = K.zeros_like(y)
    y_ones = K.equal(y, ones)
    y_zeros = K.equal(y, zeros)
    z_ones = K.equal(z, ones)
    z_zeros = K.equal(z, zeros)

    tp = count_matches(y_ones, z_ones)
    tn = count_matches(y_zeros, z_zeros)
    fp = count_matches(y_zeros, z_ones)
    fn = count_matches(y_ones, z_zeros)

    return (tp, tn, fp, fn)


def tpr(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return tp / (tp + fn)

def prec(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return tp / (tp + fp)

def acc(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return (tp + tn)/(tp + tn + fp + fn)

###################################################################################models
####################### BUILD MODELS #######################
###add batch normalization
##add regularization
##dropout
##parameter initialization


relu_initializer = ki.VarianceScaling(scale = 2, mode = "fan_in", distribution = "normal", seed = None)
tanh_initializer = ki.VarianceScaling(scale = 1, mode = "fan_in", distribution = "normal", seed = None)
Xavier_initializer = ki.glorot_normal(seed = None)


####### dna_model models
def CnnL1h8f100(input,  l2_decay, dropout):
    dna_x = kl.Conv1D(8, 100,
                      kernel_initializer = relu_initializer,
                      kernel_regularizer = kr.l2(l2_decay))(input)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 4)(dna_x)

    dna_x = kl.Flatten()(dna_x)

    dna_x = kl.Dense(units = 8,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)

    dna_x = kl.Dropout(dropout)(dna_x)
    return dna_x

def CnnL1h8f500(input,  l2_decay, dropout):
    dna_x = kl.Conv1D(8, 500,
                      kernel_initializer = relu_initializer,
                      kernel_regularizer = kr.l2(l2_decay))(input)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 4)(dna_x)

    dna_x = kl.Flatten()(dna_x)

    dna_x = kl.Dense(units = 8,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)

    dna_x = kl.Dropout(dropout)(dna_x)
    return dna_x

def CnnL1h32f50(input,  l2_decay, dropout):
    dna_x = kl.Conv1D(32, 50,
                      kernel_initializer = relu_initializer,
                      kernel_regularizer = kr.l2(l2_decay))(input)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 4)(dna_x)

    dna_x = kl.Flatten()(dna_x)

    dna_x = kl.Dense(units = 32,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)

    dna_x = kl.Dropout(dropout)(dna_x)
    return dna_x

def CnnL1h32f100(input,  l2_decay, dropout):
    dna_x = kl.Conv1D(32, 100,
                      kernel_initializer = relu_initializer,
                      kernel_regularizer = kr.l2(l2_decay))(input)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 4)(dna_x)

    dna_x = kl.Flatten()(dna_x)

    dna_x = kl.Dense(units = 32,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)

    dna_x = kl.Dropout(dropout)(dna_x)
    return dna_x

def CnnL1h64f50(input,  l2_decay, dropout):
    dna_x = kl.Conv1D(64, 50,
                      kernel_initializer = relu_initializer,
                      kernel_regularizer = kr.l2(l2_decay))(input)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 4)(dna_x)

    dna_x = kl.Flatten()(dna_x)

    dna_x = kl.Dense(units = 64,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)

    dna_x = kl.Dropout(dropout)(dna_x)
    return dna_x

def CnnL1h128f100(input,  l2_decay, dropout):
    dna_x = kl.Conv1D(128, 100,
                      kernel_initializer = relu_initializer,
                      kernel_regularizer = kr.l2(l2_decay))(input)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 4)(dna_x)

    dna_x = kl.Flatten()(dna_x)

    dna_x = kl.Dense(units = 128,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)

    dna_x = kl.Dropout(dropout)(dna_x)
    return dna_x

def CnnL1h256f10(input,  l2_decay, dropout):
    dna_x = kl.Conv1D(256, 10,
                      kernel_initializer = relu_initializer,
                      kernel_regularizer = kr.l2(l2_decay))(input)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 4)(dna_x)

    dna_x = kl.Flatten()(dna_x)

    dna_x = kl.Dense(units = 256,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)

    dna_x = kl.Dropout(dropout)(dna_x)
    return dna_x

def CnnL1h256f200(input,  l2_decay, dropout):
    dna_x = kl.Conv1D(256, 200,
                      kernel_initializer = relu_initializer,
                      kernel_regularizer = kr.l2(l2_decay))(input)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 4)(dna_x)

    dna_x = kl.Flatten()(dna_x)

    dna_x = kl.Dense(units = 256,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)

    dna_x = kl.Dropout(dropout)(dna_x)
    return dna_x

def CnnL1h256f500(input,  l2_decay, dropout):
    dna_x = kl.Conv1D(256, 500,
                      kernel_initializer = relu_initializer,
                      kernel_regularizer = kr.l2(l2_decay))(input)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 4)(dna_x)

    dna_x = kl.Flatten()(dna_x)

    dna_x = kl.Dense(units = 256,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)

    dna_x = kl.Dropout(dropout)(dna_x)
    return dna_x


def CnnL2h8f20(input,  l2_decay, dropout):
    dna_x = kl.Conv1D(8, 20,
                      kernel_initializer = relu_initializer,
                      kernel_regularizer = kr.l2(l2_decay))(input)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 4)(dna_x)
    
    dna_x = kl.Conv1D(8, 3,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 2)(dna_x)

    dna_x = kl.Flatten()(dna_x)
    
    dna_x = kl.Dense(units = 8,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.Dropout(dropout)(dna_x)

    return dna_x



def CnnL2h32f200(input,  l2_decay, dropout):
    dna_x = kl.Conv1D(32, 200,
                      kernel_initializer = relu_initializer,
                      kernel_regularizer = kr.l2(l2_decay))(input)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 4)(dna_x)
    
    dna_x = kl.Conv1D(32, 3,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 2)(dna_x)

    dna_x = kl.Flatten()(dna_x)
    
    dna_x = kl.Dense(units = 32,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.Dropout(dropout)(dna_x)

    return dna_x



def CnnL2h128f10(input,  l2_decay, dropout):
    dna_x = kl.Conv1D(128, 10,
                      kernel_initializer = relu_initializer,
                      kernel_regularizer = kr.l2(l2_decay))(input)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 4)(dna_x)
    
    dna_x = kl.Conv1D(128, 3,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 2)(dna_x)

    dna_x = kl.Flatten()(dna_x)
    
    dna_x = kl.Dense(units = 128,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.Dropout(dropout)(dna_x)

    return dna_x


def CnnL2h128f500(input,  l2_decay, dropout):
    dna_x = kl.Conv1D(128, 500,
                      kernel_initializer = relu_initializer,
                      kernel_regularizer = kr.l2(l2_decay))(input)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 4)(dna_x)
    
    dna_x = kl.Conv1D(128, 3,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 2)(dna_x)

    dna_x = kl.Flatten()(dna_x)
    
    dna_x = kl.Dense(units = 128,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.Dropout(dropout)(dna_x)

    return dna_x


def CnnL2h256f200(input,  l2_decay, dropout):
    dna_x = kl.Conv1D(256, 200,
                      kernel_initializer = relu_initializer,
                      kernel_regularizer = kr.l2(l2_decay))(input)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 4)(dna_x)
    
    dna_x = kl.Conv1D(256, 3,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 2)(dna_x)

    dna_x = kl.Flatten()(dna_x)
    
    dna_x = kl.Dense(units = 256,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.Dropout(dropout)(dna_x)

    return dna_x


def CnnL3h32f100(input,  l2_decay, dropout):
    dna_x = kl.Conv1D(32, 100,
                      kernel_initializer = relu_initializer,
                      kernel_regularizer = kr.l2(l2_decay))(input)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 4)(dna_x)
    
    dna_x = kl.Conv1D(32, 3,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 2)(dna_x)

    dna_x = kl.Conv1D(32, 3,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 2)(dna_x)

    dna_x = kl.Flatten()(dna_x)
    dna_x = kl.Dense(units = 32,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.Dropout(dropout)(dna_x)

    return dna_x


def CnnL3h32f100(input,  l2_decay, dropout):
    dna_x = kl.Conv1D(32, 500,
                      kernel_initializer = relu_initializer,
                      kernel_regularizer = kr.l2(l2_decay))(input)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 4)(dna_x)
    
    dna_x = kl.Conv1D(32, 3,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 2)(dna_x)

    dna_x = kl.Conv1D(32, 3,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 2)(dna_x)

    dna_x = kl.Flatten()(dna_x)
    dna_x = kl.Dense(units = 32,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.Dropout(dropout)(dna_x)

    return dna_x


def CnnL3h64f100(input,  l2_decay, dropout):
    dna_x = kl.Conv1D(64, 500,
                      kernel_initializer = relu_initializer,
                      kernel_regularizer = kr.l2(l2_decay))(input)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 4)(dna_x)
    
    dna_x = kl.Conv1D(64, 3,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 2)(dna_x)

    dna_x = kl.Conv1D(64, 3,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 2)(dna_x)

    dna_x = kl.Flatten()(dna_x)
    dna_x = kl.Dense(units = 64,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.Dropout(dropout)(dna_x)

    return dna_x


def CnnL3h128f20(input,  l2_decay, dropout):
    dna_x = kl.Conv1D(128, 20,
                      kernel_initializer = relu_initializer,
                      kernel_regularizer = kr.l2(l2_decay))(input)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 4)(dna_x)
    
    dna_x = kl.Conv1D(128, 3,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 2)(dna_x)

    dna_x = kl.Conv1D(128, 3,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 2)(dna_x)

    dna_x = kl.Flatten()(dna_x)
    dna_x = kl.Dense(units = 128,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.Dropout(dropout)(dna_x)

    return dna_x


def CnnL3h128f200(input,  l2_decay, dropout):
    dna_x = kl.Conv1D(128, 200,
                      kernel_initializer = relu_initializer,
                      kernel_regularizer = kr.l2(l2_decay))(input)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 4)(dna_x)
    
    dna_x = kl.Conv1D(128, 3,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 2)(dna_x)

    dna_x = kl.Conv1D(128, 3,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 2)(dna_x)

    dna_x = kl.Flatten()(dna_x)
    dna_x = kl.Dense(units = 128,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.Dropout(dropout)(dna_x)

    return dna_x


def CnnL3h128f500(input,  l2_decay, dropout):
    dna_x = kl.Conv1D(128, 500,
                      kernel_initializer = relu_initializer,
                      kernel_regularizer = kr.l2(l2_decay))(input)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 4)(dna_x)
    
    dna_x = kl.Conv1D(128, 3,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 2)(dna_x)

    dna_x = kl.Conv1D(128, 3,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.MaxPooling1D(pool_size = 2)(dna_x)

    dna_x = kl.Flatten()(dna_x)
    dna_x = kl.Dense(units = 128,
                     kernel_initializer = relu_initializer,
                     kernel_regularizer = kr.l2(l2_decay))(dna_x)
    dna_x = kl.BatchNormalization()(dna_x)
    dna_x = kl.Activation("relu")(dna_x)
    dna_x = kl.Dropout(dropout)(dna_x)

    return dna_x

####### methyl_model models

def GRUL1h256(input, l2_decay, dropout):
    shape = getattr(input, "_keras_shape")
    x_input = kl.Input(shape = shape[2:], name = "methyl/input")
    x_output = kl.Dense(256,
                       kernel_initializer = relu_initializer,
                       kernel_regularizer = kr.l2(l2_decay))(x_input)
    x_output = kl.BatchNormalization()(x_output)
    x_output = kl.Activation("relu")(x_output)
    replicate_model = km.Model(x_input, x_output)
    
    methyl_x = kl.TimeDistributed(replicate_model)(input)
    gru = kl.GRU(256, kernel_regularizer = kr.l2(l2_decay))
    methyl_x = kl.Bidirectional(gru)(methyl_x)
    methyl_x = kl.Dropout(dropout)(methyl_x)
    return methyl_x

def GRUL2h256(input, l2_decay, dropout):
    shape = getattr(input, "_keras_shape")
    x_input = kl.Input(shape = shape[2:], name = "methyl/input")
    x_output = kl.Dense(256,
                       kernel_initializer = relu_initializer,
                       kernel_regularizer = kr.l2(l2_decay))(x_input)
    x_output = kl.BatchNormalization()(x_output)
    x_output = kl.Activation("relu")(x_output)
    replicate_model = km.Model(x_input, x_output)
    
    methyl_x = kl.TimeDistributed(replicate_model)(input)
    gru = kl.GRU(128, kernel_regularizer = kr.l2(l2_decay), return_sequences = True)
    methyl_x = kl.Bidirectional(gru)(methyl_x)

    gru = kl.GRU(256, kernel_regularizer = kr.l2(l2_decay))
    methyl_x = kl.Bidirectional(gru)(methyl_x)
    methyl_x = kl.Dropout(dropout)(methyl_x)
    return methyl_x


def LSTML1h256(input, l2_decay, dropout):
    shape = getattr(input, "_keras_shape")
    x_input = kl.Input(shape = shape[2:], name = "methyl/input")
    x_output = kl.Dense(256,
                       kernel_initializer = relu_initializer,
                       kernel_regularizer = kr.l2(l2_decay))(x_input)
    x_output = kl.BatchNormalization()(x_output)
    x_output = kl.Activation("relu")(x_output)
    replicate_model = km.Model(x_input, x_output)
    
    methyl_x = kl.TimeDistributed(replicate_model)(input)
    lstm = kl.LSTM(256, kernel_regularizer = kr.l2(l2_decay))
    methyl_x = kl.Bidirectional(lstm)(methyl_x)
    methyl_x = kl.Dropout(dropout)(methyl_x)
    return methyl_x

def LSTML2h256(input, l2_decay, dropout):
    shape = getattr(input, "_keras_shape")
    x_input = kl.Input(shape = shape[2:], name = "methyl/input")
    x_output = kl.Dense(256,
                       kernel_initializer = relu_initializer,
                       kernel_regularizer = kr.l2(l2_decay))(x_input)
    x_output = kl.BatchNormalization()(x_output)
    x_output = kl.Activation("relu")(x_output)
    replicate_model = km.Model(x_input, x_output)
    
    methyl_x = kl.TimeDistributed(replicate_model)(input)
    lstm = kl.LSTM(128, kernel_regularizer = kr.l2(l2_decay), return_sequences = True)
    methyl_x = kl.Bidirectional(lstm)(methyl_x)

    lstm = kl.LSTM(256, kernel_regularizer = kr.l2(l2_decay))
    methyl_x = kl.Bidirectional(lstm)(methyl_x)
    methyl_x = kl.Dropout(dropout)(methyl_x)
    return methyl_x    


####### joint_model models 

def JointL1h512(input, l2_decay, dropout):
    joint_x = kl.Dense(512, 
                       kernel_initializer = relu_initializer,
                       kernel_regularizer = kr.l2(l2_decay))(input)
    joint_x = kl.BatchNormalization()(joint_x)
    joint_x = kl.Activation("relu")(joint_x)
    joint_x = kl.Dropout(dropout)(joint_x)
    joint_output = kl.Dense(1, activation = "sigmoid", name = "CNV")(joint_x)
    return joint_output

def JointL2h512(input, l2_decay, dropout):
    joint_x = kl.Dense(512, 
                       kernel_initializer = relu_initializer,
                       kernel_regularizer = kr.l2(l2_decay))(input)
    joint_x = kl.BatchNormalization()(joint_x)
    joint_x = kl.Activation("relu")(joint_x)
    joint_x = kl.Dropout(dropout)(joint_x)

    joint_x = kl.Dense(512,
                       kernel_initializer = relu_initializer,
                       kernel_regularizer = kr.l2(l2_decay))(joint_x)
    joint_x = kl.BatchNormalization()(joint_x)
    joint_x = kl.Activation("relu")(joint_x)
    joint_x = kl.Dropout(dropout)(joint_x)
    joint_output = kl.Dense(1, activation = "sigmoid", name = "CNV")(joint_x)
    return joint_output

def JointL3h512(input, l2_decay, dropout):
    joint_x = kl.Dense(512, 
                       kernel_initializer = relu_initializer,
                       kernel_regularizer = kr.l2(l2_decay))(input)
    joint_x = kl.BatchNormalization()(joint_x)
    joint_x = kl.Activation("relu")(joint_x)
    joint_x = kl.Dropout(dropout)(joint_x)

    joint_x = kl.Dense(512,
                       kernel_initializer = relu_initializer,
                       kernel_regularizer = kr.l2(l2_decay))(joint_x)
    joint_x = kl.BatchNormalization()(joint_x)
    joint_x = kl.Activation("relu")(joint_x)
    joint_x = kl.Dropout(dropout)(joint_x)    

    joint_x = kl.Dense(512,
                       kernel_initializer = relu_initializer,
                       kernel_regularizer = kr.l2(l2_decay))(joint_x)
    joint_x = kl.BatchNormalization()(joint_x)
    joint_x = kl.Activation("relu")(joint_x)
    joint_x = kl.Dropout(dropout)(joint_x)   
    joint_output = kl.Dense(1, activation = "sigmoid", name = "CNV")(joint_x)
    return joint_output


########################################################################################################## SET UP CALLBACKS #######################


class TrainingStopper(Callback):
    def __init__(self, max_time=None, stop_file=None,
                 verbose=1, logger=print):
        """max_time in seconds."""
        self.max_time = max_time
        self.stop_file = stop_file
        self.verbose = verbose
        self.logger = logger

    def on_train_begin(self, logs={}):
        self._time_start = time()

    def log(self, msg):
        if self.verbose:
            self.logger(msg)

    def on_epoch_end(self, batch, logs={}):
        if self.max_time is not None:
            elapsed = time() - self._time_start
            if elapsed > self.max_time:
                self.log('Stopping training after %.2fh' % (elapsed / 3600))
                self.model.stop_training = True

        if self.stop_file:
            if os.path.isfile(self.stop_file):
                self.log('Stopping training due to stop file!')
                self.model.stop_training = True


class PerformanceLogger(Callback):
    def __init__(self, metrics=['loss', 'acc'], log_freq=0.1,
                 precision=4, callbacks=[], verbose=bool, logger=print):
        self.metrics = metrics
        self.log_freq = log_freq
        self.precision = precision
        self.callbacks = callbacks
        self.verbose = verbose
        self.logger = logger
        self._line = '=' * 100
        self.epoch_logs = None
        self.val_epoch_logs = None
        self.batch_logs = []

    def _log(self, x):
        if self.logger:
            self.logger(x)

    def _init_logs(self, logs, train=True):
        logs = list(logs)
        # Select either only training or validation logs
        if train:
            logs = [log for log in logs if not log.startswith('val_')]
        else:
            logs = [log[4:] for log in logs if log.startswith('val_')]

        # `metrics` stores for each metric in self.metrics that exists in logs
        # the name for the metric itself, followed by all output metrics:
        #   metrics['acc'] = ['acc', 'output1_acc', 'output2_acc']
        metrics = OrderedDict()
        for name in self.metrics:
            if name in logs:
                metrics[name] = [name]
            output_logs = [log for log in logs if log.endswith('_' + name)]
            if len(output_logs):
                if name not in metrics:
                    # mean 'acc' does not exist in logs, but is added here to
                    # compute it later over all outputs with `_udpate_means`
                    metrics[name] = [name]
                metrics[name].extend(output_logs)

        # `logs_dict` stored the actual logs for each metric in `metrics`
        logs_dict = OrderedDict()
        # Show mean metrics first
        for mean_name in metrics:
            logs_dict[mean_name] = []
        # Followed by all output metrics
        for mean_name, names in six.iteritems(metrics):
            for name in names:
                logs_dict[name] = []

        return metrics, logs_dict

    def _update_means(self, logs, metrics):
        """Computes the mean over all outputs, if it does not exist yet."""

        for mean_name, names in six.iteritems(metrics):
            # Skip, if mean already exists, e.g. loss.
            if logs[mean_name][-1] is not None:
                continue
            mean = 0
            count = 0
            for name in names:
                if name in logs:
                    value = logs[name][-1]
                    if value is not None and not np.isnan(value):
                        mean += value
                        count += 1
            if count:
                mean /= count
            else:
                mean = np.nan
            logs[mean_name][-1] = mean

    def on_train_begin(self, logs={}):
        self._time_start = time()
        s = []
        s.append('Epochs: %d' % (self.params['epochs']))
        s = '\n'.join(s)
        self._log(s)

    def on_train_end(self, logs={}):
        self._log(self._line)

    def on_epoch_begin(self, epoch, logs={}):
        self._log(self._line)
        s = 'Epoch %d/%d' % (epoch + 1, self.params['epochs'])
        self._log(s)
        self._log(self._line)
        self._step = 0
        self._steps = self.params['steps']
        self._log_freq = int(np.ceil(self.log_freq * self._steps))
        self._batch_logs = None
        self._totals = None

    def on_epoch_end(self, epoch, logs={}):
        if self._batch_logs:
            self.batch_logs.append(self._batch_logs)

        if not self.epoch_logs:
            # Initialize epoch metrics and logs
            self._epoch_metrics, self.epoch_logs = self._init_logs(logs)
            tmp = self._init_logs(logs, False)
            self._val_epoch_metrics, self.val_epoch_logs = tmp

        # Add new epoch logs to logs table
        for metric, metric_logs in six.iteritems(self.epoch_logs):
            if metric in logs:
                metric_logs.append(logs[metric])
            else:
                # Add `None` if log value missing
                metric_logs.append(None)
        self._update_means(self.epoch_logs, self._epoch_metrics)

        # Add new validation epoch logs to logs table
        for metric, metric_logs in six.iteritems(self.val_epoch_logs):
            metric_val = 'val_' + metric
            if metric_val in logs:
                metric_logs.append(logs[metric_val])
            else:
                metric_logs.append(None)
        self._update_means(self.val_epoch_logs, self._val_epoch_metrics)

        # Show table
        table = OrderedDict()
        table['split'] = ['train']
        # Show mean logs first
        for mean_name in self._epoch_metrics:
            table[mean_name] = []
        # Show output logs
        if self.verbose:
            for mean_name, names in six.iteritems(self._epoch_metrics):
                for name in names:
                    table[name] = []
        for name, logs in six.iteritems(self.epoch_logs):
            if name in table:
                table[name].append(logs[-1])
        if self.val_epoch_logs:
            table['split'].append('val')
            for name, logs in six.iteritems(self.val_epoch_logs):
                if name in table:
                    table[name].append(logs[-1])
        self._log('')
        self._log(format_table(table, precision=self.precision))

        # Trigger callbacks
        for callback in self.callbacks:
            callback(epoch, self.epoch_logs, self.val_epoch_logs)

    def on_batch_end(self, batch, logs={}):
        self._step += 1
        batch_size = logs.get('size', 0)

        if not self._batch_logs:
            # Initialize batch metrics and logs table
            self._batch_metrics, self._batch_logs = self._init_logs(logs.keys())
            # Sum of logs up to the current batch
            self._totals = OrderedDict()
            # Number of samples up to the current batch
            self._nb_totals = OrderedDict()
            for name in self._batch_logs:
                if name in logs:
                    self._totals[name] = 0
                    self._nb_totals[name] = 0

        for name, value in six.iteritems(logs):
            # Skip value if nan, which can occur if the batch size is small.
            if np.isnan(value):
                continue
            if name in self._totals:
                self._totals[name] += value * batch_size
                self._nb_totals[name] += batch_size

        # Compute the accumulative mean over logs and store it in `_batch_logs`.
        for name in self._batch_logs:
            if name in self._totals:
                if self._nb_totals[name]:
                    tmp = self._totals[name] / self._nb_totals[name]
                else:
                    tmp = np.nan
            else:
                tmp = None
            self._batch_logs[name].append(tmp)
        self._update_means(self._batch_logs, self._batch_metrics)

        # Show logs table at a certain frequency
        do_log = False
        if self._step % self._log_freq == 0:
            do_log = True
        do_log |= self._step == 1 or self._step == self._steps

        if do_log:
            table = OrderedDict()
            prog = self._step / self._steps
            prog *= 100
            precision = []
            table['done (%)'] = [prog]
            precision.append(1)
            table['time'] = [(time() - self._time_start) / 60]
            precision.append(1)
            for mean_name in self._batch_metrics:
                table[mean_name] = []
            if self.verbose:
                for mean_name, names in six.iteritems(self._batch_metrics):
                    for name in names:
                        table[name] = []
                        precision.append(self.precision)
            for name, logs in six.iteritems(self._batch_logs):
                if name in table:
                    table[name].append(logs[-1])
                    precision.append(self.precision)

            self._log(format_table(table, precision=precision,
                                   header=self._step == 1))



def format_table(table, colwidth=None, precision=2, header=True, sep=' | '):
    col_names = list(table.keys())
    if not isinstance(precision, list):
        precision = [precision] * len(col_names)
    col_widths = []
    tot_width = 0
    nb_row = None
    ftable = OrderedDict()
    for col_idx, col_name in enumerate(col_names):
        width = max(len(col_name), precision[col_idx] + 2)
        values = []
        for value in table[col_name]:
            if value is None:
                value = ''
            elif isinstance(value, float):
                value = '{0:.{1}f}'.format(value, precision[col_idx])
            else:
                value = str(value)
            width = max(width, len(value))
            values.append(value)
        ftable[col_name] = values
        col_widths.append(width)
        if not nb_row:
            nb_row = len(values)
        else:
            nb_row = max(nb_row, len(values))
        tot_width += width
    tot_width += len(sep) * (len(col_widths) - 1)
    rows = []
    if header:
        rows.append(format_table_row(col_names, col_widths, sep=sep))
        rows.append('-' * tot_width)
    for row in range(nb_row):
        values = []
        for col_values in six.itervalues(ftable):
            if row < len(col_values):
                values.append(col_values[row])
            else:
                values.append(None)
        rows.append(format_table_row(values, col_widths, sep=sep))
    return '\n'.join(rows)


def format_table_row(values, widths=None, sep=' | '):
    """Format a row with `values` of a table."""
    if widths:
        _values = []
        for value, width in zip(values, widths):
            if value is None:
                value = ''
            _values.append('{0:>{1}s}'.format(value, width))
    return sep.join(_values)


###################################################################################template
####################### EXAMPLES #######################

###try to write the template
#python /Users/yangj25/PycharmProjects/revised_code/data_reading_generator.py  --write_h5_file  True  --dna_wlen 2001  --methyl_wlen 200  --out_dir  /Users/yangj25/PycharmProjects/deepcpg/test_data/dna2001_methyl200_p3   --methyl_profiles  /Users/yangj25/PycharmProjects/deepcpg/test_data/methyl/*.txt  --methyl_platform 450k   --CNV_profiles /Users/yangj25/PycharmProjects/deepcpg/test_data/CNV/*.txt  --dna_profile /Users/yangj25/PycharmProjects/MethyCNV/data/dna/GATK_B37_Human_ref/b37   --train_files  /Users/yangj25/PycharmProjects/deepcpg/test_data/dna2001_methyl200_p3/c{1,3,5}.h5  --val_files /Users/yangj25/PycharmProjects/deepcpg/test_data/dna2001_methyl200_p3/c{2,4,6}.h5  --batch_size 64   --steps_per_epoch_train  10 --steps_per_epoch_val  5  --dna_model Cnn1L128h11 --methyl_model GRUh256  --joint_model FC512  --dropout 0.2  --epochs  50  --filtered_CNV_dir  /Users/yangj25/PycharmProjects/deepcpg/test_data/filtered_CNV/  --NUM_PROBES 3


#############
####################### DATA STRUCTURE #######################

'''
###deepcpg old yield information
list(inputs)
['dna', 'methyl/state', 'methyl/dist']
inputs["dna"].shape
(128, 1001, 4)
inputs["methyl/state"].shape  ##in total five samples, each sample provide 128 BPs.
(128, 5, 100)
inputs["methyl/dist"].shape
(128, 5, 100)
list(outputs)
['CNV/TCGA-05-4396-01A-21D', 'CNV/TCGA-05-5429-01A-01D', 'CNV/TCGA-06-0125-01A-01D', 'CNV/TCGA-06-0125-02A-11D', 'CNV/TCGA-06-0152-02A-01D']
outputs["CNV/TCGA-05-4396-01A-21D"].shape
(128,)
list(weights)
['CNV/TCGA-05-4396-01A-21D', 'CNV/TCGA-05-5429-01A-01D', 'CNV/TCGA-06-0125-01A-01D', 'CNV/TCGA-06-0125-02A-11D', 'CNV/TCGA-06-0152-02A-01D']
weights["CNV/TCGA-05-4396-01A-21D"].shape
(128,)

###revised new yield information
list(inputs)
['dna', 'methyl/signal', 'methyl/dists']
inputs["dna"].shape
(128, 1001, 4)
inputs["methyl/signal"].shape
(128, 100)
inputs["methyl/dists"].shape
(128, 100)
outputs["CNV"].shape
(128,)
weights["CNV"].shape
(128,)
'''


