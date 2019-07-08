
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

##################################################################################
############################# filter segments #####################################

def filter_segments(input_dir, output_dir, chr_pos_file, NUM_PROBES):

    chr_pos = pd.read_table(chr_pos_file, header = None, comment = "#", usecols = [0,1], dtype = {0: np.str, 1:np.int32}, skiprows = 0)
    chr_pos.columns = ["chr", "pos"]

    if os.path.exists(output_dir):
        print("The output directory already existed.")
    else:
        os.mkdir(output_dir)
        print("Create the output directory.")
    
    for files in os.listdir(input_dir):
        if files.find("TCGA") >= 0 :
            print("Processing " + str(files))
            CNV = pd.read_table(str(input_dir) + files, header = 0, comment = "#", usecols = [0, 1, 2, 5], dtype = {0:str, 1:np.int32, 2:np.int32, 3:np.int16}, skiprows = 0)
            Filter_CNV = [] 
            sampleName = files.split(".")[0]
            for chromo in CNV.CHR.unique():
                #print(chromo)
                idx = CNV.CHR == chromo
                CNV_chrome = CNV.loc[idx]
                idx2 = chr_pos.chr == chromo
                probes = chr_pos.loc[idx2].pos.values
                for index, row in CNV_chrome.iterrows():
                    cond1 = probes > row.START
                    cond1 = cond1.reshape( (len(cond1), 1) )
                    cond2 = probes < row.END
                    cond2 = cond2.reshape( (len(cond2), 1) )
                    cond3 = np.concatenate([cond1, cond2], axis = 1)
                    cond4 = cond3[ (cond3[:,0] == True) & (cond3[:,1] == True) ]
                    if cond4.shape[0] >= NUM_PROBES:
                        #print("ADD: Segment " + str(row.START) + "-" + str(row.END) + " has " + str(cond4.shape[0]) + " probes.")
                        Filter_CNV.append('{}\t{}\t{}\t{}'.format(row.CHR, row.START, row.END, row.CN))
                        
            with open(str(output_dir) + str(sampleName) + ".filtered.CNV.txt", 'w') as fw:
                for item in Filter_CNV:
                    fw.write("%s\n" % item)


##################################################################################
############################# summarize the BP #####################################

def summarize_BP(input_dir):
    CNV = []
    for files in os.listdir(input_dir):
        if files.find("TCGA") >= 0:
            print("Reading CNV from " + files)
            ##checking file to make sure it is not empty
            num_lines = subprocess.check_output("wc -l " + input_dir + str(files), shell = True)
            num_lines2 = str(num_lines).split("'")[1].split(" ")[0]
            if int(num_lines2) > 0:
                data = pd.read_table(input_dir + str(files), header = None, comment = "#", usecols = [0, 1, 2], dtype = {0: np.str, 1: np.int32, 2: np.int32}, skiprows = 0)
                start = data.iloc[:,[0, 1]]
                start.columns = ("CHR", "POS")
                end = data.iloc[:,[0, 2]]
                end.columns = ("CHR", "POS")
                new = pd.concat([start, end])
                new["CHR"] = new["CHR"].replace("X", "23")
                new["CHR"] = new["CHR"].replace("Y", "24")
                CNV.append(new)
            else:
                print(files + " is empty, skip it.")
    ##combine all rows
    CNV2 = pd.concat(CNV, axis = 0)
    CNV2 = CNV2.drop_duplicates()
    CNV2["CHR"] = CNV2["CHR"].astype("int32")
    CNV3 = CNV2.sort_values(["CHR", "POS"], ascending = [True, True])
    CNV3["CHR"] = CNV3["CHR"].astype("str")
    return(CNV3)
     
