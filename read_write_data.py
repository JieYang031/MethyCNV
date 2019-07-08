##################################################################################
##################################################################################

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
from functions_for_import import *
import warnings
import os
import argparse
import logging
from collections import OrderedDict
import gzip
import threading
import re
import subprocess

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
            

##################################################################################
############################# set input arguments #####################################

class App(object):
    
    def run(self, args):
        name = os.path.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog = name,
            formatter_class = argparse.ArgumentDefaultsHelpFormatter,
            description = "read arguments")

        w = p.add_argument_group("write files")

        w.add_argument(
            "--write_h5_file",
            help = "wehther create new files based on raw data",
            type = str
        )
        w.add_argument(
            "--seed",
            help = "set random seed",
            default = 2019,
            type = int)
        
        w.add_argument(
            "--log_file",
            help = "write log messages to file")
        
        w.add_argument(
            "--dna_wlen",
            help = "dna window length",
            type = int
        )

        p.add_argument(
            "--methyl_wlen",
            help= "methylation model window length",
            type = int
        )

        w.add_argument(
            "--methyl_NAN",
            help = "missing value in methylation signal",
            default = -1,
            type = int
        )

        w.add_argument(
            "--out_dir",
            help = "output directory for the write out files"
        )

        w.add_argument(
            "--methyl_profiles",
            help = "the files which contain the methylation signal raw information",
            nargs = "+"
        )

        w.add_argument(
            "--methyl_platform",
            help = "the methylation platform",
            default = ["450k", "EPIC"]
        )

        w.add_argument(
            "--verbose",
            help = "More detailed log messages",
            action = "store_true"
        )

        w.add_argument(
            "--CNV_profiles",
            help = "the file contain the raw CNV information",
            nargs = "+"
        )

        w.add_argument(
            "--dna_profile",
            help = "the reference genome directory"
        )

        f = p.add_argument_group("fit model")

        f.add_argument(
            "--train_files",
            help = "the written h5 files used to train model",
            nargs = "+"
        )

        f.add_argument(
            "--val_files",
            help = "the written h5 files used to validate model",
            nargs = "+"
        )

        f.add_argument(
            "--batch_size",
            help = "number of randomly samples in each batch run",
            default = 128,
            type = int
        )

        f.add_argument(
            "--methyl_max_dist",
            help = "assign a predefined value to the unknown or not exsiting methylation probe",
            default = 22228459,  #maybe too large
            type = int
        )
        
        f.add_argument(
            "--loss_function",
            help = "the loss function used when updating models",
            default = "binary_crossentropy"
        )

        f.add_argument(
            "--epochs",
            help = "the number of epoches to run",
            default = 30,
            type = int
        )

        f.add_argument(
            "--steps_per_epoch_train",
            help = "the number of batches need to run per epoch in training process, recommended as  input #samples * #batches generated from each sample",
            type = int
        )

        f.add_argument(
            "--steps_per_epoch_val",
            help = "in validation set",
            type = int
        )

        f.add_argument(
            "--early_stopping",
            help = "early stopping patience",
            type = int,
            default = 5
        )

        f.add_argument(
            "--max_time",
            help = "#maximum time we can run",
            type = float
        )

        f.add_argument(
            "--learning_rate",
            help = "learning rate",
            default = 0.0001,
            type = float
        )

        f.add_argument(
            "--learning_rate_decay",
            help = "exponential leatning rate decay factor",
            type = float,
            default = 0.975
        )

        f.add_argument(
            "--l1_decay",
            help = "l1 learning rate",
            type = float,
            default = 0.0001
        )

        f.add_argument(
            "--l2_decay",
            help = "l2 learning rate",
            type = float,
            default = 0.0001
        )

        f.add_argument(
            "--lr_decay",
            help = "learning rate decay",
            default = 0.0001,
            type = float
        )

        f.add_argument(
            "--np_log_outputs",
            help = "Do not log performance metrics of indivisual outputs",
            action = "store_true"
        )

        f.add_argument(
            "--no_tensorboard",
            help = "do not store tensorboard summaries",
            action = "store_true"
        )

        f.add_argument(
            "--dropout",
            help = "the percentage of dropout, reduce overfitting when training",
            default = 0.1,
            type = float
        )

        f.add_argument(
            "--dna_model",
            help = "the name of the model strucutre to run for dna",
            type = str
        )

        f.add_argument(
            "--methyl_model",
            help = "the name of the model structure to run for methylation",
            type = str
        )

        f.add_argument(
            "--joint_model",
            help = "the name of the model structure to run for joint two models",
            type = str
        )

        f.add_argument(
            "--LOG_PRECISION",
            help = "the precision of log information",
            default = 4,
            type = int
        )

        f.add_argument(
            "--adam_lr",
            help = "the learning rate of adam optimizer",
            default = 0.00001,
            type = float
        )

        f.add_argument(
            "--stop_file",
            help = "File that terminates training if it exists"
        )

        f.add_argument(
            '--no_log_outputs',
            help='Do not log performance metrics of individual outputs',
            action='store_true')
        
        f.add_argument(
            "--NUM_PROBES",
            help = "The number of probes needs to be contained in segments.",
            type = int
        )

        f.add_argument(
            "--filtered_CNV_dir",
            help = "The directory where filtered CNV will be stored"
        )

        f.add_argument(
            "--model_dir",
            help = "Where the model and weights information saved"
        )

        return p

##################################################################################
############################# read data and write out #####################################

    def write_file_out(self):
        opts = self.opts
        log = self.log
        
        log.info('Reading Data Start!')
        
        make_dir(opts.out_dir)
        outputs = OrderedDict()

        #read methylation profiles
        if opts.methyl_profiles:
            log.info('Reading methylation profiles ...')
            outputs['methyl'] = read_methyl_profiles(opts.methyl_profiles,
                                                    log=log.info)

        print(opts.methyl_platform)
        if opts.methyl_platform == "450k":
            methyl_nb_probes = 485512
        elif opts.methyl_platform == "EPIC":
            methyl_nb_probes = 866836
        else:
            raise ValueError("Methylation platform need to be either 450k or EPIC!")

        if outputs["methyl"]:
            sample_list = list(outputs['methyl'])

        for sample in sample_list:
            assert outputs["methyl"][sample].shape[0] == methyl_nb_probes

        log.info("%d samples" % len(sample_list))

        ##############################
        ### filter segments and save out

        raw_CNV_dir = opts.CNV_profiles[0].split("TCGA")[0]
        chr_pos_file = opts.methyl_profiles[1]
        
        filter_segments(input_dir = raw_CNV_dir, output_dir = opts.filtered_CNV_dir,
        chr_pos_file = chr_pos_file, NUM_PROBES = opts.NUM_PROBES)


        ##############################
        ###generate BP profile
    
        bp_table = summarize_BP(input_dir = opts.filtered_CNV_dir)
        bp_table.columns = ["chromo", "pos"]
        bp_table = bp_table[bp_table.pos != 81195400]
        print("####write BP files out....")
        np.savetxt(str(opts.out_dir) + "/all_BP.txt", bp_table,
                    header = "",   fmt="%s\t%d")

        log.info('%d break points' % len(bp_table))

        ##############################
        ##read CNV profile
        if opts.filtered_CNV_dir:
            filtered_CNV_files = os.listdir(opts.filtered_CNV_dir)
            full_filtered_CNV_files = []
            for item in filtered_CNV_files:
                full_filtered_CNV_files.append(os.path.join(opts.filtered_CNV_dir, item))

            log.info("Reading CNV profiles...")
            outputs["CNV"] = read_CNV_profiles(
                filenames = full_filtered_CNV_files, log = log.info)

        ##############################
        ##prepare file for each chromosome
        for chromo in bp_table.chromo.unique():
            log.info('-' * 80)
            log.info('Chromosome %s ...' % (chromo))
            idx = bp_table.chromo == chromo
            chromo_pos = bp_table.loc[idx].pos.values
            #chromo_prob = bp_table.loc[idx].Prob.values  #new added each BP's probability
            chromo_outputs = OrderedDict()

            # Read DNA of chromosome
            chromo_dna = None
            if opts.dna_profile:
                if chromo == "23":
                    chromo = "X"
                elif chromo == "24":
                    chromo = "Y"
                chromo_dna = read_chromo(opts.dna_profile, chromo)

            # -------------------
            chunk_start = 0
            chunk_end = len(chromo_pos)
            chunk_pos = chromo_pos
            #chunk_prob = chromo_prob

            chunk_outputs = OrderedDict()
            methyl_tables = OrderedDict()

            # Create methyl_tables which store the methylation signal values for each sample
            
            for name, methyl_table in six.iteritems(outputs["methyl"]):
                methyl_table = methyl_table.loc[methyl_table.chromo == chromo]
                methyl_table = methyl_table.sort_values("pos")
                methyl_values = methyl_table.signal.values
                methyl_values = methyl_values.ravel()
                methyl_tables[name] = methyl_values
            chunk_outputs["methyl"] = methyl_tables
            chunk_outputs["methyl_mat"] = np.vstack(list(chunk_outputs["methyl"].values())).T
            filename = 'c%s.h5' % (chromo)
            filename = os.path.join(opts.out_dir, filename)
            print("###" + str(filename))
            chunk_file = h5.File(filename, 'w')
            
            # Write chromosome and positions information
            chunk_file.create_dataset('chromo', shape=(len(chunk_pos),),
                                        dtype='S2')
            #chunk_file['chromo'][:] = chromo.encode()
            chunk_file.create_dataset('pos', data=chunk_pos, dtype=np.int32)
            
            # Write CNV information out to group "outputs"
            if len(chunk_outputs):
                out_group = chunk_file.create_group('outputs')
                if opts.CNV_profiles:
                    CNV_group = out_group.create_group("CNV")
                    for name, CNV_table in six.iteritems(outputs["CNV"]):
                        CNV_table = CNV_table.loc[CNV_table.chromo == chromo]
                        CNV_table = CNV_table.sort_values("pos")
                        mapped_table = map_values(CNV_table.pos.values, chromo_pos)
                        CNV_group.create_dataset(name, data = mapped_table, compression = "gzip",
                                                    dtype = np.int32)
            # create group "inputs"
            in_group = chunk_file.create_group("inputs")
            
            # methylation probes neighours
            if opts.methyl_wlen:
                log.info('Extracting methylation probe around the targeted breakpoint ...')
                methyl_ext = KnnMethylFeatureExtractor(opts.methyl_wlen // 2)
                context_group = in_group.create_group('methyl')
            
            for name, methyl_table in six.iteritems(outputs['methyl']):
                methyl_table = methyl_table.loc[methyl_table.chromo == chromo]
                signal, dist = methyl_ext.extract(chunk_pos, methyl_table.pos.values,
                                                    methyl_table.signal.values)
                signal[signal < 0 ] = -1 #value are automatically set as np.nan
                dist[dist < 0] = -1
                assert len(dist) == len(chunk_pos)
                assert len(signal) == len(chunk_pos)
                assert np.all((dist >= 0) | (dist == -1))
                group = context_group.create_group(name)
                group.create_dataset("signal", data = signal, compression = "gzip")
            
            # store the 'dist' aside, because it is the same for all samples
            in_group.create_dataset("dist", data = dist, compression = "gzip")
            
            # DNA windows
            if chromo_dna:
                log.info('Extracting DNA sequence windows ...')
                dna_wins = extract_seq_windows(chromo_dna, pos=chunk_pos,
                                                wlen=opts.dna_wlen)
                assert len(dna_wins) == len(chunk_pos)
                in_group.create_dataset('dna', data=dna_wins, dtype=np.int32,
                                        compression='gzip')

            chunk_file.close()

        log.info('Reading Data Done!')
        return 0

