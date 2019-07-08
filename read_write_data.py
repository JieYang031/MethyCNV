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
from read_write_data_functions import *
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
            "--CNV_profiles",
            help = "the file contain the raw CNV information",
            nargs = "+"
        )

        w.add_argument(
            "--dna_profile",
            help = "the reference genome directory"
        )

        w.add_argument(
            "--NUM_PROBES",
            help = "The number of probes needs to be contained in segments.",
            type = int
        )

        w.add_argument(
            "--filtered_CNV_dir",
            help = "The directory where filtered CNV will be stored"
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

