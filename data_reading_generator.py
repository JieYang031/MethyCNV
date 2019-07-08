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


from keras import callbacks as kcbk
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras import layers as kl
from keras import regularizers as kr
from keras import models as km
from keras.layers import concatenate

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
############################# define generator #####################################

def generator(data_files, 
                  batch_size, steps_per_epoch, methyl_max_dist,
                  methyl_NAN, dna_wlen, methyl_wlen, loop = True, shuffle = True):
        
    def _prepro_dna(dna):
        if dna_wlen:
            cur_wlen = dna.shape[1]
            center = cur_wlen // 2
            delta = dna_wlen // 2
            dna = dna[:, (center - delta):(center + delta + 1)]
        return int_to_onehot(dna)

    def int_to_onehot(seqs, dim=4):
        seqs = np.atleast_2d(np.asarray(seqs))
        n = seqs.shape[0]
        l = seqs.shape[1]
        enc_seqs = np.zeros((n, l, dim), dtype='int8')
        for i in range(dim):
            t = seqs == i
            enc_seqs[t, i] = 1
        return enc_seqs

    data_files = list(to_list(data_files))
    file_idx = 0
    while True:
        if shuffle and file_idx == 0:
            np.random.shuffle(data_files)
        #print("Reading ", data_files[file_idx], "######")
        h5_file = h5.File(data_files[file_idx], 'r')
        CNV_pos_index = dict()
        CNV_neg_index = dict()
        for name in list(h5_file["outputs"]["CNV"]):
            CNV_pos_index[name] = np.where(h5_file["outputs"]["CNV"][name].value == 1)[0]
            CNV_neg_index[name] = np.where(h5_file["outputs"]["CNV"][name].value == 0)[0]
        for batch in range(steps_per_epoch):
            data_batch = dict()
            data_batch_methyl = []
            data_batch_dna = []
            data_batch_dist = []
            data_batch_CNV = []
            for i in range(int(batch_size/2)):
                #print("Generating the ", i, " sample in batch ", batch)
                #select pos_samples
                pos_sample = random.sample(list(CNV_pos_index), 1)[0]
                #print("Selected sample in ", i, " th:",pos_sample)
                #select index of BP
                while len(CNV_pos_index[pos_sample]) ==0:
                    pos_sample = random.sample(list(CNV_pos_index), 1)[0]
                pos_BP = random.sample(range(0, len(CNV_pos_index[pos_sample])), 1)[0]
                #print("Selected sample in ", i, " th:",pos_BP)
                pos_BP_index = CNV_pos_index[pos_sample][pos_BP]
                #print("Generate sample ", pos_sample, " BP_index ", pos_BP_index)
                pos_methyl = h5_file["inputs"]["methyl"][pos_sample]["signal"][pos_BP_index,]
                pos_dist = h5_file["inputs"]["dist"][pos_BP_index,]
                pos_dna = h5_file["inputs"]["dna"][pos_BP_index,]
                data_batch_methyl.append(np.expand_dims(pos_methyl, 1).T)
                data_batch_dist.append(np.expand_dims(pos_dist, 1).T)
                data_batch_dna.append(np.expand_dims(pos_dna,1).T)
                data_batch_CNV.append(1)
                #select neg_sample
                neg_sample = random.sample(list(CNV_neg_index),1)[0]
                neg_BP = random.sample(range(0, len(CNV_neg_index[neg_sample])), 1)[0]
                neg_BP_index = CNV_neg_index[neg_sample][neg_BP]
                #print("Generate sample ", neg_sample, " BP_index ", neg_BP_index)
                neg_methyl = h5_file["inputs"]["methyl"][neg_sample]["signal"][neg_BP_index,]
                neg_dist = h5_file["inputs"]["dist"][neg_BP_index,]
                neg_dna = h5_file["inputs"]["dna"][neg_BP_index,]
                data_batch_methyl.append(np.expand_dims(neg_methyl,1).T)
                data_batch_dist.append(np.expand_dims(neg_dist, 1).T)
                data_batch_dna.append(np.expand_dims(neg_dna,1).T)
                data_batch_CNV.append(0)
            data_batch_dna = np.concatenate(data_batch_dna)
            data_batch_methyl = np.concatenate(data_batch_methyl)
            data_batch_dist = np.concatenate(data_batch_dist)
            data_batch_CNV = np.asarray(data_batch_CNV)
            inputs = dict()
            inputs["dna"] = _prepro_dna(data_batch_dna)
            methyl = data_batch_methyl
            dist = data_batch_dist
            nan = methyl == methyl_NAN
            if np.any(nan):
                methyl[nan] = 5000
                dist[nan] = methyl_max_dist
            dist = np.minimum(dist, methyl_max_dist)/methyl_max_dist
            ##reshape methyl and dist
            methyl = np.reshape(methyl, (methyl.shape[0], methyl.shape[1], 1))
            dist = np.reshape(dist, (dist.shape[0], dist.shape[1], 1))
            inputs["methyl/signal"] = np.reshape(methyl, (methyl.shape[0], methyl.shape[1], 1))
            inputs["methyl/dists"] = np.reshape(dist, (dist.shape[0], dist.shape[1], 1))
            outputs = dict()
            #weights = dict()
            outputs["CNV"] = data_batch_CNV
            yield (inputs, outputs)

        h5_file.close()
        file_idx += 1
        if  file_idx == len(data_files):
            print("Used one round of files")
            if loop:
                file_idx = 0
                print("file_idx has been reset as 0")
            else:
                break


def perf_logs_str(logs):
    t = logs.to_csv(None, sep='\t', float_format='%.4f', index=False)
    return t


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



##################################################################################
############################# define callbacks #####################################
    def get_callbacks(self):
        opts = self.opts
        callbacks = [] #define callbacks arguments

        if opts.val_files:
            callbacks.append(kcbk.EarlyStopping( 
                'val_loss' if opts.val_files else 'loss', 
                patience=opts.early_stopping, 
                verbose=1 
            ))

        callbacks.append(kcbk.ModelCheckpoint( 
            os.path.join(opts.model_dir, 'model_weights_train.h5'),  
            save_best_only=False))
        
        monitor = 'val_loss' if opts.val_files else 'loss'
        
        callbacks.append(kcbk.ModelCheckpoint(
            os.path.join(opts.model_dir, 'model_weights_val.h5'),
            monitor=monitor,  #quantity to moniter
            save_best_only=True, verbose=1  # latest best model according to the quantity monitored will not be overwritten.
        ))

        max_time = int(opts.max_time * 3600) if opts.max_time else None
        callbacks.append(TrainingStopper( #Stop training after certain time or when file is detected.
            max_time=max_time, #Maximum training time in seconds
            stop_file=opts.stop_file, #Name of stop file that triggers the end of training when existing
            verbose=1 #If `True`, log message when training is stopped.
        ))

        def learning_rate_schedule(epoch): #calcualte learning rate schedule by input rate and decay rate
                lr = opts.learning_rate * opts.learning_rate_decay**epoch
                print('Learning rate: %.3g' % lr)
                return lr

        callbacks.append(kcbk.LearningRateScheduler(learning_rate_schedule)) 

        def save_lc(epoch, epoch_logs, val_epoch_logs):
            logs = {'lc_train.tsv': epoch_logs,
                    'lc_val.tsv': val_epoch_logs}
            for name, logs in six.iteritems(logs): #Returns an iterator over dictionary‘s items.
                if not logs:
                    continue
                logs = pd.DataFrame(logs)
                with open(os.path.join(opts.model_dir, name), 'w') as f:
                    f.write(perf_logs_str(logs))

        metrics = OrderedDict()
        for metric_funs in six.itervalues(self.metrics): #Returns an iterator over dictionary‘s values.
            for metric_fun in metric_funs:
                metrics[metric_fun.__name__] = True
        metrics = ['loss'] + list(metrics.keys())

        self.perf_logger = PerformanceLogger( 
            callbacks=[save_lc], 
            metrics=metrics, 
            precision=opts.LOG_PRECISION, 
            verbose=not opts.no_log_outputs
        )
        callbacks.append(self.perf_logger)

        if K._BACKEND == 'tensorflow' and not opts.no_tensorboard:
            callbacks.append(kcbk.TensorBoard( 
                log_dir=opts.model_dir, 
                histogram_freq=0, 
                write_graph=True, 
                write_images=True 
            ))

        return callbacks



##################################################################################
############################# define model structure #####################################

    def main(self, name, opts):

        print("The process start...")
        np.random.seed(opts.seed)
        logging.basicConfig(filename = opts.log_file,
                            format = '%(levelname)s (%(asctime)s): %(message)s')
        log = logging.getLogger(name)
        if opts.verbose:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)
        log.debug(opts)

        self.log = log
        self.opts = opts

        if opts.dna_wlen and opts.dna_wlen % 2 == 0:
            raise '--dna_wlen must be odd!'
        if opts.methyl_wlen and opts.methyl_wlen % 2 != 0:
            raise '--methyl_wlen must be even!'

        
        ##write the file 

        print(opts.write_h5_file)
        if opts.write_h5_file == "True":
            print("######################################")
            print("Start writing out")
            A = self.write_file_out()
            print("Ends writing out")
        elif opts.write_h5_file == "False":
            print("######################################")
            print("DO not write data")




####read the parameters out
        dna_model = opts.dna_model
        methyl_model = opts.methyl_model
        joint_model = opts.joint_model
        dna_wlen = opts.dna_wlen
        methyl_wlen = opts.methyl_wlen
        dropout = opts.dropout
        l1_decay = opts.l1_decay
        l2_decay = opts.l2_decay

######################################
#verify if the dna_wlen and methyl_wlen macthes to the input data files

        print("input file: ", opts.train_files)
        h5_file = h5.File(opts.train_files[0], "r")

        if h5_file["inputs"]["dna"].shape[1] != opts.dna_wlen:
            raise '--dna_wlen not match the train files'
        if h5_file["inputs"]["dist"].shape[1] != opts.methyl_wlen:
            raise '--methyl_wlen not match the train files'
        h5_file.close()

        if opts.val_files:
            h5_file = h5.File(opts.val_files[0], "r")
            if h5_file["inputs"]["dna"].shape[1] != opts.dna_wlen:
                raise '--dna_wlen not match the val files'
            if h5_file["inputs"]["dist"].shape[1] != opts.methyl_wlen:
                raise '--methyl_wlen not match the val files'
            h5_file.close()

######################################
        ##build dna model
        dna_input = kl.Input(shape = (dna_wlen, 4), name = "dna")
        if dna_model == "CnnL1h8f100":
            dna_x = CnnL1h8f100(input = dna_input, l2_decay = l2_decay, dropout = dropout)
        elif dna_model == "CnnL1h8f500":
            dna_x = CnnL1h8f500(input = dna_input, l2_decay = l2_decay, dropout = dropout)
        elif dna_model == "CnnL1h32f50":
            dna_x = CnnL1h32f50(input = dna_input, l2_decay = l2_decay, dropout = dropout)
        elif dna_model == "CnnL1h32f100":
            dna_x = CnnL1h32f100(input = dna_input, l2_decay = l2_decay, dropout = dropout)
        elif dna_model == "CnnL1h64f50":
            dna_x = CnnL1h64f50(input = dna_input, l2_decay = l2_decay, dropout = dropout)
        elif dna_model == "CnnL1h128f100":
            dna_x = CnnL1h128f100(input = dna_input, l2_decay = l2_decay, dropout = dropout)
        elif dna_model == "CnnL1h256f10":
            dna_x = CnnL1h256f10(input = dna_input, l2_decay = l2_decay, dropout = dropout)
        elif dna_model == "CnnL1h256f200":
            dna_x = CnnL1h256f200(input = dna_input, l2_decay = l2_decay, dropout = dropout)
        elif dna_model == "CnnL1h256f500":
            dna_x = CnnL1h256f500(input = dna_input, l2_decay = l2_decay, dropout = dropout)
        elif dna_model == "CnnL2h8f20":
            dna_x = CnnL2h8f20(input = dna_input, l2_decay = l2_decay, dropout = dropout)
        elif dna_model == "CnnL2h32f200":
            dna_x = CnnL2h32f200(input = dna_input, l2_decay = l2_decay, dropout = dropout)
        elif dna_model == "CnnL2h128f10":
            dna_x = CnnL2h128f10(input = dna_input, l2_decay = l2_decay, dropout = dropout)
        elif dna_model == "CnnL2h128f500":
            dna_x = CnnL2h128f500(input = dna_input, l2_decay = l2_decay, dropout = dropout)
        elif dna_model == "CnnL2h256f200":
            dna_x = CnnL2h256f200(input = dna_input, l2_decay = l2_decay, dropout = dropout)
        elif dna_model == "CnnL3h32f100":
            dna_x = CnnL3h32f100(input = dna_input, l2_decay = l2_decay, dropout = dropout)
        elif dna_model == "CnnL3h32f500":
            dna_x = CnnL3h32f500(input = dna_input, l2_decay = l2_decay, dropout = dropout)
        elif dna_model == "CnnL3h64f100":
            dna_x = CnnL3h64f100(input = dna_input, l2_decay = l2_decay, dropout = dropout)
        elif dna_model == "CnnL3h128f20":
            dna_x = CnnL3h128f20(input = dna_input, l2_decay = l2_decay, dropout = dropout)
        elif dna_model == "CnnL3h128f200":
            dna_x = CnnL3h128f200(input = dna_input, l2_decay = l2_decay, dropout = dropout)
        elif dna_model == "CnnL3h128f500":
            dna_x = CnnL3h128f500(input = dna_input, l2_decay = l2_decay, dropout = dropout)

        

        dna_output = kl.Dense(1, activation = "sigmoid", name = "CNV")(dna_x)
######################################     
        ''' 
        ##build methylation model
        methyl_input = kl.Input(shape = (methyl_wlen, 1), name = "methyl/signal")
        dist_input = kl.Input(shape = (methyl_wlen, 1), name = "methyl/dists")
        methyl_x = kl.concatenate([methyl_input, dist_input], axis = 2)


        if methyl_model == "GRUL1h256":
            methyl_x = GRUL1h256(input = methyl_x, l2_decay = l2_decay, dropout = dropout)
        elif methyl_model == "LSTML1h256":
            methyl_x = LSTML1h256(input = methyl_x, l2_decay = l2_decay, dropout = dropout)
        elif methyl_model == "GRUL2h256":
            methyl_x = GRUL2h256(input = methyl_x, l2_decay = l2_decay, dropout = dropout)
        elif methyl_model == "LSTML2h256":
            methyl_x = LSTML2h256(input = methyl_x, l2_decay = l2_decay, dropout = dropout)


######################################
        ##build joint model
        joint_x = kl.concatenate([dna_x, methyl_x])
        if joint_model == "JointL1h512":
            joint_output = JointL1h512(input = joint_x, l2_decay = l2_decay, dropout = dropout)
        elif joint_model == "JointL2h512":
            joint_output = JointL2h512(input = joint_x, l2_decay = l2_decay, dropout = dropout)
        elif joint_model == "JointL3h512":
            joint_output = JointL3h512(input = joint_x, l2_decay = l2_decay, dropout = dropout)

######################################
        ##form the model
        model = km.Model(inputs = [dna_input, methyl_input, dist_input],
                    outputs = [joint_output])
        '''
        
        model = km.Model(inputs = dna_input,
                         outputs = dna_output)

        print("Model strcuture:")
        print(model.summary())


        #set up the metrics
        self.metrics = dict()
        self.metrics["CNV"] = [tpr, prec, acc]
        
        #model compile
        print("Compiling model.")
        model.compile(optimizer = Adam(lr = opts.adam_lr), loss = opts.loss_function,metrics = self.metrics)

        ###create training data
        print("Generate train data:#####")
        print(opts.train_files)
        
        train_data = generator(data_files = opts.train_files, 
                               batch_size = opts.batch_size, 
                               steps_per_epoch = opts.steps_per_epoch_train, 
                               methyl_max_dist = opts.methyl_max_dist,
                               methyl_NAN = opts.methyl_NAN, 
                               dna_wlen = opts.dna_wlen, 
                               methyl_wlen = opts.methyl_wlen)
        
        ##create validation data
        if opts.val_files:
            print("Generate validation data:######")
            print(opts.val_files)
            ###maybe val_data should not use generator 
            val_data = generator(data_files = opts.val_files,
                                batch_size = opts.batch_size, 
                                steps_per_epoch = opts.steps_per_epoch_val, 
                                methyl_max_dist = opts.methyl_max_dist,
                                methyl_NAN = opts.methyl_NAN, 
                                dna_wlen = opts.dna_wlen, 
                                methyl_wlen = opts.methyl_wlen)
            validation_steps = opts.steps_per_epoch_val
        else:
            val_data = None
            validation_steps = None

        ###fitting model
        callbacks = self.get_callbacks()
        print("Training model...")
        model.fit_generator(generator = train_data, 
                            epochs = opts.epochs, 
                            callbacks = callbacks,
                            steps_per_epoch = opts.steps_per_epoch_train,
                            validation_data = val_data, 
                            validation_steps = validation_steps
                            )

        ###print out summary metrics
        print("\n Training set performance:")
        print(format_table(self.perf_logger.epoch_logs,
                        precision = opts.LOG_PRECISION))
        
        if self.perf_logger.val_epoch_logs:
            print('\nValidation set performance:')
            print(format_table(self.perf_logger.val_epoch_logs,
                               precision=opts.LOG_PRECISION))

        ##save model out
        filename = os.path.join(opts.model_dir, "model_weights_val.h5")

        model.save(os.path.join(opts.model_dir, "model.h5"))

        log.info("Done!")
        return 0


if __name__ == "__main__":
    app = App()
    app.run(sys.argv)



##steps_per_epoch: #batchs generated for each epoch. This will affect whether all input files will
#be used.
#for example, there are 5 data_files as input.
#if you want to generte 10 batches per data files, we need to set the steps_per_epoch = 5*10=50

