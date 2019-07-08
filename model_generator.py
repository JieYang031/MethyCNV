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
from model_functions import *
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

        w = p.add_argument_group("general inputs")

        w.add_argument(
            "--seed",
            help = "set random seed",
            default = 2019,
            type = int)
        
        w.add_argument(
            "--log_file",
            help = "write log messages to file")
        
        w.add_argument(
            "--verbose",
            help = "More detailed log messages",
            action = "store_true"
        )
        
        w.add_argument(
            "--model_dir",
            help = "Where the model and weights information saved"
        )

        c = p.add_argument_group("call backs argument")

        c.add_argument(
            "--max_time",
            help = "#maximum time we can run",
            type = float
        )

        c.add_argument(
            "--stop_file",
            help = "File that terminates training if it exists"
        )

        c.add_argument(
            '--no_log_outputs',
            help='Do not log performance metrics of individual outputs',
            action='store_true')
        
        c.add_argument(
            "--no_tensorboard",
            help = "do not store tensorboard summaries",
            action = "store_true"
        )

        c.add_argument(
            "--LOG_PRECISION",
            help = "the precision of log information",
            default = 4,
            type = int
        )

        f = p.add_argument_group("fitting models")
        
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
            "--dna_wlen",
            help = "dna window length",
            type = int
        )

        f.add_argument(
            "--methyl_wlen",
            help= "methylation model window length",
            type = int
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
            "--batch_size",
            help = "number of randomly samples in each batch run",
            default = 128,
            type = int
        )
        
        f.add_argument(
            "--epochs",
            help = "the number of epoches to run",
            default = 30,
            type = int
        )

        f.add_argument(
            "--methyl_max_dist",
            help = "assign a predefined value to the unknown or not exsiting methylation probe",
            default = 22228459,  #maybe too large
            type = int
        )
        
        f.add_argument(
            "--methyl_NAN",
            help = "missing value in methylation signal",
            default = -1,
            type = int
        )

        f.add_argument(
            "--loss_function",
            help = "the loss function used when updating models",
            default = "binary_crossentropy"
        )


        f.add_argument(
            "--early_stopping",
            help = "early stopping patience",
            type = int,
            default = 5
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
            "--dropout",
            help = "the percentage of dropout, reduce overfitting when training",
            default = 0.1,
            type = float
        )

        f.add_argument(
            "--adam_lr",
            help = "the learning rate of adam optimizer",
            default = 0.00001,
            type = float
        )

        return p


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

