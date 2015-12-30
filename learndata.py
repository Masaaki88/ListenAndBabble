#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
learndata.py does the auditory learning with the BRIAN hears and the echo state
network (ESN).

Usage:
    learndata.py <n_vowel> [--n_samples=N_SAMPLES] [--n_training=N_TRAINING]
            [--n_channels=N_CHANNELS] [--trains_per_worker=TRAINS_PER_WORKER]
            [--subfolder=SUBFOLDER] [--n_reservoirs=N_RESERVOIRS]
            [--leak_rate=LEAK_RATE] [--spectral_radius=SPECTRAL_RADIUS]
            [--regularization=REGULARIZATION]
            [--compare] [--separate] [--uncompressed] [--plot]
            [--logistic] [-v]
    learndata.py -h | --help
    learndata.py --version

Options:
    -h, --help                  Show this screen.
    --n_samples=N_SAMPLES       Number of samples per vowel. [default: 100]
    --n_training=N_TRAINING     Number of training samples per vowel. [default: 80]
    --n_channels=N_CHANNELS     Number of channels to use. [default: 50]
    --trains_per_worker=TPW     Number of simulations per worker. [default: 1]
    --subfolder=SUBFOLDER       Subfolder to store results in.
    --n_reservoirs=N_RES        Network sizes for variation [default: 10,20,50]
    --spectral_radius=SR        Spectral radius of leaky reservoir. [default: 0.9]
    --leak_rate=LR              Leak rate of leaky reservoir neurons. [default: 0.4]
    --regularization=RGZN       Regularization parameter. [default: 0.001]
    --uncompressed              Use uncompressed DRNL output?
    --compare                   Compare leaky networks with non-leaky networks?
    --separate                  Use infant data as test samples only?
    --logistic                  Train with logistic regression instead of ridge regression?
    --plot                      Turns plotting on.
    -v, --verbose               Verbose output.

"""

__version__ = '0.1.0'


import random
import os
import gzip
from datetime import date
import cPickle

import numpy as np
# import Oger  # load after parsing arguments as it needs forever
# import pylab  # load after parsing arguments as it needs forever
# import matplotlib
# matplotlib.use('Agg')
Oger = None
pylab = None
import mdp
import scipy as sp
from mpi4py import MPI

from docopt import docopt
from confusion_matrix import ConfusionMatrix


VOWELS = ('a', 'i', 'u')

# global syllable library
LIB_SYLL = ('/a/', '/i/', '/u/', '[o]', '[e]', '[E:]', '[2]', '[y]', '[A]',
            '[I]', '[E]', '[O]', '[U]', '[9]', '[Y]', '[@]', '[@6]')

##########################################################
#
# Support functions:
#  - plot_conf_(conf, outputfile_, N)
#  - plot_prototypes(flow, N)
#
##########################################################



def plot_conf_(conf, outputfile, nn):
    """ function to visualise a balanced confusion matrix
         modified version of the predefined Oger.utils.plot_conf function
         additional arguments: outputfile_ for plot files, nn for reservoir size"""

    outputfile_plot = outputfile+'_'+str(nn)+'.png'
    np.asarray(conf).dump(outputfile_plot+'.np')

    res = pylab.imshow(np.asarray(conf), cmap=pylab.cm.jet, interpolation='nearest')
    for i, err in enumerate(conf.correct):
                                        # display correct detection percentages
                                        # (only makes sense for CMs that are normalised per class (each row sums to 1))
        err_percent = "%d%%" % round(err * 100)
        pylab.text(i-.2, i+.1, err_percent, fontsize=14)

    pylab.colorbar(res)

    pylab.savefig(outputfile_plot)



def plot_prototypes(N, leaky, **kwargs):
    """ function to visualize output neurons' states and reservoir activations
         for prototypical vowels
         - flow: trained flow of reservoir and output neurons
         - N: current reservoir size for scaling images
        global variables:
         - n_channels: number of channels used
         - n_vow: number of classes
         - outputfile_: current generic name of output file for image file name
         - lib_syll: list of syllables for image file name and title"""

    n_channels = kwargs['n_channels']
    n_vow = kwargs['n_vow']
    outputfile_ = kwargs['outputfile_']
    lib_syll = LIB_SYLL
    uncompressed = kwargs['uncompressed']
    output = kwargs['output']
    flow = kwargs['flow']
    rank = kwargs['rank']

    if output:
        print('worker', rank, 'plotting prototypes')

    vowels_and_files = [['adult [a]', 'data/ad_a.dat.gz', 'ad_a'], ['adult [i]', 'data/ad_i.dat.gz', 'ad_i'], ['adult [u]', 'data/ad_u.dat.gz', 'ad_u'], ['infant [a]', 'data/in_a.dat.gz', 'in_a'], ['infant [i]', 'data/in_i.dat.gz', 'in_i'], ['infant [u]', 'data/in_u.dat.gz', 'in_u']]

    for i in xrange(2*n_vow):             # loop over all syllables
        outputfile_plot = outputfile_+'_'+vowels_and_files[i][2]+'_'+str(N)+'.png'

        if n_vow > 3:
            i_current = (i+1)*33 - 6        # file index of current syllable prototype
            if not uncompressed:
                inputfile = 'data/'+str(n_vow)+'vow/'+str(n_vow)+'vow_'+str(n_channels)+'chan_'+str(i_current)+'.dat.gz'
                                        # name of corresponding activation file
            if uncompressed:
                inputfile = 'data/'+str(n_vow)+'vow_'+str(n_channels)+'chan/'+str(n_vow)+'vow_'+str(n_channels)+'chan_'+str(i_current)+'.dat.gz'
        elif n_vow < 4:
            i_current = (i+1)*54 - 1        # file index of current syllable prototype
            if not uncompressed:
                inputfile = vowels_and_files[i][1]
                                        # name of corresponding activation file
            if uncompressed:
                inputfile = 'data/'+str(n_vow)+'vow_'+str(n_channels)+'chan/'+str(n_vow)+'vow_'+str(n_channels)+'chan_'+str(i_current)+'.dat.gz'
                                        # name of corresponding activation file
        if output:
            print('loading '+inputfile)
        if not os.path.exists(inputfile):
                                        # end loop if inputfile not found
            print('file not found!')
            break
        inputf = gzip.open(inputfile, 'rb')
                                        # open current inputfile in gzip read mode
        current_data = np.load(inputf)  # load numpy array from current inputfile
        inputf.close()                  # close inputfile

        xtest = current_data
                                        # read activations from input array
        ytest = flow(xtest)             # get activations of output neurons of trained network

        current_flow = flow[0].inspect()[0].T
        np.array([ytest.T, current_flow]).dump(outputfile_plot+'.np')

        n_subplots_x, n_subplots_y = 2, 1
                                        # arrange two plots in one column
        pylab.subplot(n_subplots_x, n_subplots_y, 1)
                                        # upper plot

        ytest_min = ytest.min()
        ytest_max = ytest.max()
        if abs(ytest_min) > ytest_max:
            vmin = ytest_min
            vmax = -ytest_min
        else:
            vmax = ytest_max
            vmin = -ytest_max

        if not uncompressed:
            class_activity = pylab.imshow(ytest.T, origin='lower', cmap=pylab.cm.bwr, aspect=10.0/(n_vow+1), interpolation='none', vmin=vmin, vmax=vmax)
        if uncompressed:
            class_activity = pylab.imshow(ytest.T, origin='lower', cmap=pylab.cm.bwr, aspect=10000.0/n_vow, interpolation='none', vmin=vmin, vmax=vmax)
                                        # plot output activations, adjust to get uniform aspect for all n_vow
        pylab.title("Class activations of "+vowels_and_files[i][0])
        pylab.ylabel("Class")
        pylab.xlabel('')
        pylab.yticks(range(n_vow+1), lib_syll[:n_vow]+['null'])

        if not uncompressed:
            pylab.xticks(range(0, 35, 5), np.arange(0.0, 0.7, 0.1))
        if uncompressed:
            pylab.xticks(range(0, 35000, 5000), np.arange(0.0, 0.7, 0.1))

        pylab.colorbar(class_activity)


         # plot confusion matrix (balanced, each class is equally weighted)

        n_subplots_x, n_subplots_y = 2, 1
        pylab.subplot(n_subplots_x, n_subplots_y, 2)
                                        # lower plot

        current_flow_min = current_flow.min()
        current_flow_max = current_flow.max()
        if abs(current_flow_min) > current_flow_max:
            vmin_c = current_flow_min
            vmax_c = -current_flow_min
        else:
            vmax_c = current_flow_max
            vmin_c = -current_flow_max

        if not uncompressed:
            reservoir_activity = pylab.imshow(current_flow, origin='lower', cmap=pylab.cm.bwr, aspect=10.0/N, interpolation='none', vmin=vmin_c, vmax=vmax_c)
        if uncompressed:
            reservoir_activity = pylab.imshow(current_flow, origin='lower', cmap=pylab.cm.bwr, aspect=10000.0/N, interpolation='none', vmin=vmin_c, vmax=vmax_c)
                                        # plot reservoir states of current prototype,
                                        #  adjust to get uniform aspect for all N
        pylab.title("Reservoir states")
        pylab.xlabel('Time (s)')

        if not uncompressed:
            pylab.xticks(range(0, 35, 5), np.arange(0.0, 0.7, 0.1))
        if uncompressed:
            pylab.xticks(range(0, 35000, 5000), np.arange(0.0, 0.7, 0.1))

        pylab.ylabel("Neuron")
        if N < 6:
            pylab.yticks(range(N))

        pylab.colorbar(reservoir_activity)

        pylab.savefig(outputfile_plot)   # save figure
        pylab.close('all')

    xtest = None                    # destroy xtest and ytest to free up memory
    ytest = None
    class_activity = None
    reservoir_activity = None



def get_output_folder(subfolder, rank):
    today = date.today()
    today_string = today.isoformat()
    outputpath = 'output/'+today_string+'/'+subfolder+'/'
    if rank == 1:
        nn = 2
        while True:
            try:
                os.makedirs(outputpath)
                break
            except OSError:
                outputpath = 'output/'+today_string+'-%i'%nn+'/'+subfolder+'/'
                nn += 1
    return outputpath


def save_flow(flow, N, leaky, rank, output_folder):
    if rank == 1:
        filename = output_folder + str(N) + '_leaky' + str(leaky) + '.flow'
        with open(filename, 'wb') as flow_file:
            cPickle.dump(flow, flow_file)
        os.system('cp {} data/current_auditory_system.flow'.format(filename))


def get_training_and_test_sets(n_samples, n_training, n_vow):

    vowels = VOWELS

    path = 'data/'
    n_test = n_samples - n_training
    n_timesteps = 36
    training_set = []
    test_set = []
    protolabel = -np.ones([n_timesteps, n_vow+1])
    for i in xrange(n_vow):
        label = protolabel.copy()
        for i_time in xrange(n_timesteps):
            label[i_time][i] = 1.

        current_path = path+vowels[i]
        files = os.listdir(current_path)
        current_samples = []
        for item in files:
            if '.dat.gz' in item:
                current_samples.append(np.load(gzip.open(current_path+'/'+item)))
        random.shuffle(current_samples)

        for j in xrange(n_training):
            training_set.append((current_samples[j], label.copy()))
        for j in xrange(n_test):
            test_set.append((current_samples[n_training+j], label.copy()))

    label = protolabel.copy()
    # TODO What does this code do?
    # Answer: This section deals with the null samples. 
    #  The label of the null samples need positive entries in the corresponding line
    #  -> index of class null is 3
    for i_time in xrange(n_timesteps):
        label[i_time][3] = 1.

    for i in xrange(n_vow):
        current_path = path+'null_'+vowels[i]
        files = os.listdir(current_path)
        current_samples = []
        for item in files:
            if '.dat.gz' in item:
                current_samples.append(np.load(gzip.open(current_path+'/'+item)))
        random.shuffle(current_samples)

        for j in xrange(n_test/n_vow):
            test_set.append((current_samples[j], label.copy()))
        for j in xrange(n_test/n_vow, len(files)/2):
            training_set.append((current_samples[j], label.copy()))


    random.shuffle(training_set)
    random.shuffle(test_set)

    return training_set, test_set




##########################################################
#
# Main function
#
##########################################################


def learn(n_vow, N_reservoir=100, leaky=True, classification=True, **kwargs):
    """ function to perform supervised learning on an ESN
         data: data to be learned (ndarray including AN activations and teacher signals) OLD VERSION
         n_vow: total number of vowels used
         N_reservoir: size of ESN
         leaky: boolean defining if leaky ESN is to be used
         plots: boolean defining if results are to be plotted
         output: boolean defining if progress messages are to be displayed
         testdata: provide test data for manual testing (no cross validation) OLD VERSION
         separate: boolean defining if infant data is used as test set or test set is drawn randomly from adult+infant (n_vow=3)
         n_channels: number of channels used
         classification: boolean defining if sensory classification is performed instead of motor prediction"""

    output_folder = kwargs['output_folder']
    regularization = kwargs['regularization']
    logistic = kwargs['logistic']
    leak_rate = kwargs['leak_rate']
    spectral_radius = kwargs['spectral_radius']
    n_channels = kwargs['n_channels']
    n_vow = kwargs['n_vowel']
    n_samples = kwargs['n_samples']
    n_training = kwargs['n_training']
    output = kwargs['verbose']
    flow = kwargs['flow']
    rank = kwargs['rank']

    training_set, test_set = get_training_and_test_sets(n_samples, n_training, n_vow)

    if output:
        print('samples_test = '+str(test_set))
        print('len(samples_train) = '+str(len(training_set)))

    N_classes = n_vow+1                  # number of classes is total number of vowels + null class
    input_dim = n_channels              # input dimension is number of used channels

    if output:
        print('constructing reservoir')

    # construct individual nodes
    if leaky:                           # construct leaky reservoir
        reservoir = Oger.nodes.LeakyReservoirNode(input_dim=input_dim, output_dim=N_reservoir, input_scaling=1., 
            spectral_radius=spectral_radius, leak_rate=leak_rate)
                                        # call LeakyReservoirNode with appropriate number of input units and 
                                        #  given number of reservoir units
    else:                               # construct non-leaky reservoir
        reservoir = Oger.nodes.ReservoirNode(input_dim=input_dim, output_dim=N_reservoir, input_scaling=1.)
                                        # call ReservoirNode with appropriate number of input units and given number of reservoir units

    if logistic:
        readout = Oger.nodes.LogisticRegressionNode()
    else:
        readout = Oger.nodes.RidgeRegressionNode(regularization)
                                        # construct output units with Ridge Regression training method

    flow = mdp.Flow([reservoir, readout])
                                        # connect reservoir and output nodes


    if output:
        print("Training...")

    flow.train([[], training_set])
                                        # train flow with input files provided by file iterator


    ytest = []                          # initialize list of test output

    if output:
        print("Applying to testset...")

    losses = []                         # initiate list for discrete recognition variable for each test item
    ymean = []                          # initiate list for true class of each test item
    ytestmean = []                      # initiate list for class vote of trained flow for each test item

    for i_sample in xrange(len(test_set)):       # loop over all test samples
        if output:
            print('testing with sample '+str(i_sample))

        xtest = test_set[i_sample][0]
                                        # load xtest and ytarget as separate numpy arrays
        ytarget = test_set[i_sample][1]
        ytest = flow(xtest)             # evaluate trained output units' responses for current test item

        mean_sample_vote = mdp.numx.mean(ytest, axis=0)
                                        # average each output neurons' response over time
        if output:
            print('mean_sample_vote = '+str(mean_sample_vote))
        target = mdp.numx.mean(ytarget, axis=0)
                                        # average teacher signals over time
        if output:
            print('target = '+str(target))

        argmax_vote = sp.argmax(mean_sample_vote)
                                        # winner-take-all vote for final classification
        ytestmean.append(argmax_vote)   # append current vote to votes list of all items
        argmax_target = sp.argmax(target)
                                        # evaluate true class of current test item
        ymean.append(argmax_target)     # append current true class to list of all items

        loss = Oger.utils.loss_01(mdp.numx.atleast_2d(argmax_vote), mdp.numx.atleast_2d(argmax_target))
                                        # call loss_01 to compare vote and true class, 0 if match, 1 else
        if output:
            print('loss = '+str(loss))
        losses.append(loss)             # append current loss to losses of all items

        xtest = None                    # destroy xtest, ytest, ytarget, current_data to free up memory
        ytest = None
        ytarget = None

    error = mdp.numx.mean(losses)       # error rate is average number of mismatches
    if output:
        print('error = '+str(error))

    if output:
        print("error: "+str(error))
        print('ymean: '+str(ymean))
        print('ytestmean: '+str(ytestmean))

    ytestmean = np.array(ytestmean)     # convert ytestmean and ymean lists to numpy array for confusion matrix
    ymean = np.array(ymean)

    confusion_matrix = ConfusionMatrix.from_data(N_classes, ytestmean, ymean) # 10 classes
                                        # create confusion matrix from class votes and true classes
    c_matrix = confusion_matrix.balance()
                                        # normalize confusion matrix
    c_matrix = np.array(c_matrix)

    if output:
      print('confusion_matrix = '+str(c_matrix))


    save_flow(flow, N_reservoir, leaky, rank, output_folder)

    return error, c_matrix              # return current error rate and confusion matrix




################################################


def main_size_nocompare(**kwargs):
    """ main function simulating leaky ESNs only
            globals:
            - N_reservoir: list of reservoir sizes
            - n_vow: number of vowels used
            - leaky: boolean defining if leaky ESN used -> redundant?
            - plots: boolean defining if plots are created
            - output: boolean defining if progress messages are displayed
            - separate: boolean defining if infant samples are used as test data
            - n_channels: number of used channels"""

    n_vow = kwargs['n_vowel']
    n_workers = kwargs['n_workers']
    N_reservoir = kwargs['n_reservoirs']
    trains_per_worker = kwargs['trains_per_worker']
    output = kwargs['verbose']
    rank = kwargs['rank']
    plots = kwargs['plot']


    total_errors = np.zeros([trains_per_worker, len(N_reservoir)])
                                           # prepare lists for errors of each network size
    total_cmatrices = np.zeros([len(N_reservoir), n_vow+1, n_vow+1])                   # create empty list for confusion matrices for each network size

    for j in xrange(len(N_reservoir)):     # loop over network sizes
     for train in xrange(trains_per_worker):
       print('worker', rank, 'of', n_workers,
             'simulating leaky network of size', N_reservoir[j],
             '('+str(train+1)+'/'+str(trains_per_worker)+')')
       # call learn function to execute one simulation run
       error, c_matrix = learn(n_vow, N_reservoir=N_reservoir[j], leaky=True,
                               **kwargs)
       if (train==0) and plots:
           plot_prototypes(N_reservoir[j], leaky=True, **kwargs)

       if output:
            print('c_matrix:', c_matrix)

       total_errors[train][j] = error             # collect current error rate in errors list
       total_cmatrices[j] += c_matrix             # append current confusion matrix to confusion matrices list

    total_cmatrices /= trains_per_worker

    if output:
        print('total_cmatrices:', total_cmatrices)

    return total_errors, total_cmatrices   # return error rates and confusion matrices of this worker





################################################


def main_size_compare(**kwargs):
    """ main function simulating both leaky and non-leaky ESNs
        globals:
        - N_reservoir: list of reservoir sizes
        - n_vow: number of vowels used
        - plots: boolean defining if plots are created
        - output: boolean defining if progress messages are displayed
        - separate: boolean defining if infant samples are used as test data
        - n_channels: number of used channels"""

    n_vow = kwargs['n_vowel']
    n_workers = kwargs['n_workers']
    N_reservoir = kwargs['n_reservoirs']
    trains_per_worker = kwargs['trains_per_worker']
    output = kwargs['verbose']
    rank = kwargs['rank']
    plots = kwargs['plot']



    total_errors_leaky = np.zeros([trains_per_worker, len(N_reservoir)])
    total_errors_nonleaky = np.zeros([trains_per_worker, len(N_reservoir)])
                                           # prepare lists for errors of each network size
    total_cmatrices_leaky = np.zeros([len(N_reservoir), n_vow, n_vow])
    total_cmatrices_nonleaky = np.zeros([len(N_reservoir), n_vow, n_vow])
                                           # create empty list for confusion matrices for each network size

    for j in xrange(len(N_reservoir)):     # loop over network sizes
       for train in xrange(trains_per_worker):
           print('worker', rank, 'of', n_workers, 'simulating leaky network of size', N_reservoir[j], '('+str(train+1)+'/'+str(trains_per_worker)+')')
           error_leaky, c_matrix_leaky = learn(n_vow, N_reservoir=N_reservoir[j], leaky=True, **kwargs)

           if (train==0) and plots and (rank==1):
               plot_prototypes(N_reservoir[j], leaky=True, **kwargs)

           print('worker', rank, 'of', n_workers, 'simulating non-leaky network of size', N_reservoir[j], '('+str(train+1)+'/'+str(trains_per_worker)+')')
           error_nonleaky, c_matrix_nonleaky = learn(n_vow, N_reservoir=N_reservoir[j], leaky=False, **kwargs)
                                           # call learn function to execute one simulation run
           if (train==0) and plots and (rank==1):
               plot_prototypes(N_reservoir[j], leaky=False, **kwargs)

           if output:
               print('c_matrix_leaky:', c_matrix_leaky)
               print('c_matrix_nonleaky:', c_matrix_nonleaky)

           total_errors_leaky[train][j] = error_leaky             # collect current error rate in errors list
           total_errors_nonleaky[train][j] = error_nonleaky
           total_cmatrices_leaky[j] += c_matrix_leaky             # append current confusion matrix to confusion matrices list
           total_cmatrices_nonleaky[j] += c_matrix_nonleaky

    total_cmatrices_leaky /= trains_per_worker
    total_cmatrices_nonleaky /= trains_per_worker

    if output:
        print('total_cmatrices_leaky:', total_cmatrices_leaky)
        print('total_cmatrices_nonleaky:', total_cmatrices_nonleaky)

    return total_errors_leaky, total_errors_nonleaky, total_cmatrices_leaky, total_cmatrices_nonleaky
                                        # return error rates and confusion matrices of this worker





################################################
#
# Main script: Initialize
#
################################################

def main(args):

    # n_training used to be 183
    # n_samples used to be 204

    # bag off all kinds of variables and switches
    kwargs = dict()

    # command line arguments
    n_vowel = kwargs['n_vowel'] = int(args['<n_vowel>'])
    n_reservoirs = kwargs['n_reservoirs'] = [int(nn) for nn in args['--n_reservoirs'].split(',')]
    trains_per_worker = kwargs['trains_per_worker'] = int(args['--trains_per_worker'])
    subfolder = kwargs['subfolder'] = args['--subfolder']
    if subfolder is None:
        subfolder = kwargs['subfolder'] = ''
    kwargs['n_samples'] = int(args['--n_samples'])
    kwargs['n_training'] = int(args['--n_training'])
    kwargs['leak_rate'] = float(args['--leak_rate'])
    kwargs['spectral_radius'] = float(args['--spectral_radius'])
    kwargs['regularization'] = float(args['--regularization'])

    # command line FLAGS
    output = kwargs['verbose'] = args['--verbose']
    compare = kwargs['compare'] = args['--compare']
    separate = kwargs['separate'] = args['--separate']
    uncompressed = kwargs['uncompressed'] = args['--uncompressed']
    plot = kwargs['plot'] = args['--plot']
    kwargs['logistic'] = args['--logistic']

    # Inferred and static variables
    n_channels = kwargs['n_channels'] = 50
    kwargs['flow'] = None

    # see below
    # rank
    # outputfile
    # output_folder
    # n_workers


    # use separate case only for fewer than 4 vowels
    # -> infant data only available for up to 3 vowels
    if separate and n_vowel >= 4:
        raise ValueError("If you want to set separate, you need to use a n_vowel smaller than 4.")


    comm = MPI.COMM_WORLD                   # setup MPI framework
    n_workers = comm.Get_size()             # total number of workers / parallel processes
    rank = comm.Get_rank() + 1              # id of this worker -> master: 1

    np.random.seed()                        # numpy random seed w.r.t. global runtime
    np.random.seed(np.random.randint(256) * rank)
                                            # numpy random seed w.r.t. worker
    random.seed(np.random.randint(256) * rank)
                                            # random seed w.r.t. worker

    kwargs['n_workers'] = n_workers
    kwargs['rank'] = rank


    #************************************************
    #
    # args: n_vow(int) trains_per_worker(int) size(bool(int)) compressed(bool(int))
    #       compare(bool(int)) separate(bool(int)) n_channels(int) lower(int) upper(int)
    #
    # call salloc -p sleuths -n <n_workers> mpiexec python learndata.py <n_vow> <trains_per_worker> <size> <compressed> <compare> <separate>
    #
    #output = False                          # set global output boolean
    #plots = True                            # set global plots boolean
    #
    #************************************************

    print("importing Oger")
    global Oger
    import Oger

    print("importing pylab")
    global pylab
    import pylab

    print("importing matplotlib")
    import matplotlib
    matplotlib.use('Agg')

    Oger.utils.make_inspectable(Oger.nodes.LeakyReservoirNode)
                                            # make reservoir states inspectable for plotting
    Oger.utils.make_inspectable(Oger.nodes.ReservoirNode)


    if len(n_reservoirs)==1:
        size = False
    else:
        size = True

    n_trains = n_workers * trains_per_worker# number of trained networks for each network size set to number of workers


    output_folder = get_output_folder(subfolder=subfolder, rank=rank)

    if separate:                            # adjust output file names for separate and non-separate case
        outputfile_ = output_folder+str(n_vowel)+'vow_'+str(n_channels)+'chan_size'+str(size)+'_compare'+str(compare)+'_separate.out'
    else:
        outputfile_ = output_folder+str(n_vowel)+'vow_'+str(n_channels)+'chan_size'+str(size)+'_compare'+str(compare)+'.out'

    kwargs['outputfile'] = outputfile_
    kwargs['output_folder'] = output_folder

    ################################################



    if rank == 1:
        print('learning', n_vowel, 'vowels')
        print('averaging over', n_trains, 'trials')
        print('network size:', n_reservoirs)
        print('using uncompressed DRNL output:', uncompressed)
        print('comparing leaky network to non-leaky network:',compare)
        print('using infant samples as test data:', separate)
        print('verbose mode:', output)
        print('plot mode:', plot)


    if compare:                             # simulate both leaky and non-leaky ESNs

        errors_leaky, errors_nonleaky, c_matrices_leaky, c_matrices_nonleaky = main_size_compare()

        if output:
            print 'c_matrices_leaky:', c_matrices_leaky
            print 'c_matrices_nonleaky:', c_matrices_nonleaky

        final_errors = comm.gather(errors_leaky, root=0)
                                            # master collects all errors of leaky simulations from workers
        final_errors_nonleaky = comm.gather(errors_nonleaky, root=0)
                                            # master collects all errors of non-leaky simulations from workers
        final_cmatrices = comm.gather(c_matrices_leaky, root=0)
                                            # master collects all confusion matrices of leaky simulations from workers
        final_cmatrices_nonleaky = comm.gather(c_matrices_nonleaky, root=0)
                                            # master collects all confusion matrices of non-leaky simulations from workers


    if not compare:                                   # simulate leaky ESNs only

        errors, c_matrices = main_size_nocompare(**kwargs)
                                            # call main_size_nocompare to get error rates and confusion matrices
        final_errors = comm.gather(errors, root=0)
                                            # master collects all errors from workers
        final_cmatrices = comm.gather(c_matrices, root=0)
                                            # master collects all confusion matrices from workers




    ################################################


    if rank == 1:                           # post processing only by master

        if output:
            print 'final_cmatrices:', final_cmatrices
        if output:
            try:
                print 'type of final_cmatrices:', type(final_cmatrices)
            except:
                pass
            try:
                print 'length of final_cmatrices:', len(final_cmatrices)
            except:
                pass
            try:
                print 'final_cmatrices[0]:', final_cmatrices[0]
            except:
                pass

        os.system('rm '+outputfile_)        # delete old version of output file
        os.system('touch '+outputfile_)     # create new output file
        outputfile = open(outputfile_, 'w') # open output file in write mode

        if output:
            print 'gathered errors:', final_errors
        errors = np.array(final_errors)
        errors = errors.reshape([n_trains, len(n_reservoirs)])
        if output:
            print 'reshaped errors:', errors
        final_errors = np.average(errors, axis=0)
        final_stds = np.std(errors, axis=0)

        if output:
            print 'final_cmatrices after gathering:', final_cmatrices
        final_cmatrices = np.array(final_cmatrices)
        final_cmatrices = np.average(final_cmatrices, axis=0)
        if output:
            print 'final_cmatrices:', final_cmatrices



        if compare:                         # do same for non-leaky simulations in compare case
            errors_nonleaky = np.array(final_errors_nonleaky)
            errors_nonleaky = errors_nonleaky.reshape([n_trains, len(n_reservoirs)])
            final_errors_nonleaky = np.average(errors_nonleaky, axis=0)
            final_stds_nonleaky = np.std(errors_nonleaky, axis=0)

            if output:
                print 'final_cmatrices_nonleaky after gathering:', final_cmatrices_nonleaky
            final_cmatrices_nonleaky = np.array(final_cmatrices_nonleaky)
            final_cmatrices_nonleaky = np.average(final_cmatrices_nonleaky, axis=0)
            if output:
                print 'final_cmatrices_nonleaky:', final_cmatrices_nonleaky

        if output:
            try:
                print 'n_reservoirs:', n_reservoirs
            except:
                pass
            try:
                print 'final_errors:', final_errors
            except:
                pass
            try:
                print 'final_stds:', final_stds
            except:
                pass

        outputfile.write('leaky:\n\n')
        for i in xrange(len(n_reservoirs)):  # loop over all network sizes
            outputfile.write(str(n_reservoirs[i])+'     '+str(final_errors[i])+'     '+str(final_stds[i])+'\n')
                                            # record errors and standard deviations for each network size
            outputfile.flush()

            if plot:                       # plots for current network size
                C_Matrix = final_cmatrices[i]

                C_Matrix = ConfusionMatrix(C_Matrix, labels=range(n_vowel+1))
                                            # convert to confusion matrix object

                pylab.figure()              # plot confusion matrix
                pylab.title('Confusion matrix of leaky reservoir of size '+str(n_reservoirs[i]))
                pylab.xticks(np.arange(n_vowel+1), LIB_SYLL[:n_vowel]+['null'])
                pylab.yticks(np.arange(n_vowel+1), LIB_SYLL[:n_vowel]+['null'])
                pylab.xlabel('classified as')
                pylab.ylabel('sample')
                if output:
                    print 'current C_Matrix:', C_Matrix
                plot_conf_(C_Matrix, outputfile_, n_reservoirs[i])
                                            # call plot_conf for plotting confusion matrix
    #            plot_prototypes(flow, n_reservoirs[i])
                                            # call plot_prototypes to plot reservoir and output activations for each prototype


        if compare:                         # handle non-leaky results if available
            outputfile.write('\n\nnon-leaky:\n\n')
            for i in xrange(len(n_reservoirs)):
                                            # loop over all network sizes
                outputfile.write(str(n_reservoirs[i])+'     '+str(final_errors_nonleaky[i])+'     '+str(final_stds_nonleaky[i])+'\n')
                                            # record errors and standard deviations for each network size
                outputfile.flush()

                if plot:                   # plots for current network size
                    C_Matrix = final_cmatrices_nonleaky[i]
                    if output:
                        print 'C_Matrix before conversion:', C_Matrix
                    C_Matrix = ConfusionMatrix(C_Matrix, labels=range(n_vowel))
                                            # convert to confusion matrix object

                    pylab.figure()          # plot confusion matrix
                    pylab.title('Confusion matrix of non-leaky reservoir of size '+str(n_reservoirs[i]))
                    pylab.xticks(np.arange(n_vowel), LIB_SYLL[:n_vowel])
                    pylab.yticks(np.arange(n_vowel), LIB_SYLL[:n_vowel])
                    pylab.xlabel('classified as')
                    pylab.ylabel('sample')
                    if output:
                        print 'current C_Matrix:', C_Matrix
                    plot_conf_(C_Matrix, outputfile_+'nonleaky', n_reservoirs[i])
                                            # call plot_conf for plotting confusion matrix

        outputfile.close()                  # close output file
        print 'done'

if __name__ == "__main__":
    arguments = docopt(__doc__, version='learndata %s' % __version__)
    main(arguments)

