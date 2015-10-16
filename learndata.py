import sys
sys.path.append('/home/murakami/lib/python2.7/site-packages/')

import matplotlib
matplotlib.use('Agg')

from brian import *
from brian.hears import *
from scipy.signal import resample
from VTL_API.gesWav_fun import gesToWave
import numpy as np
import argparse

import Oger
import mdp
import pylab
import scipy as sp
import numpy as np
import random
from Oger.utils import plot_conf
import sys
from confusion_matrix import ConfusionMatrix
import os
import gzip
from mpi4py import MPI
from datetime import date
import cPickle





##########################################################
#
# Support functions:
#  - loss_01_time(x, y)
#  - plot_conf_(conf, outputfile_, N)
#  - plot_prototypes(flow, N)
#
# class FileIterator(indices, n_vow)
#
##########################################################



def loss_01_time(x, y):                 
     """ function for comparing two trajectories
         modify shapes of x and y to use predefined loss_01 function"""

     global separate                    # use globally defined separation boolean

     if separate:                       # target signal has different shape in the separate case
      return Oger.utils.loss_01(mdp.numx.atleast_2d(sp.argmax(mdp.numx.mean(x, axis=0))), mdp.numx.atleast_2d(sp.argmax(y)))
                                        # call Oger.utils.loss_01, see documentation
     else:
      return Oger.utils.loss_01(mdp.numx.atleast_2d(sp.argmax(mdp.numx.mean(x, axis=0))), mdp.numx.atleast_2d(sp.argmax(mdp.numx.mean(y, axis=0))))
                                        # call Oger.utils.loss_01, see documentation



def plot_conf_(conf, outputfile_, N):
    """ function to visualise a balanced confusion matrix
         modified version of the predefined Oger.utils.plot_conf function
         additional arguments: outputfile_ for plot files, N for reservoir size"""

    outputfile_plot = outputfile_+'_'+str(N)+'.png'
    np.asarray(conf).dump(outputfile_plot+'.np')

    res = pylab.imshow(np.asarray(conf), cmap=pylab.cm.jet, interpolation='nearest')
    for i, err in enumerate(conf.correct):
                                        # display correct detection percentages 
                                        # (only makes sense for CMs that are normalised per class (each row sums to 1))
        err_percent = "%d%%" % round(err * 100)
        pylab.text(i-.2, i+.1, err_percent, fontsize=14)

    cb = pylab.colorbar(res)

    pylab.savefig(outputfile_plot)
                                        # save plot
    res = None



def plot_prototypes(N, leaky=True):
    """ function to visualize output neurons' states and reservoir activations
         for prototypical vowels
         - flow: trained flow of reservoir and output neurons
         - N: current reservoir size for scaling images
        global variables:
         - n_channels: number of channels used
         - n_vow: number of classes
         - outputfile_: current generic name of output file for image file name
         - lib_syll: list of syllables for image file name and title"""

    global n_channels, n_vow, outputfile_, lib_syll, compressed, output, flow, first_plot, cb, cb2

    if output:
        print 'worker', rank, 'plotting prototypes'

    vowels_and_files = [['adult [a]', 'data/ad_a.dat.gz', 'ad_a'], ['adult [i]', 'data/ad_i.dat.gz', 'ad_i'], ['adult [u]', 'data/ad_u.dat.gz', 'ad_u'], ['infant [a]', 'data/in_a.dat.gz', 'in_a'], ['infant [i]', 'data/in_i.dat.gz', 'in_i'], ['infant [u]', 'data/in_u.dat.gz', 'in_u']]

    for i in xrange(2*n_vow):             # loop over all syllables
        outputfile_plot = outputfile_+'_'+vowels_and_files[i][2]+'_'+str(N)+'.png'

        if n_vow > 3:
#            i_current = (i)*33 + 5
            i_current = (i+1)*33 - 6        # file index of current syllable prototype
            if compressed:
                inputfile = 'data/'+str(n_vow)+'vow/'+str(n_vow)+'vow_'+str(n_channels)+'chan_'+str(i_current)+'.dat.gz'
                                        # name of corresponding activation file
            if not compressed:
                inputfile = 'data/'+str(n_vow)+'vow_'+str(n_channels)+'chan/'+str(n_vow)+'vow_'+str(n_channels)+'chan_'+str(i_current)+'.dat.gz'
        elif n_vow < 4:
            i_current = (i+1)*54 - 1        # file index of current syllable prototype
            if compressed:
                inputfile = vowels_and_files[i][1]
                                        # name of corresponding activation file
            if not compressed:
                inputfile = 'data/'+str(n_vow)+'vow_'+str(n_channels)+'chan/'+str(n_vow)+'vow_'+str(n_channels)+'chan_'+str(i_current)+'.dat.gz'
                                        # name of corresponding activation file
        if output:
         print 'loading '+inputfile
        if not os.path.exists(inputfile):
                                        # end loop if inputfile not found
            print 'file not found!'
            break
        inputf = gzip.open(inputfile, 'rb')
                                        # open current inputfile in gzip read mode
        current_data = np.load(inputf)  # load numpy array from current inputfile
        inputf.close()                  # close inputfile

        xtest = current_data
#        xtest = np.array(list(current_data[0]))
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

        if compressed:
            class_activity = pylab.imshow(ytest.T, origin='lower', cmap=pylab.cm.bwr, aspect=10.0/(n_vow+1), interpolation='none', vmin=vmin, vmax=vmax)
        if not compressed:
            class_activity = pylab.imshow(ytest.T, origin='lower', cmap=pylab.cm.bwr, aspect=10000.0/n_vow, interpolation='none', vmin=vmin, vmax=vmax)
                                        # plot output activations, adjust to get uniform aspect for all n_vow
        pylab.title("Class activations of "+vowels_and_files[i][0])
        pylab.ylabel("Class")
        pylab.xlabel('')
        pylab.yticks(range(n_vow+1), lib_syll[:n_vow]+['null'])

        if compressed:
            pylab.xticks(range(0, 35, 5), np.arange(0.0, 0.7, 0.1))
        if not compressed:
            pylab.xticks(range(0, 35000, 5000), np.arange(0.0, 0.7, 0.1))

        cb = pylab.colorbar(class_activity)
#        cb.update_bruteforce(class_activity)


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

        if compressed:
            reservoir_activity = pylab.imshow(current_flow, origin='lower', cmap=pylab.cm.bwr, aspect=10.0/N, interpolation='none', vmin=vmin_c, vmax=vmax_c)
        if not compressed:
            reservoir_activity = pylab.imshow(current_flow, origin='lower', cmap=pylab.cm.bwr, aspect=10000.0/N, interpolation='none', vmin=vmin_c, vmax=vmax_c)
                                        # plot reservoir states of current prototype,
                                        #  adjust to get uniform aspect for all N
        pylab.title("Reservoir states")
        pylab.xlabel('Time (s)')

        if compressed:
            pylab.xticks(range(0, 35, 5), np.arange(0.0, 0.7, 0.1))
        if not compressed:
            pylab.xticks(range(0, 35000, 5000), np.arange(0.0, 0.7, 0.1))

        pylab.ylabel("Neuron")
        if N < 6:
            pylab.yticks(range(N))
        
        cb2 = pylab.colorbar(reservoir_activity)
#        cb2.update_bruteforce(reservoir_activity)

        pylab.savefig(outputfile_plot)   # save figure
        pylab.close('all')

    xtest = None                    # destroy xtest and ytest to free up memory
    ytest = None
    class_activity = None
    reservoir_activity = None

    


class File_Iterator:
    """ class to iterate input files needed for flow train function
        -> not all uncompressed input activations can be loaded at the same time
        arguments:
            - indices: list of file indices determined randomly
            - n_vow: total number of vowels
            - separate: use separate infant files?
            - compressed: use compressed DRNL output?"""

    def __init__(self, indices, n_vow=5, compressed=True):
                                        # called during File_Iterator object creation
        self.indices = indices          # assign file indices list to object
        self.n_vow = n_vow              # assign total number of vowels to object
        self.i_current = 0              # initiate current item index, this will increment while iterating
        self.separate = separate        # determine if infant files are opened separately
        self.compressed = compressed    # determine if compressed files are used

    def __iter__(self):                 # standard function for iterator classes
        return self

    def next(self):                     # called during every iteration
        global output
        try:                            # while end of indices list not reached
            self.current = self.indices[self.i_current]
                                        # current is the file index, i.e. the i_current'th integer of indices list
            if self.compressed:
                pathname = 'data/'+str(self.n_vow)+'vow/'+str(self.n_vow)+'vow_50chan_'+str(self.current)+'.dat.gz'
                                        # pathname of current file
            if not self.compressed:
                pathname = 'data/'+str(self.n_vow)+'vow_50chan/'+str(self.n_vow)+'vow_50chan_'+str(self.current)+'.dat.gz'
                                        # pathname of current file
            if not os.path.exists(pathname):
                                        # stop iteration if file not found
                print pathname+' not found! Stopping iteration.'
                raise StopIteration
            else:                       # execute if file found
                if output:
                 print 'opening '+pathname
                inputfile = gzip.open(pathname, 'rb')
                                        # open file in gzip read mode
                inputarray = np.load(inputfile)
                                        # load numpy array from file
                inputfile.close()       # close file
                inputs = np.array(list(inputarray[0]))
                                        # load inputs and outputs as separate numpy arrays
                outputs = np.array(list(inputarray[1]))
                self.i_current += 1     # increment iteration index
                return (inputs, outputs)#inputarray
                                        # return inputs, outputs tuple
        except IndexError:              # announce end of iteration if end of indices list reached
            if output:
             print 'End of training set reached. Stopping iteration.'
            raise StopIteration



def get_output_folder(subfolder):
    global rank

    today = date.today()
    today_string = today.isoformat()
    outputpath_short = 'output/'+today_string
    outputpath = 'output/'+today_string+'/'+subfolder+'/'
    if rank == 1:
        try:
            os.system('mkdir '+outputpath_short)
            os.system('mkdir '+outputpath)
        except Error:
            print Error
        finally:
            pass
    
    return outputpath




def save_flow(flow, N, leaky):
    global output_folder, rank
    
    if rank == 1:
        filename = output_folder+str(N)+'_leaky'+str(leaky)+'.flow'
        os.system('touch '+filename)
        flow_file = open(filename, 'w')
        cPickle.dump(flow, flow_file)
        flow_file.close()


def get_training_and_test_sets(n_samples, n_training):
    
    vowels = ['a', 'i', 'u']
    n_vow = len(vowels)
    path = 'data/'
#    n_samples = 204
#    n_training = 183
#    n_test = 21
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
#            test_set.append((current_samples[n_training/n_vow+j], label.copy()))
            test_set.append((current_samples[j], label.copy()))
#        for j in xrange(n_training/n_vow):
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




def learn(n_vow, N_reservoir=100, leaky=True, plots=False, output=False, separate=False, compressed=True, n_channels=50, classification=True):
    global rank, flow, logistic, spectral_radius, leak_rate, regularization, n_samples, n_training

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

    training_set, test_set = get_training_and_test_sets(n_samples, n_training)

    if output:
        print 'samples_test = '+str(test_set)
        print 'len(samples_train) = '+str(len(training_set))

    N_classes = n_vow+1                  # number of classes is total number of vowels + null class
    input_dim = n_channels              # input dimension is number of used channels

    if output:
     print 'constructing reservoir'

    # construct individual nodes
    if leaky:                           # construct leaky reservoir
#        reservoir = Oger.nodes.LeakyReservoirNode(input_dim=input_dim, output_dim=N_reservoir, input_scaling=.1, leak_rate=.3)
        reservoir = Oger.nodes.LeakyReservoirNode(input_dim=input_dim, output_dim=N_reservoir, input_scaling=1., spectral_radius=spectral_radius, leak_rate=leak_rate)
                                        # call LeakyReservoirNode with appropriate number of input units and given number of reservoir units
    else:                               # construct non-leaky reservoir
#        reservoir = Oger.nodes.ReservoirNode(input_dim=input_dim, output_dim=N_reservoir, input_scaling=.1)
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
     print "Training..."
    
    flow.train([[], training_set])
                                        # train flow with input files provided by file iterator


    ytest = []                          # initialize list of test output

    if output:
     print "Applying to testset..."

    losses = []                         # initiate list for discrete recognition variable for each test item
    ymean = []                          # initiate list for true class of each test item
    ytestmean = []                      # initiate list for class vote of trained flow for each test item

    for i_sample in xrange(len(test_set)):       # loop over all test samples
        if output:
         print 'testing with sample '+str(i_sample)

        xtest = test_set[i_sample][0]
                                        # load xtest and ytarget as separate numpy arrays
        ytarget = test_set[i_sample][1]
        ytest = flow(xtest)             # evaluate trained output units' responses for current test item

        mean_sample_vote = mdp.numx.mean(ytest, axis=0)
                                        # average each output neurons' response over time
        if output:
            print 'mean_sample_vote = '+str(mean_sample_vote)
        target = mdp.numx.mean(ytarget, axis=0)
                                        # average teacher signals over time
        if output:
            print 'target = '+str(target)

        argmax_vote = sp.argmax(mean_sample_vote)
                                        # winner-take-all vote for final classification
        ytestmean.append(argmax_vote)   # append current vote to votes list of all items
        argmax_target = sp.argmax(target)
                                        # evaluate true class of current test item
        ymean.append(argmax_target)     # append current true class to list of all items

        loss = Oger.utils.loss_01(mdp.numx.atleast_2d(argmax_vote), mdp.numx.atleast_2d(argmax_target))
                                        # call loss_01 to compare vote and true class, 0 if match, 1 else
        if output:
            print 'loss = '+str(loss)
        losses.append(loss)             # append current loss to losses of all items

        xtest = None                    # destroy xtest, ytest, ytarget, current_data to free up memory
        ytest = None
        ytarget = None
        current_data = None

    error = mdp.numx.mean(losses)       # error rate is average number of mismatches
    if output:
        print 'error = '+str(error)

    if output:
     print "error: "+str(error)
     print 'ymean: '+str(ymean)
     print 'ytestmean: '+str(ytestmean)

    ytestmean = np.array(ytestmean)     # convert ytestmean and ymean lists to numpy array for confusion matrix
    ymean = np.array(ymean)             

    confusion_matrix = ConfusionMatrix.from_data(N_classes, ytestmean, ymean) # 10 classes
                                        # create confusion matrix from class votes and true classes
    c_matrix = confusion_matrix.balance()
                                        # normalize confusion matrix
    c_matric = np.array(c_matrix)

    if output:
      print 'confusion_matrix = '+str(c_matrix)


    save_flow(flow, N_reservoir, leaky) 
        

    return error, c_matrix              # return current error rate and confusion matrix




################################################


def main_size_nocompare():
 """ main function simulating leaky ESNs only
        globals:
        - N_reservoir: list of reservoir sizes
        - n_vow: number of vowels used
        - leaky: boolean defining if leaky ESN used -> redundant?
        - plots: boolean defining if plots are created
        - output: boolean defining if progress messages are displayed
        - separate: boolean defining if infant samples are used as test data
        - n_channels: number of used channels"""
 global N_reservoir, n_vow, leaky, plots, output, separate, n_channels, compressed, rank, n_workers, trains_per_worker

 total_errors = np.zeros([trains_per_worker, len(N_reservoir)])
                                        # prepare lists for errors of each network size
 total_cmatrices = np.zeros([len(N_reservoir), n_vow+1, n_vow+1])                   # create empty list for confusion matrices for each network size
                                     
 for j in xrange(len(N_reservoir)):     # loop over network sizes
  for train in xrange(trains_per_worker):
    print 'worker', rank, 'of', n_workers, 'simulating leaky network of size', N_reservoir[j], '('+str(train+1)+'/'+str(trains_per_worker)+')'
    error, c_matrix = learn(n_vow, N_reservoir=N_reservoir[j], leaky=True, plots=plots, output=output, separate=separate, compressed=compressed, n_channels=n_channels)
                                        # call learn function to execute one simulation run
    if (train==0) and plots:
        plot_prototypes(N_reservoir[j])

    if output:
     print 'c_matrix:', c_matrix

    total_errors[train][j] = error             # collect current error rate in errors list
    total_cmatrices[j] += c_matrix             # append current confusion matrix to confusion matrices list

 total_cmatrices /= trains_per_worker
 
 if output:
  print 'total_cmatrices:', total_cmatrices

 return total_errors, total_cmatrices   # return error rates and confusion matrices of this worker





################################################


def main_size_compare():
 """ main function simulating both leaky and non-leaky ESNs
        globals:
        - N_reservoir: list of reservoir sizes
        - n_vow: number of vowels used
        - plots: boolean defining if plots are created
        - output: boolean defining if progress messages are displayed
        - separate: boolean defining if infant samples are used as test data
        - n_channels: number of used channels"""
 global N_reservoir, n_vow, plots, output, separate, n_channels, compressed, trains_per_worker, rank


 total_errors_leaky = np.zeros([trains_per_worker, len(N_reservoir)])
 total_errors_nonleaky = np.zeros([trains_per_worker, len(N_reservoir)])
                                        # prepare lists for errors of each network size
 total_cmatrices_leaky = np.zeros([len(N_reservoir), n_vow, n_vow])
 total_cmatrices_nonleaky = np.zeros([len(N_reservoir), n_vow, n_vow])
                                        # create empty list for confusion matrices for each network size
                                     
 for j in xrange(len(N_reservoir)):     # loop over network sizes
    for train in xrange(trains_per_worker):
        print 'worker', rank, 'of', n_workers, 'simulating leaky network of size', N_reservoir[j], '('+str(train+1)+'/'+str(trains_per_worker)+')'
        error_leaky, c_matrix_leaky = learn(n_vow, N_reservoir=N_reservoir[j], leaky=True, plots=plots, output=output, separate=separate, compressed=compressed, n_channels=n_channels)

        if (train==0) and plots and (rank==1):
            plot_prototypes(N_reservoir[j], leaky=True)

        print 'worker', rank, 'of', n_workers, 'simulating non-leaky network of size', N_reservoir[j], '('+str(train+1)+'/'+str(trains_per_worker)+')'
        error_nonleaky, c_matrix_nonleaky = learn(n_vow, N_reservoir=N_reservoir[j], leaky=False, plots=plots, output=output, separate=separate, compressed=compressed, n_channels=n_channels)
                                        # call learn function to execute one simulation run
        if (train==0) and plots and (rank==1):
            plot_prototypes(N_reservoir[j], leaky=False)

        if output:
            print 'c_matrix_leaky:', c_matrix_leaky
            print 'c_matrix_nonleaky:', c_matrix_nonleaky

        total_errors_leaky[train][j] = error_leaky             # collect current error rate in errors list
        total_errors_nonleaky[train][j] = error_nonleaky
        total_cmatrices_leaky[j] += c_matrix_leaky             # append current confusion matrix to confusion matrices list
        total_cmatrices_nonleaky[j] += c_matrix_nonleaky

 total_cmatrices_leaky /= trains_per_worker
 total_cmatrices_nonleaky /= trains_per_worker
 
 if output:
  print 'total_cmatrices_leaky:', total_cmatrices_leaky
  print 'total_cmatrices_nonleaky:', total_cmatrices_nonleaky

 return total_errors_leaky, total_errors_nonleaky, total_cmatrices_leaky, total_cmatrices_nonleaky
                                        # return error rates and confusion matrices of this worker





################################################
#
# Main script: Initialize
#
################################################


comm = MPI.COMM_WORLD                   # setup MPI framework
n_workers = comm.Get_size()             # total number of workers / parallel processes
rank = comm.Get_rank() + 1              # id of this worker -> master: 1

np.random.seed()                        # numpy random seed w.r.t. global runtime
np.random.seed(np.random.randint(256) * rank)
                                        # numpy random seed w.r.t. worker
random.seed(np.random.randint(256) * rank)
                                        # random seed w.r.t. worker




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

lib_syll =  ['/a/','/i/','/u/','[o]','[e]','[E:]','[2]','[y]','[A]','[I]','[E]','[O]','[U]','[9]','[Y]','[@]','[@6]']
                                        # global syllable library

Oger.utils.make_inspectable(Oger.nodes.LeakyReservoirNode)
                                        # make reservoir states inspectable for plotting
Oger.utils.make_inspectable(Oger.nodes.ReservoirNode)    



parser = argparse.ArgumentParser()
parser.add_argument('n_vow', type=int, help='number of vowels to be used')
parser.add_argument('-t', '--trains_per_worker', nargs='?', type=int, default=1, help='number of simulations per worker')
parser.add_argument('-N', '--size', action='store', nargs='*', type=int, default=[1,10,20,50,100,200,500,1000], help='network sizes for variation')
parser.add_argument('-u', '--uncompressed', action='store_true', help='use uncompressed DRNL output?')
parser.add_argument('-c', '--compare', action='store_true', help='compare leaky networks with non-leaky networks?')
parser.add_argument('-s', '--separate', action='store_true', help='use infant data as test samples only?')
parser.add_argument('-v', '--verbose', action='store_true', help='increase verbosity?')
parser.add_argument('-p', '--plots', action='store_false', help='turn plots off?')
parser.add_argument('-f', '--folder', nargs='?', type=str, default='', help='subfolder to store results in')
parser.add_argument('-l', '--logistic', action='store_true', help='train with logistic regression instead of ridge regression?')
parser.add_argument('-r', '--spectral_radius', action='store', type=float, default=0.9, help='spectral radius of leaky reservoir')
parser.add_argument('-k', '--leak_rate', action='store', type=float, default=0.4, help='leak rate of leaky reservoir neurons')
parser.add_argument('-R', '--regularization', action='store', type=float, default=0.001, help='regularization parameter')
parser.add_argument('-n', '--n_samples', nargs='?', type=int, default=204, help='number of samples per vowel')
parser.add_argument('-T', '--n_training', nargs='?', type=int, default=183, help='number of training samples per vowel')

args = parser.parse_args()
n_vow = args.n_vow
trains_per_worker = args.trains_per_worker
N_reservoir = args.size
compressed = not args.uncompressed
compare = args.compare
separate_ = args.separate
output = args.verbose
plots = args.plots
subfolder = args.folder
logistic = args.logistic
spectral_radius = args.spectral_radius
leak_rate = args.leak_rate
regularization = args.regularization
n_samples = args.n_samples
n_training = args.n_training


n_channels = 50
lower = 0
upper = 50
if len(N_reservoir)==1:
    size = False
else:
    size = True

n_trains = n_workers * trains_per_worker# number of trained networks for each network size set to number of workers

separate = separate_ and (n_vow < 4)    # use separate case only for fewer than 4 vowels
                                        # -> infant data only available for up to 3 vowels

output_folder = get_output_folder(subfolder=subfolder)

if separate:                            # adjust output file names for separate and non-separate case
    outputfile_ = output_folder+str(n_vow)+'vow_'+str(n_channels)+'chan_size'+str(size)+'_compare'+str(compare)+'_separate.out'
else:
    outputfile_ = output_folder+str(n_vow)+'vow_'+str(n_channels)+'chan_size'+str(size)+'_compare'+str(compare)+'.out'

flow = None


################################################



if rank == 1:
    print 'learning', n_vow, 'vowels'
    print 'averaging over', n_trains, 'trials'
    print 'network size:', N_reservoir
    print 'using compressed DRNL output:', compressed
    print 'comparing leaky network to non-leaky network:',compare
    print 'using infant samples as test data:', separate
    print 'verbose mode:', output
    print 'plot mode:', plots


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

    errors, c_matrices = main_size_nocompare()
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
    errors = errors.reshape([n_trains, len(N_reservoir)])
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
        errors_nonleaky = errors_nonleaky.reshape([n_trains, len(N_reservoir)])
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
            print 'N_reservoir:', N_reservoir
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
    for i in xrange(len(N_reservoir)):  # loop over all network sizes
        outputfile.write(str(N_reservoir[i])+'     '+str(final_errors[i])+'     '+str(final_stds[i])+'\n')
                                        # record errors and standard deviations for each network size
        outputfile.flush()

        if plots:                       # plots for current network size
            C_Matrix = final_cmatrices[i]

            C_Matrix = ConfusionMatrix(C_Matrix, labels=range(n_vow+1))
                                        # convert to confusion matrix object

            pylab.figure()              # plot confusion matrix
            pylab.title('Confusion matrix of leaky reservoir of size '+str(N_reservoir[i]))
            pylab.xticks(np.arange(n_vow+1), lib_syll[:n_vow]+['null'])
            pylab.yticks(np.arange(n_vow+1), lib_syll[:n_vow]+['null'])
            pylab.xlabel('classified as')
            pylab.ylabel('sample')
            if output:
                print 'current C_Matrix:', C_Matrix
            plot_conf_(C_Matrix, outputfile_, N_reservoir[i])
                                        # call plot_conf for plotting confusion matrix
#            plot_prototypes(flow, N_reservoir[i])
                                        # call plot_prototypes to plot reservoir and output activations for each prototype


    if compare:                         # handle non-leaky results if available
        outputfile.write('\n\nnon-leaky:\n\n')
        for i in xrange(len(N_reservoir)):
                                        # loop over all network sizes
            outputfile.write(str(N_reservoir[i])+'     '+str(final_errors_nonleaky[i])+'     '+str(final_stds_nonleaky[i])+'\n')
                                        # record errors and standard deviations for each network size
            outputfile.flush()

            if plots:                   # plots for current network size
                C_Matrix = final_cmatrices_nonleaky[i]
                if output:
                    print 'C_Matrix before conversion:', C_Matrix
                C_Matrix = ConfusionMatrix(C_Matrix, labels=range(n_vow))
                                        # convert to confusion matrix object

                pylab.figure()          # plot confusion matrix
                pylab.title('Confusion matrix of non-leaky reservoir of size '+str(N_reservoir[i]))
                pylab.xticks(np.arange(n_vow), lib_syll[:n_vow])
                pylab.yticks(np.arange(n_vow), lib_syll[:n_vow])
                pylab.xlabel('classified as')
                pylab.ylabel('sample')
                if output:
                    print 'current C_Matrix:', C_Matrix
                plot_conf_(C_Matrix, outputfile_+'nonleaky', N_reservoir[i])
                                        # call plot_conf for plotting confusion matrix

    outputfile.close()                  # close output file
    print 'done'
