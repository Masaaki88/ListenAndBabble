"""
CMA-ES: Evolution Strategy with Covariance Matrix Adaptation for
nonlinear function minimization.

This code refers to "The CMA Evolution Strategy: A Tutorial" by 
Nikolaus Hansen (Appendix C).
"""
import matplotlib
matplotlib.use('Agg')                  # for use on clusters
import numpy as np
import argparse
from environment_fun import evaluate_environment
from mpi4py import MPI
from datetime import date
import os
from collections import deque
from numpy.linalg import norm
import cPickle
import datetime
from time import *
from VTL_API.create_sp_finish import create_speaker_finish



###########################################################
#
# initialization
#
###########################################################


comm = MPI.COMM_WORLD                   # setup MPI framework
n_workers = comm.Get_size()             # total number of workers / parallel processes
rank = comm.Get_rank()              # id of this worker -> master: 0

lambda_ = n_workers - 1

np.random.seed()                        # numpy random seed w.r.t. global runtime
np.random.seed(np.random.randint(256) * rank)
                                        # numpy random seed w.r.t. worker


###########################################################
#### set up argument parsing
###########################################################

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='store_true', help='increase verbosity?')
parser.add_argument('-f', '--folder', nargs='?', type=str, default='', help='subfolder to store results in')
parser.add_argument('-t', '--target', nargs='?', type=str, default='a', help='target vowel for imitation')
parser.add_argument('-p', '--parameters', nargs='*', type=str, default=['TCX'], help='vocal tract parameters to learn')
parser.add_argument('-s', '--sigma', nargs='?', type=float, default=0.4, help='step-size = sigma')
parser.add_argument('-i', '--infant', action='store_true', help='simulate infant speaker?')
parser.add_argument('-N', '--n_vowels', nargs='?', type=int, default=3, help='number of vowels')
parser.add_argument('-m', '--softmax', action='store_true', help='use softmax reward?')
parser.add_argument('-I', '--intrinsic_motivation', action='store_true', help='intrinsic motivation?')
parser.add_argument('-T', '--threshold', nargs='?', type=float, default=0.5, help='threshold for convergence')
parser.add_argument('-P', '--predefined', action='store_true', help='initialize with predefined configuration?')
parser.add_argument('-e', '--energy_factor', nargs='?', type=float, default=0.1, help='energy balance factor')
parser.add_argument('-a', '--alpha', nargs='?', type=float, default=1.0, help='alpha for constraint penalty')
parser.add_argument('-r', '--resample', action='store_true', help='resample invalid motor parameters?')
parser.add_argument('-o', '--normalize', action='store_true', help='normalize ESN output?')
parser.add_argument('-c', '--no_convergence', action='store_true', help='turn off convergence criterion?')
parser.add_argument('-A', '--random_restart', action='store_true', 
                        help='restart search after bad solution from random learnt variables?')
parser.add_argument('-L', '--load_state', nargs='?', type=str, default=None, help='load saved state?')
parser.add_argument('-C', '--conditioning_maximum', nargs='?', type=float, default=1e14, help='maximal conditioning number')
parser.add_argument('-w', '--no_reward_convergence', action='store_true', help='ignore reward for convergence?')
parser.add_argument('-S', '--ptp_stop', nargs='?', type=float, default=0.001, help='convergence range')
parser.add_argument('-n', '--n_trials', nargs='?', type=int, default=1, help='number of trials for averaging?')
parser.add_argument('-F', '--flat_tongue', action='store_true', help='simulate flat tongue, i.e. set all TS to 0?')
parser.add_argument('-d', '--debug', action='store_true', help='turn on debug mode?')
parser.add_argument('-k', '--constant_sigma0', action='store_true', help='keep sigma 0 constant?')
parser.add_argument('-X', '--default_settings', action='store_true', help='use default settings?')

# usage:
#  $ salloc -p sleuths -n (lambda/int) mpirun python rl_agent_mpi.py [-v] [-n (n_samples/int)] [-f (folder/str)] [-t (target/str)]
#     [-p (parameters/str)] [-s (sigma/float)] [-i] [-N (n_vowels/int)] [-m] [-I] [-T (threshold/float)] [-P] [-e (energy_factor/float)]
#     [-a (alpha/float)] [-F]

# thesis settings:
#   salloc -p sleuths -n 100 mpirun python rl_agent_mpi.py -f default_output_folder -p all -i -m -I -r -o -A -w -c

args = parser.parse_args()
verbose = args.verbose
folder = args.folder
target = args.target
parameters = args.parameters
sigma0 = args.sigma
infant = args.infant
n_vowels = args.n_vowels
softmax = args.softmax
intrinsic_motivation = args.intrinsic_motivation
conf_threshold = args.threshold
predefined = args.predefined
energy_factor = args.energy_factor
alpha = args.alpha
resample = args.resample
normalize = args.normalize
no_convergence = args.no_convergence
random_restart = args.random_restart
load_state = args.load_state
cond_stop = args.conditioning_maximum
no_reward_convergence = args.no_reward_convergence
ptp_stop = args.ptp_stop
n_trials = args.n_trials
flat_tongue = args.flat_tongue
debug = args.debug
constant_sigma0 = args.constant_sigma0
default_settings = args.default_settings

N = len(parameters)                 # number of dimensions

if n_workers == 1:      # serial mode -> disable parallel features
    lambda_list = [4,6,7,8,8,9,9,10,10,10,11,11,11,11,12,12,12,12]
        # list of recommended lambda values for given number of
        #  dimenions (see Hansen)                    
    lambda_ = lambda_list[N-1]

if default_settings:    # shortcut for default settings
                        # -> get rid of alternative settings?
    infant = True
    intrinsic_motivation = True
    resample = True
    normalize = True
    random_restart = True
    no_reward_convergence = True
    no_convergence = True

if infant:                          # declare speaker
    speaker = 'infant'
else:
    speaker = 'adult'





###########################################################
#
# functions
#
###########################################################


def get_abs_coord(x):
    """
    function for coordinate transformation from relative to absolute coordinates
    -> computations take place in relative coordinates (boundary conditions), 
        the vocal tract model uses absolute coordinates.
     - argument x: numpy.array of length 16, contains relative coordinates in [0,1]
     - global infant: boolean defining if infant coordinate system is used
     - output abs_coord: numpy.array of length 16, contains absolute coordinates
    """
    global infant

    if infant:                      # case: agent uses infant system
        low_boundaries = np.array([0.0, -3.228, -7.0, -1.0, -1.102, 0.0, -3.194, -1.574, 0.873, -1.5744, -3.194, -1.574, -1.081, -1.081, -1.081, -1.081])
                                    # lowest physiological parameter values
        high_boundaries = np.array([1.0, -1.85, 0.0, 1.0, 2.205, 1.0, 2.327, 0.63, 3.2, 1.457, 2.327, 2.835, 1.081, 1.081, 1.081, 1.081])
                                    # highest physiological parameter values
    else:                           # case: agent uses adult system
        low_boundaries = np.array([0.0, -6.0, -7.0, -1.0, -2.0, 0.0, -3.0, -3.0, 1.5, -3.0, -3.0, -3.0, -1.4, -1.4, -1.4, -1.4])
        high_boundaries = np.array([1.0, -3.5, 0.0, 1.0, 4.0, 1.0, 4.0, 1.0, 5.5, 2.5, 4.0, 5.0, 1.4, 1.4, 1.4, 1.4])

    abs_coord = np.zeros(16)        # prepare output
    for i in xrange(16):            # loop over all coordinates
        abs_coord[i] = low_boundaries[i] + x[i] * (high_boundaries[i] - low_boundaries[i])
                                    # coordinate transformation

    return abs_coord


###########################################################


def get_rel_coord(x):
    """
    function for coordinate transformation from absolute to relative coordinates
    -> computations take place in relative coordinates (boundary conditions), 
        the vocal tract model uses absolute coordinates.
     - argument x: numpy.array of length 16, contains absolute coordinates
     - global infant: boolean defining if infant coordinate system is used
     - output rel_coord: numpy.array of length 16, contains relative coordinates
    """
    global infant

    if infant:                      # case: agent uses infant system
        low_boundaries = np.array([0.0, -3.228, -7.0, -1.0, -1.102, 0.0, -3.194, -1.574, 0.873, -1.5744, -3.194, -1.574, -1.081, -1.081, -1.081, -1.081])
                                    # lowest physiological parameter values
        high_boundaries = np.array([1.0, -1.85, 0.0, 1.0, 2.205, 1.0, 2.327, 0.63, 3.2, 1.457, 2.327, 2.835, 1.081, 1.081, 1.081, 1.081])
                                    # highest physiological parameter values
    else:                           # case: agent uses adult system
        low_boundaries = np.array([0.0, -6.0, -7.0, -1.0, -2.0, 0.0, -3.0, -3.0, 1.5, -3.0, -3.0, -3.0, -1.4, -1.4, -1.4, -1.4])
        high_boundaries = np.array([1.0, -3.5, 0.0, 1.0, 4.0, 1.0, 4.0, 1.0, 5.5, 2.5, 4.0, 5.0, 1.4, 1.4, 1.4, 1.4])

    rel_coord = np.zeros(16)        # prepare output
    for i in xrange(16):            # loop over all coordinates
        rel_coord[i] = (x[i] - low_boundaries[i]) / (high_boundaries[i] - low_boundaries[i])
                                    # coordinate transformation
                                    # -> for each coordinate: 0 is lowest physiological value, 1 is highest physiological value
                                    # -> [0,1]^16 cube is search domain
    return rel_coord


###########################################################
### prepare motor parameters
###########################################################


params_i = np.array([0.8580, -2.9237, -1.8808, -0.0321, 0.5695, 0.1438, 0.6562, -0.3901, 2.6431, -0.6510, 2.1213, 0.3124, 0.3674, 0.034, -0.1274, -0.2887])
params_u = np.array([0.9073, -3.2279, -4.0217, 1.0, 0.3882, 0.5847, 0.3150, -0.5707, 2.0209, -1.0122, 1.8202, -0.1492, 0.5620, 0.1637, 0.0602, -0.0386])
params_a = np.array([0.3296, -2.3640, -4.3032, 0.0994, 0.8196, 1.0, -0.4878, -1.2129, 1.9036, -1.5744, 1.3212, -1.0896, 1.0313, -0.1359, 0.4925, 0.0772])
params_schwa = np.array([1.0, -2.643, -2.0, -0.07, 0.524, 0.0, -0.426, -0.767, 2.036, -0.578, 1.163, 0.321, 0.0, 0.046, 0.116, 0.116])

params_ad_e = np.array([0.44, -4.0949, -2.6336, -0.1571, 0.7513, 0.4073, 2.5611, -0.836, 4.6531, -0.954, 4.0, 0.3971, -0.2859, -0.986, 1.288, 0.364, 0.084, 0.05])
params_schwa_ad = np.array([0.6259, -4.8156, -4.6286, 0.1203, 0.6552, 0.5709, 1.3037, -1.9642, 4.8069, -1.019, 3.3425, -0.3516, -1.7212, -1.9642, 0.752, 0.696, 0.9, 0.256])
params_ad_a = np.array([0.3296, -4.2577, -4.3032, 0.0994, 0.8095, 1.0, -0.1154, -2.0006, 4.2957, -1.3323, 3.0737, -0.3857, -2.8066, -2.9354, 1.336, -0.176, 0.638, 0.1])
params_ad_i = np.array([0.858, -5.2436, -1.8808, -0.091, 0.6751, 0.1438, 2.5295, -0.5805, 4.6333, -0.8665, 3.9, 0.646, -0.17, -0.7805, 1.316, 0.044, -0.165, -0.374])
params_ad_o = np.array([0.5314, -5.1526, -6.2288, 0.7435, 0.1846, 0.503, -0.1834, -1.0274, 2.4269, -1.1931, 2.0194, 0.1551, -1.3385, -2.6564, 0.556, -0.392, 0.0, 0.05])
params_ad_u = np.array([1.0, -5.6308, -4.0217, 1.0, 0.2233, 0.5847, 0.6343, -0.9421, 2.7891, -0.694, 2.4431, 0.3572, -1.3615, -2.8154, 1.4, 0.212, 0.078, -0.05])
                                    # set of known parameters, ad refers to adult parameters, rest refers to infant parameters
                                    # -> absolute coordinates

par_table = {'HX':0, 'HY':1, 'JA':2, 'LP':3, 'LD':4, 'VS':5, 'TCX':6, 'TCY':7, 'TTX':8, 'TTY':9, 'TBX':10, 'TBY':11, 'TS1':12, 'TS2':13, 'TS3':14, 'TS4':15}
                                    # look up dictionary for parameter indices
                                    #  -> complete set of all parameters are handed to the environment
                                    #  -> need to know the indices of the learnt parameters

parameters_indices = []             # prepare list for parameter indices
if rank==1:
    print 'parameters:', parameters

if parameters == ['all']:           # case: full-dimensional problem
    parameters_indices = range(16)  # all 16 parameters are being learnt
    N = 16                          # change dimension from 1 to 16 
elif parameters == ['flat']:
    parameters_indices = range(12)
    N = 12
else:                               # case: only some parameters are begin learnt, rest is fixed
  for par in parameters:            # loop over all parameters
    try:
        parameters_indices.append(par_table[par])
                                    # use look up dictionary to find indices of learnt parameters
    except KeyError:
        print 'Error: parameter '+str(par)+' not found!'
if verbose and rank==1:
    print 'parameters_indices:', parameters_indices

if intrinsic_motivation:            # case: all target vowels are being learnt, agent chooses sequence of targets
    i_target = -1                   # "i_target = -1" means "no current target. searching for next target"
    if infant:
        params = params_schwa.copy() # initial condition: use parameters of schwa vowel (@)
    else:
        params = params_schwa_ad.copy()
elif target == 'a':                 # case: only one specified target is being learnt
    if infant:
        params = params_a.copy()    # initial condition: use parameters of target vowel
                                    # -> learnt parameters are changed, rest of vocal tract is in target configuration
                                    # -> ensures target can actually be reproduced
    else:
        params = params_ad_a.copy()
    i_target = 0
elif target == 'i':
    if infant:
        params = params_i.copy()
    else:
        params = params_ad_i.copy()
    i_target = 2
elif target == 'u':
    if infant:
        params = params_u.copy()
    else:
        params = params_ad_u.copy()
    i_target = 1
elif target == 'o':                 # prespecified configurations of /o/ and /e/ exist only in adult system
    params = params_ad_o.copy()
    i_target = 3
elif target == 'e':
    params = params_ad_e.copy()
    i_target = 4
else:
    print 'Target vowel', target, 'not recognized!'

params_r_target = get_rel_coord(params)
                                    # transform current parameters into relative coordinates
                                    # -> only relevant if intrinsic_motivation = False:
                                    #       these are the relative coordinates of the mentor configuration for target vowel
                                    # -> used for determining the distance between learnt configuration and mentor configuration

if not predefined:                  # case: initial condition is schwa @
                                    # -> predefined: start with mentor configuration from VTL
    if infant:
        for i in parameters_indices: # loop over all learnt parameters
            params[i] = params_schwa[i]
                                    # initial condition: set learnt parameters to schwa @
    else:
        for i in parameters_indices:
            params[i] = params_schwa_ad[i]

params_r = get_rel_coord(params)    # transform parameters into relative coordinates
                                    # -> search space should be physiological space, i.e. [0,1]^N cube
if verbose and rank==1:
    print 'params:', params


###########################################################


def get_output_folder(subfolder):
    global rank
    """ 
    function for creating and returning directory of current simulation
     - argument subfolder: str, current simulation name, subfolder under current date
     - global rank: int, identity of current worker
        -> only master creates new folders
     - output outputpath: str, subfolder for current simulation"""

    today = date.today()                  # get system date
    today_string = today.isoformat()      # convert date to string
    outputpath_short = 'output/'+today_string
                                        # date yields super folder
    outputpath = 'output/'+today_string+'/'+subfolder+'/'
                                        # date + subfolder yields working folder
    if rank==0:                         # master attempts to create folders
        try:
            os.system('mkdir '+outputpath_short)
            os.system('mkdir '+outputpath) # create super folder, then subfolder
            os.system('mkdir '+outputpath+'data/')
        finally:
            pass
    
    return outputpath


###########################################################


def parallel_evaluation(x_mean, sigma, B_D, i_count, i_target):
    """
    function for communication between master and slaves
    -> interface of agent and environment, called during each environment evaluation
    -> executed only by master
     - arguments:
      - x_mean: numpy.array of length N, mean of current sampling distribution
      - sigma: float, width of current sampling distribution
      - B_D: numpy.array of shape (N,N), covariance matrix of current sampling distribution
      - i_count: int, current iteration step
      - i_target: int, index of current target vowel      
     - globals:
      - n_workers: int, number of worker (=slaves+master)
      - verbose: bool, for printed stuff
      - n_vowels: int, total number of target vowels
     - outputs:
      - z: numpy.array of shape (lambda, N), sampled z values of each slave for each coordinate
      - x: numpy.array of shape (lambda, N), corresponding parameter values of each slave for each coordinate
      - confidences: numpy.array of shape (lambda, n_vowels+1), corresponding confidence levels for each target vowel + null class
      - energy_cost: numpy.array of length lambda, corresponding energy penalty for each slave's sample
      - boundary_penalty: numpy.array of length lambda, corresponding boundary penalty for each slave's sample
    """

    global n_workers, verbose, n_vowels, lambda_, N

    items_broadcast = x_mean,sigma,B_D,i_count,i_target
                                        # whatever the master distributes to the slaves
    tag = int(i_count/(n_workers-1))    # each transmission carries a specific tag to identify the corresponding slave

    if N_reservoir > 20:
        confidences = np.zeros([n_workers-1,n_vowels+1])
    else:
        confidences = np.zeros([n_workers-1,n_vowels])
    energy_cost = np.zeros(n_workers-1)
    boundary_penalty = np.zeros(n_workers-1)
    z = np.zeros([lambda_, N])
    x = np.zeros([lambda_, N])
    N_resampled = np.zeros([lambda_, N], dtype=int)

    print 'current tag (master):', tag


    for i_worker in xrange(1,n_workers):
        comm.send(items_broadcast, dest=i_worker, tag=tag)
    for i_worker in xrange(1,n_workers):

        z[i_worker-1],x[i_worker-1],confidences[i_worker-1],energy_cost[i_worker-1],boundary_penalty[i_worker-1],N_resampled[i_worker-1] = comm.recv(source=i_worker, tag=tag)

    N_resampled_sum = N_resampled.sum()
    if verbose:
        print N_resampled_sum, 'samples rejected'

    return z, x, confidences, energy_cost, boundary_penalty, N_resampled_sum





def serial_evaluation(x_mean, sigma, B_D, i_count, i_target):
    """
    evaluate rewards if only one worker exists
    -> interface of agent and environment, called during each environment evaluation
    -> executed only by master
     - arguments:
      - x_mean: numpy.array of length N, mean of current sampling distribution
      - sigma: float, width of current sampling distribution
      - B_D: numpy.array of shape (N,N), covariance matrix of current sampling distribution
      - i_count: int, current iteration step
      - i_target: int, index of current target vowel      
     - globals:
      - n_workers: int, number of worker (=slaves+master)
      - verbose: bool, for printed stuff
      - n_vowels: int, total number of target vowels
     - outputs:
      - z: numpy.array of shape (lambda, N), sampled z values of each slave for each coordinate
      - x: numpy.array of shape (lambda, N), corresponding parameter values of each slave for each coordinate
      - confidences: numpy.array of shape (lambda, n_vowels+1), corresponding confidence levels for each target vowel + null class
      - energy_cost: numpy.array of length lambda, corresponding energy penalty for each slave's sample
      - boundary_penalty: numpy.array of length lambda, corresponding boundary penalty for each slave's sample
    """

    global verbose, n_vowels, lambda_, N


    confidences = np.zeros([lambda_,n_vowels+1])
    energy_cost = np.zeros(lambda_)
    boundary_penalty = np.zeros(lambda_)
    z = np.zeros([lambda_, N])
    x = np.zeros([lambda_, N])
    N_resampled = -lambda_

    for i in xrange(lambda_):   # offspring generation loop

        invalid = True
        print 'sampling parameters...'
        if resample:
          while invalid:
            N_resampled += 1
            z_i = np.random.randn(N)      # standard normally distributed vector
            x_i = x_mean + sigma*(np.dot(B_D, z_i))  # add mutation, Eq. 37
            invalid = (x_i < 0.0).any() or (x_i > 1.0).any()

          boundary_penalty_i = 0.0
        else:
          N_resampled = 0
          z_i = np.random.randn(N)          # standard normally distributed vector
          x_i = x_mean + sigma*(np.dot(B_D, z_i)) # add mutation, Eq. 37
          boundary_penalty_i = 0.0
          if (x_i<0.0).any() or (x_i>1.0).any(): # check boundary condition
            if verbose:
                print 'boundary violated. repairing and penalizing.'
            x_repaired = x_i.copy()       # repair sample
            for i_component in xrange(len(x_i)):
                if x_i[i_component] > 1.0:
                    x_repaired[i_component] = 1.0
                elif x_i[i_component] < 0.0:
                    x_repaired[i_component] = 0.0
            boundary_penalty_i = np.linalg.norm(x_i-x_repaired)**2
                                        # penalize boundary violation, Eq. 51
            x_i = x_repaired

        z[i] = z_i
        x[i] = x_i
        boundary_penalty[i] = boundary_penalty_i
 
        params_full = get_full_parameters(x_i, i_target)
        energy_cost[i] = get_energy_cost(params_full)   
        params_abs = get_abs_coord(params_full)

        confidences[i] = evaluate_environment(params_abs, i_count, simulation_name=folder, outputfolder=outputfolder, i_target=i_target, rank=rank, speaker=speaker, n_vow=n_vowels, normalize=normalize)

        # end of offspring generation loop

    if verbose:
        print N_resampled_sum, 'samples rejected'

    return z, x, confidences, energy_cost, boundary_penalty, N_resampled









def get_next_target(confidences, indices_learnt):
    global n_vowels, verbose

    confidences_flat = confidences.flatten()
    confidences_argsort = confidences_flat.argsort()
    for i in xrange(len(confidences_argsort)-1, -1, -1):
        i_next = np.mod(confidences_argsort[i], n_vowels+1)
        if not i_next in indices_learnt:
            break

    if verbose:
        print 'confidences:', confidences
        print 'indices_learnt:', indices_learnt
        print 'i_next:', i_next

    return i_next





def get_deviations(params_r):
    global params_i, params_u, params_a, params_ad_e, params_ad_a, params_ad_i, params_ad_o, params_ad_u, infant

    if infant:
        params_targets = [params_a, params_i, params_u]
    else:
        params_targets = [params_ad_a, params_ad_u, params_ad_i, params_ad_e, params_ad_o]

    n = len(params_targets)
    deviations = np.zeros(n)
    for i in xrange(n):
        deviations[i] = norm(params_r - get_rel_coord(params_targets[i]))

    return deviations




def get_energy_cost(params_r):
    global params_schwa, params_schwa_ad, infant

    if infant:
        params_neutral = params_schwa
    else:
        params_neutral = params_schwa_ad

    deviation = norm(params_r - get_rel_coord(params_neutral))

    return deviation

    



def get_full_parameters(x, i_target):
    global params_i, params_u, params_a, params_ad_e, params_ad_a, params_ad_i, params_ad_o, params_ad_u, infant, parameters_indices,\
        flat_tongue

    if infant:
        params_targets = [params_a, params_i, params_u]
    else:
        params_targets = [params_ad_a, params_ad_u, params_ad_i, params_ad_e, params_ad_o]
        
    params = get_rel_coord(params_targets[i_target])
    if flat_tongue:
        for i in xrange(-4,0):
            params[i] = 0.5

    for i in xrange(len(x)):
        params[parameters_indices[i]] = x[i]

    return params



def save_state(state, flag):
    global outputfolder    

    save_file = outputfolder+'save.'+flag
    os.system('rm '+save_file)
    os.system('touch '+save_file)
    save_file_write = open(save_file, 'w')
    cPickle.dump(state, save_file_write)
    save_file_write.close()



def load_saved_state(load_state):
    inputfile_dynamic = open(load_state+'.dyn', 'r')
    inputfile_static = open(load_state+'.stat', 'r')
   
    static_params = cPickle.load(inputfile_static)
    dynamic_params = cPickle.load(inputfile_dynamic)

    inputfile_static.close()
    inputfile_dynamic.close()

    return static_params, dynamic_params
    



def now():
    return strftime("%d %b %H:%M:%S", localtime())




def cmaes():                        # actual CMA-ES part
    global N, verbose, params_r, n_workers, rank, parameters_indices, sigma0, params_r_target, conf_threshold, intrinsic_motivation,\
        i_target, n_vowels, energy_factor, alpha, N_reservoir, no_convergence, outputfolder, random_restart, load_state, cond_stop,\
        i_start, no_reward_convergence, ptp_stop, speaker, flat_tongue, record_scalars, record_B_D, record_z, record_x,\
        record_confidences, record_fitness, record_params_abs, record_x_mean, record_params_r, record_p_c, record_p_s, record_C,\
        record_D, record_fitness_recent, record_ptp_fitness_recent, record_x_recent, record_ptp_x_recent, debug, records,\
        constant_sigma0

    #######################################################
    # Initialization
    #######################################################

    print 'main function starts at:', datetime.datetime.now()

    outputfolder_old = None
    if not load_state == None:
        static_params, dynamic_params = load_saved_state(load_state)
        [N, verbose, params_r, parameters_indices, sigma0, params_r_target, conf_threshold, intrinsic_motivation, n_vowels, 
            energy_factor, alpha, no_convergence, outputfolder_old, random_restart, cond_stop, flat_tongue] = static_params
        print 'loaded static parameters:'
        print 'N:', N, '\nverbose:', verbose, '\nparams_r:', params_r, '\nparameters_indices:', parameters_indices, '\nsigma0:',\
            sigma0, '\nparams_r_target:', params_r_target, '\nconf_threshold:', conf_threshold, '\nintrinsic_motivation:',\
            intrinsic_motivation, '\nn_vowels:', n_vowels, '\nenergy_factor:', energy_factor, '\nalpha:', alpha,\
            '\nno_convergence:', no_convergence, '\noutputfolder_old:', outputfolder_old, '\nrandom_restart:',\
            random_restart, '\ncond_stop:', cond_stop, '\nflat_tongue:', flat_tongue


    sigma = sigma0
    current_sigma0 = sigma0
    x_mean = []
    x_recent = deque()
    fitness_recent = deque()
    for i in parameters_indices:
        x_mean.append(params_r[i])
    if verbose:
        print 'parameters_indices:', parameters_indices
        print 'x_mean:', x_mean
    x_recent.append(x_mean)
    print '[507] x_recent:', x_recent
    print '[508] fitness_recent:', fitness_recent
    vowel_list = ['a', 'i', 'u', '@']
    motor_memory = []
                                    # population size, offspring number
    if verbose:
        print 'lambda =', lambda_, ',  n_workers =', n_workers-1

    mu_ = lambda_ / 2.0             # mu_ is float
    mu = int(np.floor(mu_))         # mu is integer = number of parents/points for recombination
    weights = np.zeros(mu)
    for i in xrange(mu):            # muXone recombination weights
        weights[i] = np.log(mu_+0.5) - np.log(i+1)
    weights /= sum(weights)         # normalize recombination weights array 
    mu_eff = sum(weights)**2 / sum(weights**2)
                                    # variance-effective size of mu
    print '[503] mu_eff:', mu_eff

    convergence_interval = int(10+np.ceil(30.0*N/lambda_))       # window for convergence test
    i_reset = 0

    # Strategy parameter setting: Adaptation
    c_c = (4.0+mu_eff/N) / (N+4.0+2*mu_eff/N)
                                    # time constant for cumulation for C
    print '[511] c_c:', c_c
    c_s = (mu_eff+2.0) / (N+mu_eff+5.0)
                                    # time constant for cumulation for sigma control
    print '[516] c_s:', c_s
    c_1 = 2.0 / ((N+1.3)**2 + mu_eff) # learning rate for rank-one update of C
    print '[518] c_1:', c_1
    c_mu = 2 * (mu_eff-2.0+1.0/mu_eff) / ((N+2.0)**2 + 2*mu_eff/2.0)
                                    # and for rank-mu update
    print '[521] c_mu:', c_mu
    print '[522] mu_eff:', mu_eff, '\nN:', N
    damps = 1.0 + 2*np.max([0, np.sqrt((mu_eff-1.0)/(N+1.0))-1.0]) + c_s
                                    # damping for sigma
    

    # Initialize dynamic (internal) strategy parameters and constants
    p_c = np.zeros(N)               # evolution path for C
    p_s = np.zeros(N)               # evolution path for sigma
    B = np.eye(N)                   # B defines the coordinate system
    D = np.eye(N)                   # diagonal matrix D defines the scaling
    B_D = np.dot(B,D)
    C = np.dot(B_D, (B_D).T)               # covariance matrix
    i_eigen = 0                     # for updating B and D
    chi_N = np.sqrt(N) * (1.0-1.0/(4.0*N) + 1.0/(21.0*N**2))
                                    # expectation of ||N(0,I)|| == norm(randn(N,1))
    print '[537] chi_N:', chi_N

    # Initialize arrays
    fitness = np.zeros(lambda_)
    indices_learnt = [n_vowels]     # last vowel index corresponds to null class
    x_learnt = [[x_mean, 3]]


    #######################################################
    # Output preparation
    #######################################################


    output_write.write('initial conditions: time=('+str(datetime.datetime.now())+') N='+str(N)+', lambda='+str(lambda_)+', x=')
    for x_ in x_mean:
        output_write.write(str(x_)+' ')
    output_write.write(', distance='+str(norm(params_r-params_r_target)))
    output_write.write(', sigma='+str(sigma))
    output_write.write(', energy_factor='+str(energy_factor))
    output_write.write(', alpha='+str(alpha))
    output_write.write(', conf_threshold='+str(conf_threshold))
    output_write.write(', cond_stop='+str(cond_stop))
    output_write.write('\n')
    output_write.write('time    sampling step   mean fitness   i_target   ')
    for i in xrange(n_vowels):
        output_write.write('confidence['+str(i)+']   ')
    for i in xrange(n_vowels):
        output_write.write('motor deviation['+str(i)+']   ')
    output_write.write('energy_cost boundary_penalty sigma N_resampled\n')

    if debug:
        record_scalars.write('i_count   i_target   fitness_mean   h_sig   sigma   cond   N_resampled\n\n')

    static_params = [N, verbose, params_r, parameters_indices, sigma0, params_r_target, conf_threshold, intrinsic_motivation, 
        n_vowels, energy_factor, alpha, no_convergence, outputfolder, random_restart, cond_stop, flat_tongue]
    save_state(static_params, 'stat')



    #######################################################
    # Generation Loop
    #######################################################

    error = False
    fitness_mean = 0.0

    i_count = 0                     # the next 40 lines contain the 20 lines of interesting code

    if not load_state == None:
      print 'loading state', load_state
      [current_time, i_count, i_target, p_s , p_c, C, i_eigen, sigma, x_recent, fitness_recent, i_reset, current_sigma0, x_learnt,\
        indices_learnt, B_D, x_mean] = dynamic_params
      print 'current_time:', current_time, '\ni_count:', i_count, '\ni_target:', i_target, '\np_s:', p_s, '\np_c:', p_c, '\nC:', C,\
        '\ni_eigen:', i_eigen, '\nsigma:', sigma, '\nx_recent:', x_recent, '\nfitness_recent:', fitness_recent, '\ni_reset:',\
        i_reset, '\ncurrent_sigma0:', current_sigma0, '\nx_learnt:', x_learnt, '\nindices_learnt:', indices_learnt, '\nB_D:', B_D,\
        '\nx_mean:', x_mean
      i_start = i_count
      print 'i_start = i_count =', i_start
      for i_worker in xrange(1,n_workers):
          comm.send(i_start, dest=i_worker)

    return_dict = {'a_steps':0, 'a_time':0, 'a_reward':0,'i_steps':0, 'i_time':0, 'i_reward':0,'u_steps':0, 'u_time':0, 'u_reward':0,\
        'steps':0, 'time':0}

    t_0 = datetime.datetime.now()
    t_reset = t_0

    while True:

      # Generate and evaluate lambda offspring

      if n_workers == 1:
          z, x, confidences, energy_cost, boundary_penalty, N_resampled_trial = serial_evaluation(x_mean, 
            sigma, B_D, i_count, i_target)
      else:
          z, x, confidences, energy_cost, boundary_penalty, N_resampled_trial = parallel_evaluation(x_mean, 
            sigma, B_D, i_count, i_target)

      i_count += lambda_       

      if i_target == -1:
        idle = True
        i_target = get_next_target(confidences, indices_learnt)
      else:
        fitness = -confidences.T[i_target]+energy_factor*energy_cost+alpha*boundary_penalty

        if no_convergence and (fitness < -conf_threshold).any():
                i_argmax = fitness.argmin()
                x_mean = x[i_argmax]
                x_learnt.append([x_mean, i_target])

                indices_learnt.append(i_target)

                params_full = get_full_parameters(x_mean, i_target)
                params_abs = get_abs_coord(params_full)

                motor_memory.append([vowel_list[i_target], params_abs])

                results_write.write(vowel_list[i_target]+'    '+str(i_count)+'    '+str(-fitness[i_argmax])+'    '+now()+'\n')
                results_write.write('relative coordinates:\n '+str(params_full)+'\n')
                results_write.write('absolute coordinates:\n '+str(params_abs)+'\n\n')
                results_write.flush()

                current_vowel = vowel_list[i_target]
                return_dict[current_vowel+'_steps'] = i_count-i_reset
                return_dict[current_vowel+'_time'] = datetime.datetime.now()-t_reset
                return_dict[current_vowel+'_reward'] = -fitness[i_argmax]

                i_reset = i_count
                t_reset = datetime.datetime.now()


                os.system('cp '+outputfolder+'data/vowel_'+str(i_target)+'_'+str(i_argmax+1)+'.wav '+outputfolder+'vowel_'+str(i_target)+'.wav')
                os.system('cp '+outputfolder+'data/vowel_'+str(i_target)+'_'+str(i_argmax+1)+'.png '+outputfolder+'vowel_'+str(i_target)+'.png')
                os.system('cp VTL_API/speakers/'+speaker+'_'+folder+'_input_'+str(i_argmax+1)+'.speaker '+outputfolder+'vowel_'+str(i_target)+'.speaker')

                if verbose:
                    print '[589] x_learnt:', x_learnt
                    print '[591] confidence:', -fitness[i_argmax], ', i_reset:', i_reset
                    print '[595] params_full:', params_full
                    print 'x:', x
                print 'iteration:',i_count,', now:', datetime.datetime.now(), ', i_target:', i_target, ', reward:', -fitness[i_argmax], ', parameter:',x[i_argmax]


                if (len(indices_learnt) < n_vowels+1) and intrinsic_motivation:
                    i_target = -1
                    p_c = np.zeros(N)        
                    p_s = np.zeros(N)        
                    B = np.eye(N)             
                    D = np.eye(N)   
                    B_D = np.dot(B,D)        
                    C = np.dot(B_D, (B_D).T)
                    i_eigen = 0
                    current_sigma0 = sigma0               
                    sigma = current_sigma0
                    i_reset = 0

                else:
                    print 'terminating.'
                    print 'i_reset:', i_reset, ', confidence:', -fitness[i_argmax]
                    tag = int(i_count/(n_workers-1))
                    for i_worker in xrange(1,n_workers):
                        comm.send((None,None,None,None,None), dest=i_worker, tag=tag)
                    break

        else:

            # Sort by fitness and compute weighted mean into x_mean
            indices = np.arange(lambda_)
            to_sort = zip(fitness, indices)
                                        # minimization
            to_sort.sort()
            fitness, indices = zip(*to_sort)
            fitness = np.array(fitness)
            indices = np.array(indices)
            x_mean = np.zeros(N)
            z_mean = np.zeros(N)
            fitness_mean = 0.0
            for i in xrange(mu):
                x_mean += weights[i] * x[indices[i]]
                                        # recombination, Eq. 39
                z_mean += weights[i] * z[indices[i]]
                                        # == D^-1 * B^T * (x_mean-x_old)/sigma
                fitness_mean += weights[i] * fitness[indices[i]]

            for j in xrange(len(x_mean)):
                params_r[parameters_indices[j]] = x_mean[j]

            deviations = get_deviations(params_r)

    
            # Output
            params_full = get_full_parameters(x_mean, i_target)
            params_abs = get_abs_coord(params_full)

            output_write.write(str(datetime.datetime.now()-t_0)+'  '+str(i_count)+'  '+str(-fitness_mean)+'  '+str(i_target)+'  ')
            for confidence in confidences[0]:
                output_write.write(str(confidence)+'  ')
            for deviation in deviations:
                output_write.write(str(deviation)+'  ')
            output_write.write(str(energy_cost[0])+'  '+str(boundary_penalty[0])+'  '+str(sigma)+'  '+str(N_resampled_trial)+'\n')
            output_write.write('    rel coords: '+str(params_full)+'\n')
            output_write.write('    abs coords: '+str(params_abs)+'\n\n')
            output_write.flush()
            
            # Cumulation: Update evolution paths
            p_s = (1.0-c_s)*p_s + (np.sqrt(c_s*(2.0-c_s)*mu_eff)) * np.dot(B,z_mean)
                                        # Eq. 40
            h_sig = int(np.linalg.norm(p_s) / np.sqrt(1.0-(1.0-c_s)**(2.0*i_count/lambda_))/chi_N < 1.4+2.0/(N+1.0))
            p_c = (1.0-c_c)*p_c + h_sig * np.sqrt(c_c*(2.0-c_c)*mu_eff) * np.dot(B_D,z_mean)
                                        # Eq. 42

    
            # Adapt covariance matrix C
           
            C_new = (1.0-c_1-c_mu)*C + c_1*(np.dot(p_c,p_c.T) + (1.0-h_sig)*c_c*(2.0-c_c)*C) + c_mu*np.dot(np.dot((np.dot(B_D, z[indices[:mu]].T)),np.diag(weights)),(np.dot(B_D, z[indices[:mu]].T)).T)

            if not (np.isfinite(C_new)).all():
                print 'Warning! C contains invalid elements!'
                error = True
            else:
                C = C_new               # regard old matrix plus rank one update plus minor correction plus rank mu update, Eq. 43
    
            # Adapt step-size sigma
            sigma = sigma * np.exp((c_s/damps) * (np.linalg.norm(p_s)/chi_N - 1.0))
                                        # Eq. 41

            # Update B and D from C
            if i_count - i_eigen > lambda_/(c_1+c_mu)/N/10.0:
                                        # to achieve O(N**2)
                i_eigen = i_count
                C_new = np.triu(C) + np.triu(C,1).T
                                        # enforce symmetry
                cond = np.linalg.cond(C_new)
                if not (np.isfinite(C_new)).all():# or (C_new < 0.0).any()):
                    print 'Warning! C contains invalid elements!'
                    print 'C:', C_new
                    print 'repaired to C=', C
                    print 'conditioning number of C:', cond
                    error = True
                else:
                    C = C_new

                if (np.iscomplex(C)).any():
                    print 'Warning! C contains complex elements!'
                    print 'C:', C
                    print 'conditioning number of C:', cond
                    error = True

                D, B = np.linalg.eig(C) # eigen decomposition, B==normalized eigenvectors?
                if verbose:
                    print '[713] D:', D, '\nB:', B
                if (D < 0.0).any():
                    print 'Warning! D contains negative elements!'
                    for i in xrange(len(D)):
                        if D[i] < 0.0:
                            D[i] = -D[i]
                            print -D[i], 'repaired to', D[i]
                D = np.diag(np.sqrt(D)) # D contains standard deviations now
                B_D = np.dot(B,D)


            # Escape flat fitness, or better terminate?
            print 'fitness:', fitness
            if fitness[0] == fitness[int(np.ceil(0.7*lambda_))]:
                sigma *= np.exp(0.2+c_s/damps)
                print 'warning: flat fitness, consider reformulating the objective'


            while len(x_recent) > convergence_interval - 1:
                x_recent.popleft()
            while len(fitness_recent) > convergence_interval - 1:
                fitness_recent.popleft()
            x_recent.append(x_mean)
            fitness_recent.append(fitness_mean)

            cond = np.linalg.cond(C)
            if verbose:
                print '[579] B_D:', B_D
                print '[573] z:', z, '\nx:', x, '\nconfidences:', confidences, '\nboundary_penalty:', boundary_penalty
                print '[580] confidences:', confidences
                print '[582] fitness:', fitness
                print '[597] params_abs:', params_abs
                print '[650] x_mean:', x_mean, 'z_mean:', z_mean, 'fitness_mean:', fitness_mean
                print '[653] params_r:', params_r
                print '[655] deviations:', deviations
                print '[671] p_s:', p_s
                print 'h_sig:', h_sig
                print 'p_c:', p_c
                print '[681] C_new:', C_new
                print '[696] sigma:', sigma
                print '[716] D:', D
                print '[794] fitness_recent:', fitness_recent
                print '[796] np.ptp(fitness_recent):', np.ptp(fitness_recent)
                print '[791] x_recent:', x_recent
                print '[793] np.ptp(x_recent, axis=0):', np.ptp(x_recent, axis=0)
                print '[785] len(x_recent):', len(x_recent)
                print '[786] convergence_interval - 1:', convergence_interval - 1
                print '[747] cond(C):', cond

            if no_reward_convergence:
                termination = (np.ptp(x_recent, axis=0) < ptp_stop).all() or (cond > cond_stop)
            else:
                termination = ((np.ptp(x_recent, axis=0) < ptp_stop).all()) and (np.ptp(fitness_recent) < ptp_stop) or (cond > cond_stop)

            if termination:
                print 'convergence criterion reached.'
                if (fitness[0] > -conf_threshold): # confidence worse than desired
                    print 'reward too low. resetting sampling distribution.'
                    print 'reward', -fitness[0], '<', conf_threshold
                    p_c = np.zeros(N)        
                    p_s = np.zeros(N)        
                    B = np.eye(N)             
                    D = np.eye(N)
                    B_D = np.dot(B, D)           
                    C = np.dot(B_D, (B_D).T)
                    i_eigen = 0

                    if random_restart:
                        if current_sigma0 < 0.9 and not constant_sigma0:
                            current_sigma0 += 0.05              
                        sigma = current_sigma0
                        print 'sigma set to', sigma
                    
                        random_index = np.random.randint(len(x_learnt))
                        x_mean = x_learnt[random_index][0]
                        print 'agent chose to restart search from learnt parameters of', vowel_list[x_learnt[random_index][1]]

                    else:
                        if current_sigma0 < 0.9 and not constant_sigma0:
                            current_sigma0 += 0.1              
                        sigma = current_sigma0
                        print 'sigma set to', sigma


                else:
                    print 'confidence:', -fitness[0], ', i_reset:', i_reset
                    indices_learnt.append(i_target)
    
                    if (len(indices_learnt) < n_vowels+1) and intrinsic_motivation:
                        i_target = -1
                        p_c = np.zeros(N)        
                        p_s = np.zeros(N)        
                        B = np.eye(N)             
                        D = np.eye(N)           
                        C = np.dot(B_D, (B_D).T)    
                        i_eigen = 0
                        current_sigma0 = sigma0               
                        sigma = current_sigma0
                        i_reset = 0
                    else:
                        print 'terminating.'
                        print 'i_reset:', i_reset, ', confidence:', -fitness[0]
                        tag = int(i_count/(n_workers-1))
                        for i_worker in xrange(1,n_workers):
                            comm.send((None,None,None,None,None), dest=i_worker, tag=tag)
                        break

        if error:
            print 'Critical error occurred!\nTerminating simulations.'
            tag = int(i_count/(n_workers-1))
            for i_worker in xrange(1,n_workers):
                comm.send((None,None,None,None,None), dest=i_worker, tag=tag)
            break
        
        if verbose:
            print 'x:', x
        print 'iteration:',i_count,', time elapsed:', datetime.datetime.now()-t_0, ', i_target:', i_target, ', reward:', -fitness_mean, ', parameter:',x_mean

        current_time = datetime.datetime.now()
        dynamic_params = [current_time, i_count, i_target, p_s , p_c, C, i_eigen, sigma, x_recent, fitness_recent, i_reset, current_sigma0, x_learnt, indices_learnt, B_D, x_mean]
        save_state(dynamic_params, 'dyn')
        if verbose:
            print 'saved dynamic parameters:'
            print '\ncurrent_time:', current_time, '\ni_count:', i_count, '\ni_target:', i_target, '\np_s:', p_s, '\np_c:', p_c,\
                '\nC:', C, '\ni_eigen:', i_eigen, '\nsigma:', sigma, '\nx_recent:', x_recent, '\nfitness_recent:', fitness_recent,\
                '\ni_reset:', i_reset, '\ncurrent_sigma0:', current_sigma0, '\nx_learnt:', x_learnt, '\nindices_learnt:',\
                indices_learnt, '\nB_D:', B_D, '\nx_mean:', x_mean

        if debug:
            record_z.write(str(i_count)+'   '+str(z)+'\n')
            record_x.write(str(i_count)+'   '+str(x)+'\n')
            record_confidences.write(str(i_count)+'   '+str(confidences)+'\n') 
            record_scalars.write(str(i_count)+'   '+str(i_target)+'   '+str(fitness_mean)+'   '+str(h_sig)+'   '+str(sigma)+'   '+\
                str(cond)+'   '+str(N_resampled_trial)+'\n')
            record_fitness.write(str(i_count)+'   '+str(fitness)+'\n')
            record_params_abs.write(str(i_count)+'   '+str(params_abs)+'\n')
            record_x_mean.write(str(i_count)+'   '+str(x_mean)+'\n')
            record_params_r.write(str(i_count)+'   '+str(params_r)+'\n')
            record_p_s.write(str(i_count)+'   '+str(p_s)+'\n')
            record_p_c.write(str(i_count)+'   '+str(p_c)+'\n')
            record_C.write(str(i_count)+'   '+str(C)+'\n')
            record_D.write(str(i_count)+'   '+str(D)+'\n')
            record_B_D.write(str(i_count)+'   '+str(B_D)+'\n')
            record_x_recent.write(str(i_count)+'   '+str(x_recent)+'\n')
            record_fitness_recent.write(str(i_count)+'   '+str(fitness_recent)+'\n')
            record_ptp_fitness_recent.write(str(i_count)+'   '+str(np.ptp(fitness_recent, axis=0))+'\n')
            record_ptp_x_recent.write(str(i_count)+'   '+str(np.ptp(x_recent, axis=0))+'\n')

            for record in records:
                record.flush()

    return_dict['time'] = datetime.datetime.now()-t_0
    return_dict['steps'] = i_count    

    x_min = x_mean
    if verbose:
        print 'x:', x
        print 'x_mean:', x_mean

    create_speaker_finish(speaker, motor_memory, outputfolder)

    return return_dict







def main(n_trials):
  global rank, load_state, i_start, n_workers, resample, verbose, folder, speaker, n_vowels, normalize, results_write, N

  if rank==0:
      results_write.write('a:steps   a:time   a:reward   i:steps   i:time   i:reward   u:steps   u:time   u:reward   steps   time\n\n')
      results_write.flush()


  for i_trial in xrange(n_trials):
    if rank==0:
        x_min = cmaes()

    else:
        if not load_state == None:
            i_start = comm.recv(source=0)
        i = int(i_start/(n_workers-1))
        while True:
            print 'current tag (slave):', i

            x_mean,sigma,B_D,i_count,i_target = comm.recv(source=0, tag=i)
            if x_mean == None:
                break

            invalid = True
            if rank == 1:
                print 'sampling parameters...'
            if resample:
              N_resampled = -1
              while invalid:
                N_resampled += 1
                z = np.random.randn(N)      # standard normally distributed vector
                x = x_mean + sigma*(np.dot(B_D, z))  # add mutation, Eq. 37
                invalid = (x < 0.0).any() or (x > 1.0).any()
              boundary_penalty = 0.0
            else:
              N_resampled = 0
              z = np.random.randn(N)          # standard normally distributed vector
              x = x_mean + sigma*(np.dot(B_D, z)) # add mutation, Eq. 37
              boundary_penalty = 0.0
              if (x<0.0).any() or (x>1.0).any(): # check boundary condition
                if verbose:
                    print 'boundary violated. repairing and penalizing.'
                x_repaired = x.copy()       # repair sample
                for i_component in xrange(len(x)):
                    if x[i_component] > 1.0:
                        x_repaired[i_component] = 1.0
                    elif x[i_component] < 0.0:
                        x_repaired[i_component] = 0.0
                boundary_penalty = np.linalg.norm(x-x_repaired)**2
                                            # penalize boundary violation, Eq. 51
                x = x_repaired
     
            params_full = get_full_parameters(x, i_target)
            energy_cost = get_energy_cost(params_full)   
            params_abs = get_abs_coord(params_full)


            confidences = evaluate_environment(params_abs, i_count, simulation_name=folder, outputfolder=outputfolder,
                i_target=i_target, rank=rank, speaker=speaker, n_vow=n_vowels, normalize=normalize)

            if rank==1 and verbose:
                print 'z:', z, ', x:', x, ', confidences:', confidences, ', energy costs:', energy_cost, ', boundary penalties:',\
                    boundary_penalty

            send_back = z,x,confidences,energy_cost,boundary_penalty,N_resampled
            comm.send(send_back, dest=0, tag=i)
            i += 1



outputfolder = get_output_folder(folder)
i_start = 0

if rank==0:
    outputfile = outputfolder+'out.dat'
    results = outputfolder+'results.dat'
    os.system('rm '+outputfile)   
    os.system('rm '+results)
    os.system('touch '+outputfile)
    os.system('touch '+results)
    output_write = open(outputfile, 'w')
    results_write = open(results, 'w')

    if debug:
        os.system('mkdir '+outputfolder+'debug')
        record_scalars_file = outputfolder+'debug/scalars.txt'
        record_B_D_file = outputfolder+'debug/B_D.txt'
        record_z_file = outputfolder+'debug/z.txt'
        record_x_file = outputfolder+'debug/x.txt'
        record_confidences_file = outputfolder+'debug/confidences.txt'
        record_fitness_file = outputfolder+'debug/fitness.txt'
        record_params_abs_file = outputfolder+'debug/params_abs.txt'
        record_x_mean_file = outputfolder+'debug/x_mean.txt'
        record_params_r_file = outputfolder+'debug/params_r.txt'
        record_p_c_file = outputfolder+'debug/p_c.txt'
        record_p_s_file = outputfolder+'debug/p_s.txt'
        record_C_file = outputfolder+'debug/C.txt'
        record_D_file = outputfolder+'debug/D.txt'
        record_fitness_recent_file = outputfolder+'debug/fitness_recent.txt'
        record_ptp_fitness_recent_file = outputfolder+'debug/ptp_fitness_recent.txt'
        record_x_recent_file = outputfolder+'debug/x_recent.txt'
        record_ptp_x_recent_file = outputfolder+'debug/ptp_x_recent.txt'
        record_files = [record_scalars_file, record_B_D_file, record_z_file, record_x_file, record_confidences_file,\
            record_fitness_file, record_params_abs_file, record_x_mean_file, record_params_r_file, record_p_c_file,\
            record_p_s_file, record_C_file, record_D_file, record_fitness_recent_file, record_ptp_fitness_recent_file,\
            record_x_recent_file, record_ptp_x_recent_file]

        for file_item in record_files:
            os.system('rm '+file_item)
            os.system('touch '+file_item)

        record_scalars = open(record_scalars_file, 'w')
        record_B_D = open(record_B_D_file, 'w')
        record_z = open(record_z_file, 'w')
        record_x = open(record_x_file, 'w')
        record_confidences = open(record_confidences_file, 'w')
        record_fitness = open(record_fitness_file, 'w')
        record_params_abs = open(record_params_abs_file, 'w')
        record_x_mean = open(record_x_mean_file, 'w')
        record_params_r = open(record_params_r_file, 'w')
        record_p_c = open(record_p_c_file, 'w')
        record_p_s = open(record_p_s_file, 'w')
        record_C = open(record_C_file, 'w')
        record_D = open(record_D_file, 'w')
        record_fitness_recent = open(record_fitness_recent_file, 'w')
        record_ptp_fitness_recent = open(record_ptp_fitness_recent_file, 'w')
        record_x_recent = open(record_x_recent_file, 'w')
        record_ptp_x_recent = open(record_ptp_x_recent_file, 'w')
        records = [record_scalars, record_B_D, record_z, record_x, record_confidences, record_fitness, record_params_abs,\
            record_x_mean, record_params_r, record_p_c, record_p_s, record_C, record_D, record_fitness_recent,\
            record_ptp_fitness_recent, record_x_recent, record_ptp_x_recent]


main(n_trials)

if rank==0:
    output_write.close()
    results_write.close()
    for record in records:
        record.close()
