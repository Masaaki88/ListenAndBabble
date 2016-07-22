import numpy as np
from numpy.linalg import norm
from numpy import exp
import sys
sys.path.append('/home/murakami/lib/python2.7/site-packages/')
from brian import *
from brian.hears import *
from scipy.signal import resample
from VTL_API.parWav_fun import parToWave
import cPickle
from matplotlib import pylab
import gzip

infant = True

'''
params_schwa = np.array([1.0, -2.643, -2.0, -0.07, 0.524, 0.0, -0.426, -0.767, 2.036, -0.578, 1.163, 0.321, -1.853, -1.7267, 0.0, 0.046, 0.116, 0.116])
params_i = np.array([0.8580, -2.9237, -1.8808, -0.0321, 0.5695, 0.1438, 0.6562, -0.3901, 2.6431, -0.6510, 2.1213, 0.3124, -1.2385, -0.5088, 0.3674, 0.034, -0.1274, -0.2887])
params_u = np.array([0.9073, -3.2279, -4.0217, 1.0, 0.3882, 0.5847, 0.3150, -0.5707, 2.0209, -1.0122, 1.8202, -0.1492, -1.2155, -0.8928, 0.5620, 0.1637, 0.0602, -0.0386])
params_a = np.array([0.3296, -2.3640, -4.3032, 0.0994, 0.8196, 1.0, -0.4878, -1.2129, 1.9036, -1.5744, 1.3212, -1.0896, -2.4673, -1.7963, 1.0313, -0.1359, 0.4925, 0.0772])
'''

params_i = np.array([0.8580, -2.9237, -1.8808, -0.0321, 0.5695, 0.1438, 0.6562, -0.3901, 2.6431, -0.6510, 2.1213, 0.3124, 0.3674, 0.034, -0.1274, -0.2887])
params_u = np.array([0.9073, -3.2279, -4.0217, 1.0, 0.3882, 0.5847, 0.3150, -0.5707, 2.0209, -1.0122, 1.8202, -0.1492, 0.5620, 0.1637, 0.0602, -0.0386])
params_a = np.array([0.3296, -2.3640, -4.3032, 0.0994, 0.8196, 1.0, -0.4878, -1.2129, 1.9036, -1.5744, 1.3212, -1.0896, 1.0313, -0.1359, 0.4925, 0.0772])
params_schwa = np.array([1.0, -2.643, -2.0, -0.07, 0.524, 0.0, -0.426, -0.767, 2.036, -0.578, 1.163, 0.321, 0.0, 0.046, 0.116, 0.116])

params_ad_e = np.array([0.44, -4.0949, -2.6336, -0.1571, 0.7513, 0.4073, 2.5611, -0.836, 4.6531, -0.954, 4.0, 0.3971, -0.2859, -0.986, 1.288, 0.364, 0.084, 0.05])
params_ad_a = np.array([0.3296, -4.2577, -4.3032, 0.0994, 0.8095, 1.0, -0.1154, -2.0006, 4.2957, -1.3323, 3.0737, -0.3857, -2.8066, -2.9354, 1.336, -0.176, 0.638, 0.1])
params_ad_i = np.array([0.858, -5.2436, -1.8808, -0.091, 0.6751, 0.1438, 2.5295, -0.5805, 4.6333, -0.8665, 3.9, 0.646, -0.17, -0.7805, 1.316, 0.044, -0.165, -0.374])
params_ad_o = np.array([0.5314, -5.1526, -6.2288, 0.7435, 0.1846, 0.503, -0.1834, -1.0274, 2.4269, -1.1931, 2.0194, 0.1551, -1.3385, -2.6564, 0.556, -0.392, 0.0, 0.05])
params_ad_u = np.array([1.0, -5.6308, -4.0217, 1.0, 0.2233, 0.5847, 0.6343, -0.9421, 2.7891, -0.694, 2.4431, 0.3572, -1.3615, -2.8154, 1.4, 0.212, 0.078, -0.05])
params_schwa_ad = np.array([0.6259, -4.8156, -4.6286, 0.1203, 0.6552, 0.5709, 1.3037, -1.9642, 4.8069, -1.019, 3.3425, -0.3516, -1.7212, -1.9642, 0.752, 0.696, 0.9, 0.256])

proto_a = Sound('infant_a.wav')
proto_i = Sound('infant_i.wav')
proto_u = Sound('infant_u.wav')

flow1000 = cPickle.load(open('3vow_N1000.flow', 'r'))
flow1000_biased = cPickle.load(open('3vow_N1000_biased.flow', 'r'))
flow1000_reg = cPickle.load(open('3vow_N1000_reg.flow', 'r'))




def get_abs_coord(x):
    global infant

    if infant:
#        low_boundaries = np.array([0.0, -3.228, -7.0, -1.0, -1.102, 0.0, -3.194, -1.574, 0.873, -1.574, -3.194, -1.574, -4.259, -3.228, -1.081, -1.081, -1.081, -1.081])
#        high_boundaries = np.array([1.0, -1.85, 0.0, 1.0, 2.205, 1.0, 2.327, 0.63, 3.2, 1.457, 2.327, 2.835, 1.163, 0.079, 1.081, 1.081, 1.081, 1.081])
        low_boundaries = np.array([0.0, -3.228, -7.0, -1.0, -1.102, 0.0, -3.194, -1.574, 0.873, -1.574, -3.194, -1.574, -1.081, -1.081, -1.081, -1.081])
        high_boundaries = np.array([1.0, -1.85, 0.0, 1.0, 2.205, 1.0, 2.327, 0.63, 3.2, 1.457, 2.327, 2.835, 1.081, 1.081, 1.081, 1.081])
    else:
        low_boundaries = np.array([0.0, -6.0, -7.0, -1.0, -2.0, 0.0, -3.0, -3.0, 1.5, -3.0, -3.0, -3.0, -4.0, -6.0, -1.4, -1.4, -1.4, -1.4])
        high_boundaries = np.array([1.0, -3.5, 0.0, 1.0, 4.0, 1.0, 4.0, 1.0, 5.5, 2.5, 4.0, 5.0, 2.0, 0.0, 1.4, 1.4, 1.4, 1.4])

#    abs_coord = np.zeros(18)
#    for i in xrange(18):
    abs_coord = np.zeros(16)
    for i in xrange(16):
        abs_coord[i] = low_boundaries[i] + x[i] * (high_boundaries[i] - low_boundaries[i])

    return abs_coord


def get_rel_coord(x):
    global infant

    if infant:
#        low_boundaries = np.array([0.0, -3.228, -7.0, -1.0, -1.102, 0.0, -3.194, -1.574, 0.873, -1.574, -3.194, -1.574, -4.259, -3.228, -1.081, -1.081, -1.081, -1.081])
#        high_boundaries = np.array([1.0, -1.85, 0.0, 1.0, 2.205, 1.0, 2.327, 0.63, 3.2, 1.457, 2.327, 2.835, 1.163, 0.079, 1.081, 1.081, 1.081, 1.081])
        low_boundaries = np.array([0.0, -3.228, -7.0, -1.0, -1.102, 0.0, -3.194, -1.574, 0.873, -1.574, -3.194, -1.574, -1.081, -1.081, -1.081, -1.081])
        high_boundaries = np.array([1.0, -1.85, 0.0, 1.0, 2.205, 1.0, 2.327, 0.63, 3.2, 1.457, 2.327, 2.835, 1.081, 1.081, 1.081, 1.081])
    else:
        low_boundaries = np.array([0.0, -6.0, -7.0, -1.0, -2.0, 0.0, -3.0, -3.0, 1.5, -3.0, -3.0, -3.0, -4.0, -6.0, -1.4, -1.4, -1.4, -1.4])
        high_boundaries = np.array([1.0, -3.5, 0.0, 1.0, 4.0, 1.0, 4.0, 1.0, 5.5, 2.5, 4.0, 5.0, 2.0, 0.0, 1.4, 1.4, 1.4, 1.4])

#    rel_coord = np.zeros(18)
#    for i in xrange(18):
    rel_coord = np.zeros(16)
    for i in xrange(16):
        rel_coord[i] = (x[i] - low_boundaries[i]) / (high_boundaries[i] - low_boundaries[i])

    return rel_coord


def get_energy_cost(params_r):
    global params_schwa, params_schwa_ad, infant

    if infant:
        params_neutral = params_schwa
    else:
        params_neutral = params_schwa_ad

    deviation = norm(params_r - get_rel_coord(params_neutral))

    return deviation



def get_full_parameters(x, i_target, parameters):
    global params_i, params_u, params_a, params_ad_e, params_ad_a, params_ad_i, params_ad_o, params_ad_u, infant

    if infant:
        params_targets = [params_a, params_u, params_i]
    else:
        params_targets = [params_ad_a, params_ad_u, params_ad_i, params_ad_e, params_ad_o]
        
    params = get_rel_coord(params_targets[i_target])

    parameters_indices = get_parameters_indices(parameters)

    for i in xrange(len(x)):
        params[parameters_indices[i]] = x[i]

    return params


def get_parameters_indices(parameters):
#    par_table = {'HX':0, 'HY':1, 'JA':2, 'LP':3, 'LD':4, 'VS':5, 'TCX':6, 'TCY':7, 'TTX':8, 'TTY':9, 'TBX':10, 'TBY':11, 'TRX':12, 'TRY':13, 'TS1':14, 'TS2':15, 'TS3':16, 'TS4':17}
    par_table = {'HX':0, 'HY':1, 'JA':2, 'LP':3, 'LD':4, 'VS':5, 'TCX':6, 'TCY':7, 'TTX':8, 'TTY':9, 'TBX':10, 'TBY':11, 'TS1':12, 'TS2':13, 'TS3':14, 'TS4':15}

    n_dim = len(parameters)
    parameters_indices = np.zeros(n_dim, dtype='int')
    for i in xrange(n_dim):
        parameters_indices[i] = par_table[parameters[i]]

    return parameters_indices




def get_abs_coord_flag(parameters, target, flag):
#    key_table = {0:'HX', 1:'HY', 2:'JA', 3:'LP', 4:'LD', 5:'VS', 6:'TCX', 7:'TCY', 8:'TTX', 9:'TTY', 10:'TBX', 11:'TBY', 12:'TRX', 13:'TRY', 14:'TS1', 15:'TS2', 16:'TS3', 17:'TS4'}
    key_table = {0:'HX', 1:'HY', 2:'JA', 3:'LP', 4:'LD', 5:'VS', 6:'TCX', 7:'TCY', 8:'TTX', 9:'TTY', 10:'TBX', 11:'TBY', 12:'TS1', 13:'TS2', 14:'TS3', 15:'TS4'}
    if flag == 'TC':
        parameters_indices = [6,7]            
    elif flag == 'visual':
#        parameters_indices = [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        parameters_indices = [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    elif flag == 'all':
#        parameters_indices = range(18)
        parameters_indices = range(16)
    else:
        print 'Error: flag not recognized!'

    if target == 'a':
        rel_params = get_rel_coord(params_a.copy())
    elif target == 'i':
        rel_params = get_rel_coord(params_i.copy())
    elif target == 'u':
        rel_params = get_rel_coord(params_u.copy())
    else:
        print 'Error: target not recognized!'

    if len(parameters) != len(parameters_indices):
        print "Error: parameters and flag don't match!"


    L = len(parameters)
    for i in xrange(L):
        rel_params[parameters_indices[i]] = parameters[i]
    abs_coord = get_abs_coord(rel_params)

    return_params = []
#    for i in xrange(18):
    for i in xrange(16):
#        if i in parameters_indices:
            return_params.append([key_table[i], round(abs_coord[i],3)])

    return return_params





def get_motor_deviation_flag(parameters, target, flag='all'):
#    key_table = {0:'HX', 1:'HY', 2:'JA', 3:'LP', 4:'LD', 5:'VS', 6:'TCX', 7:'TCY', 8:'TTX', 9:'TTY', 10:'TBX', 11:'TBY', 12:'TRX', 13:'TRY', 14:'TS1', 15:'TS2', 16:'TS3', 17:'TS4'}
    key_table = {0:'HX', 1:'HY', 2:'JA', 3:'LP', 4:'LD', 5:'VS', 6:'TCX', 7:'TCY', 8:'TTX', 9:'TTY', 10:'TBX', 11:'TBY', 12:'TS1', 13:'TS2', 14:'TS3', 15:'TS4'}
    if flag == 'TC':
        parameters_indices = [6,7]            
    elif flag == 'visual':
#        parameters_indices = [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        parameters_indices = [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    elif flag == 'all':
#        parameters_indices = range(18)
        parameters_indices = range(16)
    else:
        print 'Error: flag not recognized!'

    if target == 'a':
        rel_params = get_rel_coord(params_a.copy())
    elif target == 'i':
        rel_params = get_rel_coord(params_i.copy())
    elif target == 'u':
        rel_params = get_rel_coord(params_u.copy())
    else:
        print 'Error: target not recognized!'

    if len(parameters) != len(parameters_indices):
        if len(parameters) == 16:
            parameters2 = []
            for index in parameters_indices:
                parameters2.append(parameters[index])
            parameters = parameters2
        else:
            print "Error: parameters and flag don't match!"
    
    target_params = []
    for i in parameters_indices:
        target_params.append(rel_params[i])

    parameters = np.array(parameters)
    target_params = np.array(target_params)

    motor_deviation = norm(target_params-parameters)

    return motor_deviation





def get_confidences(mean_sample_vote):

    n_classes = len(mean_sample_vote)
    confidences = np.zeros(n_classes)
    norm_sum = 0.0
    for i in xrange(n_classes):
        confidence_i = exp(mean_sample_vote[i])
        confidences[i] = confidence_i
        norm_sum += confidence_i
    confidences /= norm_sum

    return confidences


def drnl(sound, n_channels=50, compressed=True):
  """ use predefined cochlear model, see Lopez-Poveda et al 2001"""
  cf = erbspace(100*Hz, 8000*Hz, n_channels)    # centre frequencies of ERB scale
                                        #  (equivalent rectangular bandwidth)
                                        #  between 100 and 8000 Hz
  drnl_filter = DRNL(sound, cf, type='human')
                                        # use DNRL model, see documentation
  print 'processing sound'
  out = drnl_filter.process()           # get array of channel activations
  if compressed:
      out = out.clip(0.0)               # -> fast oscillations can't be downsampled otherwise
      out = resample(out, int(round(sound.nsamples/1000.0)))
                                        # downsample sound for memory reasons
  return out


def normalize_activity(x):

    x_normalized = x.copy()
    minimum = x.min()
    maximum = x.max()
    range_ = maximum - minimum
    bias = abs(maximum) - abs(minimum)
    
    x_normalized -= bias/2.0
    x_normalized /= range_/2.0
    
    return x_normalized


def sound_to_confidences(sound, flow, normalize=False):
    drnl_out = drnl(sound)
    flow_out = flow(drnl_out)
    if normalize:
        flow_out = normalize_activity(flow_out)
    mean_activations = np.mean(flow_out, axis=0)
    confidences = get_confidences(mean_activations)

    return confidences




def correct_initial(sound):
  """ function for removing initial burst from sounds"""

  low = 249                             # duration of initial silence
  for i in xrange(low):                 # loop over time steps during initial period
    sound[i] = 0.0                      # silent time step

  return sound


#*********************************************************


def get_resampled(sound):		
  """ function for adapting sampling frequency to AN model
	    VTL samples with 22kHz, AN model requires 50kHz"""

  target_nsamples = int(50*kHz * sound.duration)
                                        # calculate number of samples for resampling
                                        # (AN model requires 50kHz sampling rate)
  resampled = resample(sound, target_nsamples)
                                        # resample sound to 50kHz
  sound_resampled = Sound(resampled, samplerate = 50*kHz)
                                        # declare new sound object with new sampling rate
  return sound_resampled


#*********************************************************


def get_extended(sound):		
  """ function for adding silent period to shortened vowel
	     ESN requires all samples to have the same dimensions"""

  target_nsamples = 36358               # duration of longest sample
  resized = sound.resized(target_nsamples)
                                        # resize samples
  return resized




def relParToWave(x_rel):
    global infant

    x_abs = get_abs_coord(x_rel)

    if infant:
        wavFile = parToWave(x_abs, speaker='infant', simulation_name='infant', pitch_var=0.0, len_var=1.0, verbose=False, rank=1, different_folder='aux_infant.wav', monotone=False)
    else:
        wavFile = parToWave(x_abs, speaker='adult', simulation_name='adult', pitch_var=0.0, len_var=1.0, verbose=False, rank=1, different_folder='aux_adult.wav', monotone=False)

    sound = Sound(wavFile)
    sound = correct_initial(sound)      # call correct_initial to remove initial burst
    sound_resampled = get_resampled(sound)
                                        # call get_resampled to adapt generated sound to AN model
    sound_extended = get_extended(sound_resampled)
                                        # call get_extended to equalize duration of all sounds
    sound_extended.save(wavFile)

    return sound_extended





[N, verbose, params_r, i_stop, parameters_indices, sigma0, params_r_target, conf_threshold, intrinsic_motivation, n_vowels, energy_factor, alpha, N_reservoir, no_convergence, outputfolder, random_restart, indices_learnt, i_count, i_target, p_s , p_c, C, i_eigen, sigma, x_recent, fitness_recent, i_reset, current_sigma0, x_learnt, indices_learnt, current_time, cond_stop] = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]




def load_saved_state(load_state):
    global N, verbose, params_r, i_stop, parameters_indices, sigma0, params_r_target, conf_threshold, intrinsic_motivation, n_vowels, energy_factor, alpha, N_reservoir, no_convergence, outputfolder, random_restart, indices_learnt, i_count, i_target, p_s , p_c, C, i_eigen, sigma, x_recent, fitness_recent, i_reset, current_sigma0, x_learnt, indices_learnt, current_time, cond_stop

    inputfile_dynamic = open(load_state+'.dyn', 'r')
    inputfile_static = open(load_state+'.stat', 'r')
   
    static_params = cPickle.load(inputfile_static)
    dynamic_params = cPickle.load(inputfile_dynamic)

    inputfile_static.close()
    inputfile_dynamic.close()

    static_list = ['N', 'verbose', 'params_r', 'i_stop', 'parameters_indices', 'sigma0', 'params_r_target', 'conf_threshold', 'intrinsic_motivation', 'n_vowels', 'energy_factor', 'alpha', 'N_reservoir', 'no_convergence', 'outputfolder', 'random_restart', 'cond_stop']
    dynamic_list = ['current_time', 'i_count', 'i_target', 'p_s' , 'p_c', 'C', 'i_eigen', 'sigma', 'x_recent', 'fitness_recent', 'i_reset', 'current_sigma0', 'x_learnt', 'indices_learnt']

    [N, verbose, params_r, i_stop, parameters_indices, sigma0, params_r_target, conf_threshold, intrinsic_motivation, n_vowels, energy_factor, alpha, N_reservoir, no_convergence, outputfolder, random_restart, cond_stop] = static_params
    [current_time, i_count, i_target, p_s , p_c, C, i_eigen, sigma, x_recent, fitness_recent, i_reset, current_sigma0, x_learnt, indices_learnt] = dynamic_params

    print '\nstatic parameters:\n'
    for i in xrange(len(static_list)):
        print static_list[i], ':', static_params[i]
    print '\ndynamic parameters:\n'
    for i in xrange(len(dynamic_list)):
        print dynamic_list[i], ':', dynamic_params[i]

    return static_params, dynamic_params




def plot_inputs():
    
    figure() 
    inputfiles = ['canonical_adult/a_adult[0.0_1.0].wav', 'canonical_adult/i_adult[0.0_1.0].wav', 'canonical_adult/u_adult[0.0_1.0].wav', 'canonical_infant/a_infant[0.0_1.0].wav', 'canonical_infant/i_infant[0.0_1.0].wav', 'canonical_infant/u_infant[0.0_1.0].wav']
    labels = ['adult [a]', 'adult [i]', 'adult [u]', 'infant [a]', 'infant [i]', 'infant [u]']

    for i in xrange(len(inputfiles)):
        inputfile_name = 'VTL_API/output/'+inputfiles[i]
        inputsound = Sound(inputfile_name)
        data = drnl(inputsound, compressed=True)
        if i < 3:
            data = data[:len(data)-4]
        else:
            data = data[2:len(data)-2]

        splot = subplot(231+i)
        title(labels[i])
        xticks(range(0,35,5), np.arange(0.0,0.7,0.1))
        imshow(data.T, origin='lower', cmap=plt.cm.bwr, aspect=0.2, vmax=data.max(), vmin=-data.max(), interpolation='none')

        if i > 2:
            xlabel('Time (s)')
        if i%3 == 0:
            ylabel('Neurons')
    tight_layout()

    figure() 
    for i in xrange(len(inputfiles)):
        inputfile_name = 'VTL_API/output/'+inputfiles[i]
        inputsound = Sound(inputfile_name)
        data = drnl(inputsound, compressed=False)

        splot = subplot(231+i)
        title(labels[i])
        xticks(range(0,35000,5000), np.arange(0.0,0.7,0.1))
        imshow(data.T, origin='lower', cmap=plt.cm.bwr, aspect=200, interpolation='none')

        if i > 2:
            xlabel('Time (s)')
        if i%3 == 0:
            ylabel('Neurons')


    tight_layout()
    show()



def plot_single_input(inputfile, label):
    
    figure() 

    inputfile_name = 'VTL_API/output/'+inputfile
    inputsound = Sound(inputfile_name)
    data = drnl(inputsound, compressed=True)
    data = data[:len(data)-4]

    title(label)
    xticks(range(0,35,5), np.arange(0.0,0.7,0.1))
#        ylabel('Neurons')
#        print data
    imshow(data.T, origin='lower', cmap=plt.cm.bwr, aspect=0.2, vmax=data.max(), vmin=-data.max(), interpolation='none')

    xlabel('Time (s)')
    ylabel('Neurons')

    tight_layout()
    show()



def plot_spectrograms(flag='default'):
    
    figure() 

    if flag == 'default':    
        inputfiles = ['infant_@_infant_1.wav', 'canonical_infant/a_infant[0.0_1.0].wav', 'canonical_infant/i_infant[0.0_1.0].wav', 'canonical_infant/u_infant[0.0_1.0].wav']
        labels = ['infant [@]', 'infant [a]', 'infant [i]', 'infant [u]']
    elif flag == 'TC':
        inputfiles = ['TC_a.wav', 'VTL_API/output/canonical_infant/a_infant[0.0_1.0].wav', 'TC_i.wav', 'VTL_API/output/canonical_infant/i_infant[0.0_1.0].wav', 'TC_u.wav', 'VTL_API/output/canonical_infant/u_infant[0.0_1.0].wav']
        labels = ['learnt [a]', 'mentor [a]', 'learnt [i]', 'mentor [i]', 'learnt [u]', 'mentor [u]']
    elif flag == 'visual':
        inputfiles = ['visual_a.wav', 'VTL_API/output/canonical_infant/a_infant[0.0_1.0].wav', 'visual_i.wav', 'VTL_API/output/canonical_infant/i_infant[0.0_1.0].wav', 'visual_u.wav', 'VTL_API/output/canonical_infant/u_infant[0.0_1.0].wav']
        labels = ['learnt [a]', 'mentor [a]', 'learnt [i]', 'mentor [i]', 'learnt [u]', 'mentor [u]']
    elif flag == 'full':
        inputfiles = ['full_a.wav', 'VTL_API/output/canonical_infant/a_infant[0.0_1.0].wav', 'full_i.wav', 'VTL_API/output/canonical_infant/i_infant[0.0_1.0].wav']
        labels = ['learnt [a]', 'mentor [a]', 'learnt [i]', 'mentor [i]']
    elif flag == 'ambient':
        inputfiles = ['adult_a.wav', 'adult_i.wav', 'adult_u.wav', 'adult_0.wav', 'infant_a.wav', 'infant_i.wav', 'infant_u.wav', 'infant_0.wav']
        labels = None

    ylabel('Frequency')

    for i in xrange(len(inputfiles)):
#        inputfile_name = 'VTL_API/output/'+inputfiles[i]
        inputfile_name = inputfiles[i]
        inputsound = Sound(inputfile_name)
#        if i > 0:
        data = inputsound[2850:len(inputsound)-2500]
#        else:
#            data = inputsound[1200:len(inputsound)]
#        length = len(inputsound)-2500-2850
#        print length

        if flag == 'default':
            splot = subplot(221+i)
            title(labels[i])
        elif flag in ['TC', 'visual', 'full']:
            splot = subplot(321+i)
            title(labels[i])
        elif flag == 'ambient':
            if i < 4:
                splot = subplot(421+i*2)
            else:
                splot = subplot(421+i*2-7)

        if flag == 'ambient':
            data.spectrogram(low=0*Hz, high=8000*Hz)
        else:
            data.spectrogram(low=100*Hz, high=8000*Hz)
            yticks(range(0,8001,2000), range(0,9,2))

        if flag == 'ambient':
            if (i+1)%4 == 0:
                xlabel('Time (s)')
            else:
#                xticks(np.arange(0.0,0.61,0.1), ['','','','','',''])
                tick_params(axis='x', labelbottom='off')
                xlabel('')
            if i < 4:
                yticks(range(0,8001,4000), range(0,9,4))
#                if i%2 == 0:
#                    ylabel('(kHz)')
#                else:
#                    ylabel('Frequency')
                if i==1:
                    ylabel('(kHz)')
                elif i==2:
                    ylabel('Frequency')
                else:
                    ylabel('')
            else:
                yticks(range(0,8001,4000), ['','',''])
                ylabel('')
        else:
            if i>4:
                xlabel('Time (s)')
            else:
                xlabel('')
            if i%2 == 0:
                ylabel('Frequency (kHz)')
            else:
               ylabel('')
    '''
        if i > 1:
            xlabel('Time (s)')
        if i%2 == 0:
            ylabel('Frequency (kHz)')
    '''


    tight_layout()
    show()






def plot_single_spectrogram(pathname):
    
    figure() 

    inputsound = Sound(pathname)
    data = inputsound[2850:len(inputsound)-2500]
    data.spectrogram(low=100*Hz, high=8000*Hz)
    yticks(range(0,8001,2000), range(0,9,2))
    xlabel('Time (s)')
    ylabel('Frequency (kHz)')
#    tight_layout()
    show()







def plot_conf_matrices():
#def plot_conf_(conf, outputfile_, N):
    """ function to visualise a balanced confusion matrix
         modified version of the predefined Oger.utils.plot_conf function
         additional arguments: outputfile_ for plot files, N for reservoir size"""

#    outputfile_plot = outputfile_+'_'+str(N)+'.png'
#    np.asarray(conf).dump(outputfile_plot+'.np')

    filelist = ['conf1.np', 'conf10.np', 'conf100.np', 'conf1000.np']

    figure()

    for i in xrange(len(filelist)):
        subplot(221+i)

        inputfile = open(filelist[i], 'r')
        conf = np.load(inputfile)
        res = pylab.imshow(np.asarray(conf), vmin=0.0, vmax=1.0, cmap=pylab.cm.jet, interpolation='nearest')
 
#        for i, err in enumerate(conf.correct):
        for j in xrange(4):
            err = conf[j][j]
                                        # display correct detection percentages 
                                        # (only makes sense for CMs that are normalised per class (each row sums to 1))
            err_percent = "%d%%" % round(err * 100)
            pylab.text(j-.3, j+.1, err_percent, fontsize=12)
#            pylab.text(j-.2, j+.1, err_percent, fontsize=12)
  
        if i > 1:
            xticks(np.arange(0.0,3.1,1.0), ['/a/','/i/','/u/','null'])
            xlabel('classified as')
        else:
            xticks(np.arange(0.0,3.1,1.0), ['','','',''])
        if i % 2==0:
            yticks(np.arange(0.0,3.1,1.0), ['/a/','/i/','/u/','null'])
            ylabel('test sample')
        else:
            yticks(np.arange(0.0,3.1,1.0), ['','','',''])
    

        inputfile.close()

#        cb = pylab.colorbar(res)


    subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = axes([0.85, 0.1, 0.03, 0.8])
    colorbar(cax=cax)





#    pylab.savefig(outputfile_plot)
    show()
                                       # save plot
#    res = None





def plot_target_ranges(par_index=-1, separate=False):
    global params_schwa, params_a, params_i, params_u
    
    inputfile = open('target_ranges.dat', 'r')
    inputdata = np.load(inputfile)
    inputfile.close()

    par_table = {0:'HX', 1:'HY', 2:'JA', 3:'LP', 4:'LD', 5:'VS', 6:'TCX', 7:'TCY', 8:'TTX', 9:'TTY', 10:'TBX', 11:'TBY', 12:'TS1', 13:'TS2', 14:'TS3', 15:'TS4'}
    params_schwa_rel = get_rel_coord(params_schwa)
    params_a_rel = get_rel_coord(params_a)
    params_i_rel = get_rel_coord(params_i)
    params_u_rel = get_rel_coord(params_u)
    params_rel = [params_a_rel, params_i_rel, params_u_rel]

    x = np.arange(0.0,1.01,0.01)
    y = np.ones(101)
    y *= 0.5


    if par_index==-1:
      for i_param in xrange(len(inputdata)):

        if not separate:
            mod = i_param % 4
            if mod == 0:
                figure()
            subplot(411+mod)

        else:
            figure()

        plot(y, 'k--')
        for i_vow in xrange(3):
            if i_vow == 0:
                plot(inputdata[i_param][0], 'r-')
            elif i_vow == 1:
                plot(inputdata[i_param][1], 'g-')
            elif i_vow == 2:
                plot(inputdata[i_param][2], 'b-')
            xticks(np.arange(0.0,101.0,10.0),np.arange(0.0,1.1,0.1))
            xlabel(par_table[i_param])
            ylim(0.0,0.7)
            yticks(np.arange(0.1,0.6,0.2))
            ylabel('reward')

#            vlines(params_schwa_rel[i_param]*100, 0.0, 0.5, linestyles=u'dotted')
#            print 'i_param:', i_param
#            print 'params_a_rel[i_param]:', params_a_rel[i_param]
            if params_a_rel[i_param] < 0.1:
#                print i_param
                vlines((params_a_rel[i_param]*100)+1, 0.0, 0.5, colors=u'r', linestyles=u'dotted', linewidth=2)
            elif params_a_rel[i_param] > 0.9:
                vlines((params_a_rel[i_param]*100)-1, 0.0, 0.5, colors=u'r', linestyles=u'dotted', linewidth=2)
            else:
                vlines(params_a_rel[i_param]*100, 0.0, 0.5, colors=u'r', linestyles=u'dotted', linewidth=2)
            if params_i_rel[i_param] < 0.1:
                vlines((params_i_rel[i_param]*100)+1, 0.0, 0.5, colors=u'g', linestyles=u'dotted', linewidth=2)
            elif params_i_rel[i_param] > 0.9:
                vlines((params_i_rel[i_param]*100)-1, 0.0, 0.5, colors=u'g', linestyles=u'dotted', linewidth=2)
            else:
                vlines(params_i_rel[i_param]*100, 0.0, 0.5, colors=u'g', linestyles=u'dotted', linewidth=2)
            if params_u_rel[i_param] < 0.1:
                vlines((params_u_rel[i_param]*100)+1, 0.0, 0.5, colors=u'b', linestyles=u'dotted', linewidth=2)
            elif params_u_rel[i_param] > 0.9:
                vlines((params_u_rel[i_param]*100)-1, 0.0, 0.5, colors=u'b', linestyles=u'dotted', linewidth=2)
            else:
                vlines(params_u_rel[i_param]*100, 0.0, 0.5, colors=u'b', linestyles=u'dotted', linewidth=2)
        if not separate:
            tight_layout()

    else:
        colors_style = ['r-', 'g-', 'b-']
        colors_vlines = [u'r', u'g', u'b']

        figure()

        for i_vow in xrange(3):
            subplot(311+i_vow)

            plot(y, 'k--')
            plot(inputdata[par_index][i_vow], colors_style[i_vow])
            xticks(np.arange(0.0,101.0,10.0),np.arange(0.0,1.1,0.1))
            ylim(0.0,0.7)
            yticks(np.arange(0.1,0.6,0.2))
            ylabel('reward')

            if params_rel[i_vow][par_index] < 0.1:
                vlines((params_rel[i_vow][par_index]*100)+1, 0.0, 0.5, colors=colors_vlines[i_vow], linestyles=u'dotted')
            elif params_rel[i_vow][par_index] > 0.9:
                vlines((params_rel[i_vow][par_index]*100)-1, 0.0, 0.5, colors=colors_vlines[i_vow], linestyles=u'dotted')
            else:
                vlines(params_rel[i_vow][par_index]*100, 0.0, 0.5, colors=colors_vlines[i_vow], linestyles=u'dotted')

            vlines(0.465*100, 0.0, 0.5, colors=u'Gray', linestyles=u'dotted')

        tight_layout()

    show()




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

#    global n_channels, n_vow, outputfile_, lib_syll, compressed, output, flow, first_plot, cb, cb2

    import Oger

    Oger.utils.make_inspectable(Oger.nodes.LeakyReservoirNode)
                                        # make reservoir states inspectable for plotting
    Oger.utils.make_inspectable(Oger.nodes.ReservoirNode) 

    n_channels = 50
    n_vow = 3
    lib_syll =  ['/a/','/i/','/u/']
    compressed = True
    flow = cPickle.load(open('3vow_N1000_reg.flow', 'r'))
    

    vowels_and_files = [['adult [a]', 'data/ad_a.dat.gz', 'ad_a'], ['infant [a]', 'data/in_a.dat.gz', 'in_a'], ['adult [i]', 'data/ad_i.dat.gz', 'ad_i'], ['infant [i]', 'data/in_i.dat.gz', 'in_i'], ['adult [u]', 'data/ad_u.dat.gz', 'ad_u'], ['infant [u]', 'data/in_u.dat.gz', 'in_u']]

    pylab.figure()
    for i in xrange(2*n_vow):             # loop over all syllables
#        outputfile_plot = outputfile_+'_'+vowels_and_files[i][2]+'_'+str(N)+'.png'

        n_subplots_x, n_subplots_y = 3, 2
                                        # arrange two plots in one column
        pylab.subplot(n_subplots_x, n_subplots_y, i+1)
                                        # upper plot

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

        inputf = gzip.open(inputfile, 'rb')
                                        # open current inputfile in gzip read mode
        current_data = np.load(inputf)  # load numpy array from current inputfile
        inputf.close()                  # close inputfile

        xtest = current_data
#        xtest = np.array(list(current_data[0]))
                                        # read activations from input array
        ytest = flow(xtest)             # get activations of output neurons of trained network

        current_flow = flow[0].inspect()[0].T
#        np.array([ytest.T, current_flow]).dump(outputfile_plot+'.np')

#        n_subplots_x, n_subplots_y = 2, 1
                                        # arrange two plots in one column
#        pylab.subplot(n_subplots_x, n_subplots_y, 1)
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
#        pylab.title("Class activations of "+vowels_and_files[i][0])
        if i%2 == 0:
            pylab.ylabel("Class", fontsize=15)
            pylab.yticks(range(n_vow+1), lib_syll[:n_vow]+['null'], fontsize=15)
        else:
            pylab.yticks(range(n_vow+1), ['','','',''])
        if i > 3:
            pylab.xlabel('Time (s)', fontsize=15)
            pylab.xticks(range(0, 35, 5), np.arange(0.0, 0.7, 0.1), fontsize=15)
        else:
            pylab.xticks(range(0, 35, 5), ['','','','','','',''])

        '''
        pylab.figure()
        cb = pylab.colorbar(class_activity)
        '''

#        cb.update_bruteforce(class_activity)


         # plot confusion matrix (balanced, each class is equally weighted)

        '''
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
        '''
#        cb2.update_bruteforce(reservoir_activity)

#        pylab.savefig(outputfile_plot)   # save figure
#        pylab.close('all')



    pylab.tight_layout()
    pylab.show()





# confidences of prototypes (flow N=1000, normalized, September 4):
#   a: 0.57 (unbiased), 0.58 (biased), 0.59 (regularized)
#   i: 0.60 (unbiased), 0.59 (biased), 0.60 (regularized)
#   u: 0.59 (unbiased), 0.59 (biased), 0.58 (regularized)
#
# motor deviations of prototypes (e=1)
#   a: 1.69 (TC), 4.32 (visual), 6.40 (all)
#   i: 0.93 (TC), 4.20 (visual), 4.97 (all)
#   u: 1.07 (TC), 4.13 (visual), 6.07 (all)
