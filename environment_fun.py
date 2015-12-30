import os
from brian import kHz, Hz, exp, isinf
from brian.hears import Sound, erbspace, loadsound, DRNL
from scipy.signal import resample
from VTL_API.parWav_fun import parToWave
#from VTL_API.par_to_wav import par_to_wav
import numpy as np
import matplotlib
matplotlib.use('Agg')                  # for use on clusters
import Oger
import pylab
from datetime import date
import cPickle





###########################################################
#
# Argument parsing
#
###########################################################


output = False                      # verbose mode
plots = True                        # plot mode
subfolder = None                    # initiate subfolder string
playback = False                   # only relevant if n_workers == 1




################################################
#
# Initialization
#
################################################



uncompressed = False
n_vow = 5                               # number of reservoir output units
lib_syll =  ['/a/','/i/','/u/','[o]','[e]','[E:]','[2]','[y]','[A]','[I]','[E]','[O]','[U]','[9]','[Y]','[@]','[@6]']
                                        # global syllable library


np.random.seed()                        # numpy random seed w.r.t. global runtime
Oger.utils.make_inspectable(Oger.nodes.LeakyReservoirNode)
                                        # make reservoir states inspectable for plotting



############### Announce simulation parameters

if False:
    print 'verbose mode:', output
    print 'playing back produced sound:', playback
    print 'using trained reservoir with', N, 'reservoir units and', n_vow, 'output units'
    print 'plotting classifier states:', plots



##########################################################
#
# Support functions:
#  - correct_initial(sound)
#  - get_resampled(sound)
#  - get_extended(sound)
#  - drnl(sound, n_channels)
#  - get_output_folder(subfolder)
#
##########################################################


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


#*********************************************************


def drnl(sound, n_channels=50):
  global uncompressed, output
  """ use predefined cochlear model, see Lopez-Poveda et al 2001"""

  cf = erbspace(100*Hz, 8000*Hz, n_channels)    # center frequencies of ERB scale
                                        #  (equivalent rectangular bandwidth)
                                        #  between 100 and 8000 Hz
  drnl_filter = DRNL(sound, cf, type='human')
                                        # use DNRL model, see documentation
  if output:
   print 'processing sound'
  out = drnl_filter.process()           # get array of channel activations
  if not uncompressed:
      out = out.clip(0.0)                    # -> fast oscillations can't be downsampled otherwise
                                        # downsample sound for memory reasons
      out = resample(out, int(round(sound.nsamples/1000.0)))
  return out


#*********************************************************


def get_output_folder(subfolder):
  """ create and return directory of current simulation"""

  today = date.today()                  # get system date
  today_string = today.isoformat()      # convert date to string
  outputpath_short = 'output/'+today_string
                                        # date yields super folder
  outputpath = 'output/'+today_string+'/'+subfolder+'/'
                                        # date + subfolder yields working folder
  return outputpath


#*********************************************************


def plot_reservoir_states(flow, y, i_target, folder, n_vow, rank):

    global lib_syll, subfolder
    """ plot reservoir states"""


    current_flow = flow[0].inspect()[0].T # reservoir activity for most recent item
    N = flow[0].output_dim              # reservoir size

    n_subplots_x, n_subplots_y = 2, 1   # arrange two plots in one column
    pylab.subplot(n_subplots_x, n_subplots_y, 1)
                                        # upper plot
    y_min = y.min()
    y_max = y.max()
    if abs(y_min) > y_max:              # this is for symmetrizing the color bar
            vmin = y_min                # -> 0 is always shown as white
            vmax = -y_min
    else:
            vmax = y_max
            vmin = -y_max

    class_activity = pylab.imshow(y.T, origin='lower', cmap=pylab.cm.bwr, aspect=10.0/(n_vow+1), interpolation='none', vmin=vmin, vmax=vmax)
                                        # plot output activations, adjust to get uniform aspect for all n_vow
    pylab.title("Class activations")
    pylab.ylabel("Class")
    pylab.xlabel('')
    pylab.yticks(range(n_vow+1), lib_syll[:n_vow]+['null'])
    pylab.xticks(range(0, 35, 5), np.arange(0.0, 0.7, 0.1))
    cb = pylab.colorbar(class_activity)

    n_subplots_x, n_subplots_y = 2, 1
    pylab.subplot(n_subplots_x, n_subplots_y, 2)
                                        # lower plot
    current_flow_min = current_flow.min()
    current_flow_max = current_flow.max()
    if abs(current_flow_min) > current_flow_max:
            vmin_c = current_flow_min   # symmetrizing color, see above
            vmax_c = -current_flow_min
    else:
            vmax_c = current_flow_max
            vmin_c = -current_flow_max

    reservoir_activity = pylab.imshow(current_flow, origin='lower', cmap=pylab.cm.bwr, aspect=10.0/N, interpolation='none', vmin=vmin_c, vmax=vmax_c)
                                        # plot reservoir states of current prototype,
                                        #  adjust to get uniform aspect for all N
    pylab.title("Reservoir states")
    pylab.xlabel('Time (s)')
    pylab.xticks(range(0, 35, 5), np.arange(0.0, 0.7, 0.1))
    pylab.ylabel("Neuron")
    pylab.yticks(range(0,N,N/7))
    cb2 = pylab.colorbar(reservoir_activity)

    pylab.savefig(folder+'data/vowel_'+str(i_target)+'_'+str(rank)+'.pdf')

    pylab.close('all')






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






def get_reward(mean_sample_vote, sound_extended, i_target, speaker, loudness_factor, softmax=False):


    if softmax:
        reward = exp(mean_sample_vote[i_target])
        other_exp = 0.0
        for i in xrange(len(mean_sample_vote)):
            if i!=i_target:
                other_exp += exp(mean_sample_vote[i])
        reward /= reward + other_exp

    else:
        reward = mean_sample_vote[i_target]
        for i in xrange(len(mean_sample_vote)):
            if i!=i_target:
                reward -= mean_sample_vote[i]

        if speaker == 'adult':
            target_loudness = [72.77, 65.20, 66.04, 68.37, 68.47]
        else:
            target_loudness = [73.78, 68.68, 69.78]

        level = float(sound_extended.level)
        if isinf(level):
            level = 0.0
        loudness_reward = level - target_loudness[i_target]

        if loudness_reward > 0.0:
            loudness_reward = 0.0

        reward += loudness_factor * loudness_reward



    return reward





def normalize_activity(x):

    x_normalized = x.copy()
    minimum = x.min()
    maximum = x.max()
    range_ = maximum - minimum
    bias = abs(maximum) - abs(minimum)

    x_normalized -= bias/2.0
    x_normalized /= range_/2.0

    return x_normalized




##########################################################
#
# Main script
#
##########################################################



def evaluate_environment(params, i_global, simulation_name, outputfolder, i_target=0, rank=1, speaker='adult', n_vow=5, normalize=False):

    folder = outputfolder

    ############### Sound generation

    if output:
     print 'simulating vocal tract'

    wavFile = parToWave(params, speaker, simulation_name, verbose=output, rank=rank) # call parToWave to generate sound file
#    wavFile = par_to_wav(params, speaker, simulation_name, verbose=output, rank=rank) # call parToWave to generate sound file
    if output:
     print 'wav file '+str(wavFile)+' produced'

    sound = loadsound(wavFile)          # load sound file for brian.hears processing
    if output:
     print 'sound loaded'



    ############### Audio processing

    sound = correct_initial(sound)      # call correct_initial to remove initial burst

    sound_resampled = get_resampled(sound)
                                        # call get_resampled to adapt generated sound to AN model
    sound_extended = get_extended(sound_resampled)
                                        # call get_extended to equalize duration of all sounds
    sound_extended.save(wavFile)        # save current sound as sound file

    os.system('cp '+wavFile+' '+folder+'data/vowel_'+str(i_target)+'_'+str(rank)+'.wav')

    if playback:
        print 'playing back...'
        sound_extended.play(sleep=True) # play back sound file

    if output:
     print 'sound acquired, preparing auditory processing'

    out = drnl(sound_extended)          # call drnl to get cochlear activation



    ############### Classifier evaluation

    flow_name = 'data/current_auditory_system.flow'
    flow_file = open(flow_name, 'r')    # open classifier file
    flow = cPickle.load(flow_file)      # load classifier
    flow_file.close()                   # close classifier file

    sample_vote_unnormalized = flow(out)                       # evaluate trained output units' responses for current item
    if normalize:
        sample_vote = normalize_activity(sample_vote_unnormalized)
    else:
        sample_vote = sample_vote_unnormalized
    mean_sample_vote = np.mean(sample_vote, axis=0)
                                        # average each output neurons' response over time


    confidences = get_confidences(mean_sample_vote)

    plot_reservoir_states(flow, sample_vote, i_target, folder, n_vow, rank)


    return confidences
