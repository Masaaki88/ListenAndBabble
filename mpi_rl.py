import os
import argparse

vowels = ['a', 'i', 'u', 'e', 'o']

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='store_true', help='increase verbosity?')
parser.add_argument('-f', '--folder', nargs='?', type=str, default='', help='subfolder to store results in')
parser.add_argument('-n', '--n_samples', nargs='?', type=int, default=1000000000, help='number of samples per worker')
parser.add_argument('-t', '--target', nargs='?', type=str, default='a', help='target vowel for imitation')
parser.add_argument('-p', '--parameters', nargs='*', type=str, default=['all'], help='vocal tract parameters to learn')
parser.add_argument('-s', '--sigma', nargs='?', type=float, default=0.4, help='step-size = sigma')
parser.add_argument('-i', '--infant', action='store_true', help='simulate infant speaker?')
parser.add_argument('-N', '--n_vowels', nargs='?', type=int, default=3, help='number of vowels')
#parser.add_argument('-l', '--loudness_factor', nargs='?', type=float, default=0.1, help='weighting of loudness in reward?')
#parser.add_argument('-m', '--softmax', action='store_true', help='use softmax reward?')
parser.add_argument('-I', '--intrinsic_motivation', action='store_true', help='emulate intrinsic motivation?')
parser.add_argument('-T', '--threshold', nargs='?', type=float, default=0.5, help='threshold for convergence')
parser.add_argument('-P', '--predefined', action='store_true', help='initialize with predefined configuration?')
parser.add_argument('-e', '--energy_factor', nargs='?', type=float, default=0.0, help='energy balance factor')
parser.add_argument('-a', '--alpha', nargs='?', type=float, default=1.0, help='alpha for constraint penalty')
parser.add_argument('-b', '--biased', action='store_true', help='use biased ESN?')
parser.add_argument('-R', '--N_reservoir', nargs='?', type=int, default=1000, help='size of used ESN')
parser.add_argument('-r', '--resample', action='store_true', help='resample invalid motor parameters?')
parser.add_argument('-o', '--normalize', action='store_true', help='normalize ESN output?')
parser.add_argument('-c', '--no_convergence', action='store_true', help='turn off convergence?')
parser.add_argument('-A', '--random_restart', action='store_true', help='restart search after bad solution from random learnt variables?')
parser.add_argument('-g', '--regularized', action='store_true', help='use regularized network?')
parser.add_argument('-l', '--lambda_', nargs='?', type=int, default=0, help='lambda: CMA-ES offspring population size')
parser.add_argument('-L', '--load_state', nargs='?', type=str, default=None, help='load saved state?')
parser.add_argument('-x', '--partition', nargs='?', type=str, default='sleuths', help='slurm partition to use')
parser.add_argument('-C', '--conditioning_maximum', nargs='?', type=float, default=1e14, help='maximal conditioning number')
parser.add_argument('-w', '--no_reward_convergence', action='store_true', help='ignore reward for convergence?')
parser.add_argument('-S', '--ptp_stop', nargs='?', type=float, default=0.001, help='convergence range')
parser.add_argument('-X', '--standard_params', action='store_true', help='use default parameters?')
parser.add_argument('-F', '--flat_tongue', action='store_true', help='simulate flat tongue, i.e. set all TS to 0?')
parser.add_argument('-d', '--debug', action='store_true', help='turn on debug mode?')
parser.add_argument('-k', '--constant_sigma0', action='store_true', help='keep sigma 0 constant?')
parser.add_argument('-z', '--repeat_simulations', type=int, default=0, help='number of simulation runs')

# Typical use:
# python mpi_rl.py -f all_example -p all -s 0.2 -i -N 3 -I -T 0.5 -e 0.0 -r -b -R 1000 -o -c -A -g -l 100 -C 1e10
# python mpi_rl.py -f all_default -p all -X

args = parser.parse_args()
verbose = args.verbose
folder = args.folder
i_stop = args.n_samples
target = args.target
parameters = args.parameters
sigma = args.sigma
infant = args.infant
n_vowels = args.n_vowels
#loudness_factor = args.loudness_factor
#softmax = args.softmax
softmax = True
intrinsic_motivation = args.intrinsic_motivation
#intrinsic_motivation = True
conf_threshold = args.threshold
predefined = args.predefined
energy_factor = args.energy_factor
alpha = args.alpha
biased = args.biased
N_reservoir = args.N_reservoir
resample = args.resample
normalize = args.normalize
no_convergence = args.no_convergence
random_restart = args.random_restart
reg = args.regularized
lambda_ = args.lambda_
load_state = args.load_state
partition = args.partition
cond_stop = args.conditioning_maximum
no_reward_convergence = args.no_reward_convergence
ptp_stop = args.ptp_stop
default = args.standard_params
flat_tongue = args.flat_tongue
debug = args.debug
constant_sigma0 = args.constant_sigma0
repeat_simulations = args.repeat_simulations

lambda_list = [4,6,7,8,8,9,9,10,10,10,11,11,11,11,12,12,12,12]

N = len(parameters)



if default:
    lambda_ = 100
    infant = True
    predefined = False
    biased = True
    softmax = True
    resample = True
    normalize = True
    no_convergence = True
    random_restart = True
    reg = True
    no_reward_convergence = True
    intrinsic_motivation = True
    sigma = 0.4
    conf_threshold = 0.5
    energy_factor = 0.0
    N_reservoir = 1000
    cond_stop = 1e14
    ptp_stop = 0.001
    n_vowels = 3
    

if lambda_ == 0:
    if N == 1:
        current_lambda = 12
    else:
        current_lambda = lambda_list[N-1]
else:
    current_lambda = lambda_


if infant:
    infant_string = '-i'
else:
    infant_string = ''
if predefined:
    predefined_string = '-P'
else:
    predefined_string = ''

p_string = ''
if N == 1:
    p_string = 'all'
else:
  for parameter in parameters:
    p_string += ' '+parameter
if biased:
    b_string = '-b'  
else:
    b_string = ''
'''
if N == 2:
    p_string = 'TCX TCY'

if '15' in parameters:
    p_string = 'HX HY VS TCX TCY TTX TTY TBX TBY TRX TRY TS1 TS2 TS3 TS4'
#'''
if 'visual' in parameters:
    p_string = 'HX HY VS TCX TCY TTX TTY TBX TBY TS1 TS2 TS3 TS4'
    if lambda_ == 0:
        current_lambda = 11
if 'flat_all' in parameters:
    p_string = 'HX HY JA LP LD VS TCX TCY TTX TTY TBX TBY'
    if lambda_ == 0:
        current_lambda = 11
if 'flat_visual' in parameters:
    p_string = 'HX HY VS TCX TCY TTX TTY TBX TBY'
    if lambda_ == 0:
        current_lambda = 10
if 'only_visual' in parameters:
    p_string = 'JA LP LD'
    if lambda_ == 0:
        current_lambda = 7

'''
elif '12' in paramters:
    p_string = 'TXC TCY TTX TTY TBX TBY TRX TRY TS1 TS2 TS3 TS4'
    current_lambda = 11
else:
    p_string = 'all'
#'''


n_workers = current_lambda + 1
#n_samples = i_stop * current_lambda


if softmax:
    soft_string = ' -m '
else:
    soft_string = ''

if resample:
    resample_string = '-r'
else:
    resample_string = ''

if normalize:
    normalize_string = '-o'
else:
    normalize_string = ''

if no_convergence:
    convergence_string = '-c'
else:
    convergence_string = ''

if random_restart:
    random_restart_string = '-A'
else:
    random_restart_string = ''

if reg:
    reg_string = '-g'
else:
    reg_string = ''

if load_state == None:
    load_string = ''
else:
    load_string = '-L '+load_state

if no_reward_convergence:
    no_reward_convergence_string = '-w'
else:
    no_reward_convergence_string = ''

if flat_tongue:
    flat_tongue_string = '-F'
else:
    flat_tongue_string = ''

if verbose:
    verbose_string = '-v'
else:
    verbose_string = ''

if debug:
    debug_string = '-d'
else:
    debug_string = ''

if constant_sigma0:
    constant_sigma0_string = '-k'
else:
    constant_sigma0_string = ''

if intrinsic_motivation:
    IM_string = '-I'
else:
    IM_string = ''



if repeat_simulations==0:
    os.system('salloc -t 10-1 -p '+partition+' -n '+str(n_workers)+' mpirun python rl_agent_mpi.py -f '+folder+' '+verbose_string+' -s '+str(sigma)+' -p '+p_string+' -N '+str(n_vowels)+' '+infant_string+' '+soft_string+' '+IM_string+' -T '+str(conf_threshold)+' -e '+str(energy_factor)+' -a '+str(alpha)+' '+b_string+' -R '+str(N_reservoir)+' '+normalize_string+' '+random_restart_string+' '+resample_string+' '+convergence_string+' '+reg_string+' '+load_string+' -C '+str(cond_stop)+' '+no_reward_convergence_string+' '+flat_tongue_string+' -S '+str(ptp_stop)+' '+debug_string+' '+constant_sigma0_string+' > logs/log_'+folder+'.dat')

elif repeat_simulations>0:
    for i in xrange(z):
        os.system('salloc -t 10-1 -p '+partition+' -n '+str(n_workers)+' mpirun python rl_agent_mpi.py -f '+folder+'_'+str(i)+' '+verbose_string+' -s '+str(sigma)+' -p '+p_string+' -N '+str(n_vowels)+' '+infant_string+' '+soft_string+' '+IM_string+' -T '+str(conf_threshold)+' -e '+str(energy_factor)+' -a '+str(alpha)+' '+b_string+' -R '+str(N_reservoir)+' '+normalize_string+' '+random_restart_string+' '+resample_string+' '+convergence_string+' '+reg_string+' '+load_string+' -C '+str(cond_stop)+' '+no_reward_convergence_string+' '+flat_tongue_string+' -S '+str(ptp_stop)+' '+debug_string+' '+constant_sigma0_string+' > logs/log_'+folder+'.dat')

'''
else:
#  for i in xrange(n_vowels):
#    vowel = vowels[i]

    os.system('salloc -t 10-1 -p '+partition+' -n '+str(n_workers)+' mpirun python rl_agent_mpi.py -f '+folder+' '+verbose_string+' -t '+target+' -s '+str(sigma)+' -p '+p_string+' '+predefined_string+' -N '+str(n_vowels)+' -T '+str(conf_threshold)+' -e '+str(energy_factor)+' -a '+str(alpha)+' '+infant_string+' '+soft_string+' '+normalize_string+' '+convergence_string+' '+resample_string+' -R '+str(N_reservoir)+' '+reg_string+' '+load_string+' -C '+str(cond_stop)+' '+no_reward_convergence_string+' '+flat_tongue_string+' -S '+str(ptp_stop)+' '+debug_string+' '+constant_sigma0_string+' > logs/log_'+folder+'.dat')
'''
'''
    if N == 2:
        if infant:
            os.system('salloc -p sleuths -n '+str(n_workers)+' mpirun python rl_agent_mpi.py -f '+vowel+'_'+folder+' -n '+str(n_samples)+' -t '+vowel+' -s '+str(sigma)+' -p TCX TCY -N '+str(n_vowels)+' -i -v > log_'+vowel+'_'+folder+'.dat')
        else:
            os.system('salloc -p sleuths -n '+str(n_workers)+' mpirun python rl_agent_mpi.py -f '+vowel+'_'+folder+' -n '+str(n_samples)+' -t '+vowel+' -s '+str(sigma)+' -p TCX TCY -N '+str(n_vowels)+' -v > log_'+vowel+'_'+folder+'.dat')
    if N == 1:
        if infant:
            os.system('salloc -p sleuths -n '+str(n_workers)+' mpirun python rl_agent_mpi.py -f '+vowel+'_'+folder+' -n '+str(n_samples)+' -t '+vowel+' -s '+str(sigma)+' -p all -N '+str(n_vowels)+' -i -v > log_'+vowel+'_'+folder+'.dat')
        else:
            os.system('salloc -p sleuths -n '+str(n_workers)+' mpirun python rl_agent_mpi.py -f '+vowel+'_'+folder+' -n '+str(n_samples)+' -t '+vowel+' -s '+str(sigma)+' -p all -N '+str(n_vowels)+' -v > log_'+vowel+'_'+folder+'.dat')
'''
