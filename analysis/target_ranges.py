from aux_functions import *
from mpi4py import MPI


comm = MPI.COMM_WORLD                   # setup MPI framework
n_workers = comm.Get_size()             # total number of workers / parallel processes
rank = comm.Get_rank()              # id of this worker -> master: 0

flow = flow1000_reg


___FLAT___ = True


def relParToWave_mpi(x_rel,rank):

    x_abs = get_abs_coord(x_rel)

    wavFile = parToWave(x_abs, speaker='infant', simulation_name='infant', pitch_var=0.0, len_var=1.0, verbose=False, rank=rank, different_folder='aux_infant_'+str(rank)+'.wav', monotone=False)

    sound = Sound(wavFile)
    sound = correct_initial(sound)      # call correct_initial to remove initial burst
    sound_resampled = get_resampled(sound)
                                        # call get_resampled to adapt generated sound to AN model
    sound_extended = get_extended(sound_resampled)
                                        # call get_extended to equalize duration of all sounds
    sound_extended.save(wavFile)

    return sound_extended


rel_coords = (get_rel_coord(params_a), get_rel_coord(params_i), get_rel_coord(params_u))
    
if ___FLAT___:
    for rel_coord in rel_coords:
        rel_coord[-4:] = np.zeros(4)

data = []
for i_vow in xrange(3):
    data_current_vowel = []
    for deviation in np.arange(0.0,1.001,0.01):
        current_par = rel_coords[i_vow].copy()
        current_par[rank] = deviation
        
        current_sound = relParToWave_mpi(current_par,rank)
        current_confidences = sound_to_confidences(current_sound, flow, True)

        data_current_vowel.append(current_confidences[i_vow])
    data.append(data_current_vowel)

send = [rank, data]

collect_data = comm.gather(send, root=0)

if rank == 0:
    print 'collected data:', collect_data
    collect_data.sort()
    to_write = []
    for i in xrange(n_workers):
        to_write.append(collect_data[i][1])
    to_write = np.array(to_write)

    if ___FLAT___:
        out = open('target_ranges_flat.dat', 'w')    
    else:
        out = open('target_ranges.dat', 'w')    
    to_write.dump(out)
    out.close()
