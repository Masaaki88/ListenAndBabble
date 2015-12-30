#!/usr/bin/env python2
# -*- encoding: utf-8 -*-

import ctypes
import os
import shutil

from create_sp_par import create_speaker    # for creating speakers (-> create_sp.py)


LIB_VTL = ctypes.cdll.LoadLibrary(os.path.abspath('VTL_API/VocalTractLabApi.so'))

# define working header
WAV_HEADER = 'RIFF'+chr(0x8C)+chr(0x87)+chr(0x00)+chr(0x00)+'WAVEfmt'+chr(0x20)+chr(0x10)+chr(0x00)+chr(0x00)+chr(0x00)+chr(0x01)+chr(0x00)+chr(0x01)+chr(0x00)+'"V'+chr(0x00)+chr(0x00)+'D'+chr(0xAC)+chr(0x00)+chr(0x00)+chr(0x02)+chr(0x00)+chr(0x10)+chr(0x00)+'data'




def par_to_wav(params, speaker='adult', simulation_name='', pitch_var=0.0,
               len_var=1.0, verbose=False, rank=1, different_folder='',
               monotone=False):
    """
    Creates a wave file with vocal tract lab out of the given parameters.

    """

    if verbose:
        print('simulating ' + speaker)
    name = speaker + '_' + simulation_name


    if different_folder == '':
        wav_file = os.path.relpath(os.path.join('VTL_API/output/', name, '_',
                                               str(rank), '.wav'))
    else:
        wav_file = os.path.relpath(different_folder)
    area_file = os.path.relpath(os.path.join('VTL_API/output/', name, '_',
                                             str(rank), '.txt'))


    # gestureFile = create_gesture(name, speaker, pitch_var, len_var)
    if not speaker in ('adult', 'infant'):
        raise ValueError("speaker needs to be either 'adult' or 'infant'.")
    if monotone:
        gesture_file = os.path.relpath('VTL_API/gestures/input_%s_monotone.ges' % speaker)
    else:
        gesture_file = os.path.relpath('VTL_API/gestures/input_%s.ges' % speaker)

    speaker_file = create_speaker(speaker, params, name, verbose=verbose, rank=rank)


    # run through vocal tract lab
    LIB_VTL.vtlGesToWav(speaker_file, gesture_file, wav_file, area_file)


    # Repair header of wav File
    with open(wav_file, 'rb') as file_:
        content = file_.read()

    shutil.move(wav_file, wav_file + '.bkup')

    with open(wav_file, 'wb') as newfile:
        newcontent = WAV_HEADER + content[68:]
        newfile.write(newcontent)

    os.remove(wav_file + '.bkup')

    return wav_file

