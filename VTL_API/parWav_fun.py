import ctypes				# for accessing C libraries 
from ctypes import cdll
import numpy				# for array handling
from numpy import array as ar
from create_ges_par import create_gesture	# for creatung gesture scores (-> create_ges.py)
from create_sp_par import create_speaker	# for creating speakers (-> create_sp.py)
from sys import argv			# for command line arguments
from os import system			# for file handling and message output
from os import remove
from os import path as pth



#########################################################################################
#
# Initialization
#
#########################################################################################


def parToWave(params, speaker='adult', simulation_name='', pitch_var=0.0, len_var=1.0, verbose=False, rank=1, different_folder='', monotone=False):

 if verbose:
  print 'simulating '+speaker
					# announce simulation task
 name = speaker+'_'+simulation_name		# create simulation name



 dllFile = pth.abspath('VTL_API/VocalTractLabApi.so')	# C library to be used
 if different_folder == '':
     wavFile = pth.relpath('VTL_API/output/'+name+'_'+str(rank)+'.wav')	# output sound file
 else:
     wavFile = pth.relpath(different_folder)
 areaFile = pth.relpath('VTL_API/output/'+name+'_'+str(rank)+'.txt')	# output area file





#########################################################################################
#
# Generate gesture score and modify speaker
#
#########################################################################################



 if speaker == 'adult':
  if monotone:
    gestureFile = pth.relpath('VTL_API/gestures/input_adult_monotone.ges')
  else:
    gestureFile = pth.relpath('VTL_API/gestures/input_adult.ges')
 if speaker == 'infant':
  if monotone:
    gestureFile = pth.relpath('VTL_API/gestures/input_infant_monotone.ges')
  else:
    gestureFile = pth.relpath('VTL_API/gestures/input_infant.ges')
					# create gesture score and define created .ges file


 if (speaker != 'adult') and (speaker != 'infant'):	# only create speaker if neither predefined adult nor infant used
    print 'Error: no valid speaker detected!'
    print 'Choose either "adult" or "infant" speaker.'
    return True
 else:
  speakerFile = create_speaker(speaker, params, name, verbose=verbose, rank=rank)




#########################################################################################
#
# Simulate Vocal Tract
#
#########################################################################################



 lib = cdll.LoadLibrary(dllFile)		# load C library
 lib.vtlGesToWav(speakerFile, gestureFile, wavFile, areaFile)
					# call function to generate sound file from gesture score



#########################################################################################
#
# Repair header of wav File
#
#########################################################################################


 backup = wavFile+'.bkup'
 system('cp '+wavFile+' '+backup)	# create backup of wav file

 f = open(wavFile, 'r')			# open wav file for reading
 content = f.read()			# read content of wav file

 f.close()				# close wav file
 system('rm '+wavFile)
 system('touch '+wavFile)		# create empty file with the same name

 newfile = open(wavFile, 'w')		# open new file for writing


 header = 'RIFF'+chr(0x8C)+chr(0x87)+chr(0x00)+chr(0x00)+'WAVEfmt'+chr(0x20)+chr(0x10)+chr(0x00)+chr(0x00)+chr(0x00)+chr(0x01)+chr(0x00)+chr(0x01)+chr(0x00)+'"V'+chr(0x00)+chr(0x00)+'D'+chr(0xAC)+chr(0x00)+chr(0x00)+chr(0x02)+chr(0x00)+chr(0x10)+chr(0x00)+'data'
					# define working header

 newcontent = header+content[68:]	# concatenate header with sound data
 newfile.write(newcontent)		# write new file

 newfile.close()				# close new file

 system('rm '+backup)

 return wavFile
