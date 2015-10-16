import ctypes				# for accessing C libraries 
from ctypes import cdll
import numpy				# for array handling
from numpy import array as ar
from create_ges import create_gesture	# for creatung gesture scores (-> create_ges.py)
from create_sp import create_speaker	# for creating speakers (-> create_sp.py)
from sys import argv			# for command line arguments
from os import system			# for file handling and message output
from os import remove
from os import path as pth



#########################################################################################
#
# Initialization
#
#########################################################################################


def gesToWave(syllable='a', speaker='adult', pitch_var=0.0, len_var=1.0):

 print 'simulating '+speaker+' speaking "'+syllable+'"'
					# announce simulation task
 name = syllable+'_'+speaker+'['+str(pitch_var)+'_'+str(len_var)+']'		# create simulation name



 dllFile = pth.abspath('VTL_API/VocalTractLabApi.so')	# C library to be used
 wavFile = pth.relpath('VTL_API/output/'+name+'.wav')	# output sound file
 areaFile = pth.relpath('VTL_API/output/'+name+'.txt')	# output area file


 params_u = ar([0.9073, -3.2279, 0.0, -4.0217, 1.0, 0.3882, 0.5847, -0.1, 0.3150, -0.5707, 2.0209, -1.0122, 1.8202, -0.1492, 0.5620, 0.1637, 0.0602, -0.0386])
 params_a = ar([0.3296, -2.3640, 0.0, -4.3032, 0.0994, 0.8196, 1.0, -0.1, -0.4878, -1.2129, 1.9036, -1.5744, 1.3212, -1.0896, 1.0313, -0.1359, 0.4925, 0.0772])
 params_i = ar([0.8580, -2.9237, -0.1488, -1.8808, -0.0321, 0.5695, 0.1438, -0.1, 0.6562, -0.3901, 2.6431, -0.6510, 2.1213, 0.3124, 0.3674, 0.034, -0.1274, -0.2887])
 params_schwa = ar([1.0, -2.643, 0.0, -2.0, -0.07, 0.524, 0.0, -0.1, 0.0, -0.426, -0.767, 2.036, -0.578, 1.163, 0.321, -1.853, -1.7267, 0.0, 0.046, 0.116, 0.116])
                                                        # prototypical vowel shapes of infant speaker
 params = params_u                                      # arbitrary shape parameters for new speaker (for sensorimotor learning)



#########################################################################################
#
# Generate gesture score and modify speaker
#
#########################################################################################



 gestureFile = create_gesture(name, syllable, speaker, pitch_var, len_var)
					# create gesture score and define created .ges file


 if (speaker != 'adult') and (speaker != 'infant'):	# only create speaker if neither predefined adult nor infant used
  speakerFile = create_speaker(speaker, params)
 else:
  speakerFile = pth.relpath('VTL_API/speakers/'+speaker+'.speaker')




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
 remove(wavFile)				# delete wav file
 system('touch '+wavFile)		# create empty file with the same name

 newfile = open(wavFile, 'w')		# open new file for writing


 header = 'RIFF'+chr(0x8C)+chr(0x87)+chr(0x00)+chr(0x00)+'WAVEfmt'+chr(0x20)+chr(0x10)+chr(0x00)+chr(0x00)+chr(0x00)+chr(0x01)+chr(0x00)+chr(0x01)+chr(0x00)+'"V'+chr(0x00)+chr(0x00)+'D'+chr(0xAC)+chr(0x00)+chr(0x00)+chr(0x02)+chr(0x00)+chr(0x10)+chr(0x00)+'data'
					# define working header

 newcontent = header+content[68:]	# concatenate header with sound data
 newfile.write(newcontent)		# write new file

 newfile.close()				# close new file

 remove(backup)				# delete backup


 return wavFile
