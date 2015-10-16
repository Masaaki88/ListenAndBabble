import ctypes				# for accessing C libraries 
from ctypes import cdll
import numpy				# for array handling
from numpy import array as ar
from create_ges import create_gesture	# for creatung gesture scores (-> create_ges.py)
from create_sp import create_speaker	# for creating speakers (-> create_sp.py)
from sys import argv			# for command line arguments
from os import system			# for file handling and message output
from os import path as pth



#########################################################################################
#
# Initialization
#
#########################################################################################


try:					# get syllable to simulate
 syllable = argv[1]
except IndexError:			# default = 'a'
 syllable = 'a'

try:					# get speaker to simulate
 speaker = argv[2]
except IndexError:			# default = 'adult'
 speaker = 'adult'


#system('echo simulating '+speaker+' speaking "'+syllable+'"')
print 'simulating '+speaker+' speaking "'+syllable+'"'
					# announce simulation task

name = syllable+'_'+speaker		# create simulation name


path = r'output/ '			# define output path
path = path[:-1]

def p_(path_):				# function for appending output path to file names
 global path
 full_path = path+path_
 return full_path



#dllFile = '/home/murakami/Master/VocalTractLab/VocaltractlabApi/VocalTractLabApi.so'		# C library to be used
dllFile = pth.abspath('VocalTractLabApi.so')
#print 'dllFile: '+dllFile
wavFile = pth.relpath('output/'+name+'.wav')			# output sound file
#wavFile = p_(wavFile)
areaFile = pth.relpath('output/'+name+'.txt')			# output area file
#areaFile = p_(areaFile)

params_u = ar([0.9073, -3.2279, 0.0, -4.0217, 1.0, 0.3882, 0.5847, -0.1, 0.3150, -0.5707, 2.0209, -1.0122, 1.8202, -0.1492, 0.5620, 0.1637, 0.0602, -0.0386])
params_a = ar([0.3296, -2.3640, 0.0, -4.3032, 0.0994, 0.8196, 1.0, -0.1, -0.4878, -1.2129, 1.9036, -1.5744, 1.3212, -1.0896, 1.0313, -0.1359, 0.4925, 0.0772])
params_i = ar([0.8580, -2.9237, -0.1488, -1.8808, -0.0321, 0.5695, 0.1438, -0.1, 0.6562, -0.3901, 2.6431, -0.6510, 2.1213, 0.3124, 0.3674, 0.034, -0.1274, -0.2887])

params = 0.1*params_u+0.9*params_u



#########################################################################################
#
# Generate gesture score and modify speaker
#
#########################################################################################



gestureFile = create_gesture(name, syllable, speaker)
					# create gesture score and define created .ges file


if (speaker != 'adult') and (speaker != 'infant'):	# only create speaker if neither adult nor infant used
 speakerFile = create_speaker(speaker, params)
else:
 speakerFile = r'speakers/ '[:-1]+speaker+'.speaker'





#########################################################################################
#
# Simulate Vocal Tract
#
#########################################################################################

#print 'VocalTractLabApi.so exists: '+str(pth.exists(dllFile))

lib = cdll.LoadLibrary(dllFile)		# load C library
lib.vtlGesToWav(speakerFile, gestureFile, wavFile, areaFile)
					# call function to generate sound file from gesture score


system('cp '+wavFile+' '+wavFile+'.bkup')

f = open(wavFile, 'r')
content = f.read()

f.close()
system('rm '+wavFile)
system('touch '+wavFile)

newfile = open(wavFile, 'w')

#system('rm new.wav')
#system('touch new.wav')

#newfile = open('new.wav', 'w')

#newfile.write(content[:8]+content[16:32]+content[40:44]+content[48:52]+content[56:64]+content[68:])
header = 'RIFF'+chr(0x8C)+chr(0x87)+chr(0x00)+chr(0x00)+'WAVEfmt'+chr(0x20)+chr(0x10)+chr(0x00)+chr(0x00)+chr(0x00)+chr(0x01)+chr(0x00)+chr(0x01)+chr(0x00)+'"V'+chr(0x00)+chr(0x00)+'D'+chr(0xAC)+chr(0x00)+chr(0x00)+chr(0x02)+chr(0x00)+chr(0x10)+chr(0x00)+'data'

newcontent = header+content[68:]

newfile.write(newcontent)

#f.close()
newfile.close()

system('rm '+wavFile+'.bkup')

#system('echo starting playback')
#system('start '+wavFile)		# play back sound file
