import ctypes			# for accessing C libraries 
import os			# for retrieving path names




#########################################################################################
#
# Load Library
#
#########################################################################################

dllFile = os.path.abspath('VocalTractLabApi.so')
lib = ctypes.cdll.LoadLibrary(dllFile)



#########################################################################################
#
# Synthesize from a gestural score
#
#########################################################################################


speakerFileName = os.path.abspath('adult.speaker')
filename = raw_input('file name: ')
gestureFileName = os.path.abspath(filename+'.ges')
wavFileName = os.path.abspath(filename+'.wav')
areaFileName = os.path.abspath(filename+'-areas.txt')

print 'Calling gesToWav()...'

failure = lib.vtlGesToWav(speakerFileName, gestureFileName, wavFileName, areaFileName)

if (failure != 0):
    print 'Error in vtlGesToWav()!'

print 'Repairing header...'

backup = wavFileName+'.bkup'
os.system('cp '+wavFileName+' '+backup)	# create backup of wav file

f = open(wavFileName, 'r')			# open wav file for reading
content = f.read()			# read content of wav file

f.close()				# close wav file
os.remove(wavFileName)				# delete wav file
os.system('touch '+wavFileName)		# create empty file with the same name

newfile = open(wavFileName, 'w')		# open new file for writing


header = 'RIFF'+chr(0x8C)+chr(0x87)+chr(0x00)+chr(0x00)+'WAVEfmt'+chr(0x20)+chr(0x10)+chr(0x00)+chr(0x00)+chr(0x00)+chr(0x01)+chr(0x00)+chr(0x01)+chr(0x00)+'"V'+chr(0x00)+chr(0x00)+'D'+chr(0xAC)+chr(0x00)+chr(0x00)+chr(0x02)+chr(0x00)+chr(0x10)+chr(0x00)+'data'
					# define working header

newcontent = header+content[68:]	# concatenate header with sound data
newfile.write(newcontent)		# write new file

newfile.close()				# close new file

os.remove(backup)				# delete backup

print 'Finished.'
