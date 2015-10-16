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


speakerFileName = os.path.abspath('child-1y.speaker')
gestureFileName = os.path.abspath('mama-child.ges')
wavFileName = os.path.abspath('mama-child.wav')
areaFileName = os.path.abspath('mama-child-areas.txt')

print 'Calling gesToWav()...'

failure = lib.vtlGesToWav(speakerFileName, gestureFileName, wavFileName, areaFileName)

if (failure != 0):
    print 'Error in vtlGesToWav()!'

print 'Finished.'
