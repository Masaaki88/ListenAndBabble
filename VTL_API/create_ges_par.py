from os import system				# for file handling and message output
from os import path as pth




#########################################################################################
#
# Create gesture scores
#
#########################################################################################



def create_gesture(filename, speaker, pitch_var=0.0, len_var=1.0):
	# main function takes file name, syllable to be gestured, used speaker, pitch perturbation,
	#  and duration factor; outputs resulting file name

 filename_ = pth.relpath('VTL_API/gestures/'+filename+'.ges')

 print 'creating gesture file '+filename_
			
 system('rm '+filename_)                # delete previous version of gesture score
 system('touch '+filename_)             # create empty gesture score file

 print 'opening gesture file'
 file = open(filename_, 'w')			# access gesture score file for writing

 print 'writing gestural score'
 file.write('<gestural_score>\n')		# from here on: write gesture score -> vowel gestures



 ########################
 # Vowel gestures
 ########################
 duration_s = str(0.661085 * len_var)
 time_constant_s = str(0.015 * len_var)

 file.write('  <gesture_sequence type="vowel-gestures" unit="">\n')
						# this accesses the modified versions of the vowel to be learned from within the modified speaker file
 file.write('    <gesture value="input" slope="0.000000" duration_s="'+duration_s+'" time_constant_s="'+time_constant_s+'" neutral="0" />\n')
 file.write('  </gesture_sequence>\n')

 ########################
 # Lip gestures
 ########################
 file.write('  <gesture_sequence type="lip-gestures" unit="">\n')
 if consonant == 'm':
    file.write('    <gesture value="ll-labial-nas" slope="0.000000" duration_s="0.208372" time_constant_s="0.015000" neutral="0" />\n')
    file.write('    <gesture value="" slope="0.000000" duration_s="0.090543" time_constant_s="0.015000" neutral="1" />\n')
    file.write('    <gesture value="ll-labial-nas" slope="0.000000" duration_s="0.107907" time_constant_s="0.015000" neutral="0" />\n')
    file.write('    <gesture value="" slope="0.000000" duration_s="0.347287" time_constant_s="0.015000" neutral="1" />\n')
 file.write('  </gesture_sequence>\n')

 ########################
 # Tongue tip gestures
 ########################
 file.write('  <gesture_sequence type="tongue-tip-gestures" unit="">\n')

 file.write('  </gesture_sequence>\n')

 ########################
 # Tongue body gestures
 ########################
 file.write('  <gesture_sequence type="tongue-body-gestures" unit="">\n')

 file.write('  </gesture_sequence>\n')

 ########################
 # Velic gestures
 ########################
 file.write('  <gesture_sequence type="velic-gestures" unit="">\n')

 file.write('  </gesture_sequence>\n')

 ########################
 # Glottal shape gestures
 ########################

 time_constant_glottal = str(0.02 * len_var)

 file.write('  <gesture_sequence type="glottal-shape-gestures" unit="">\n')
 file.write('    gesture value="modal" slope="0.000000" duration_s="'+duration_s+'" time_constant_s="'+time_constant_glottal+'" neutral="0" />\n')
 file.write('  </gesture_sequence>\n')

 ########################
 # F0 gestures
 ########################


 durations = [str(0.084341*len_var), str(0.179845*len_var), str(0.235659*len_var)]
 time_constants = [str(0.01515*len_var), str(0.0099*len_var), str(0.0078*len_var)]
 slopes = [str(0.0/len_var), str(9.28/len_var), str(-27.68/len_var)]

 file.write('  <gesture_sequence type="f0-gestures" unit="st">\n')
 if speaker == 'adult':

  pitch = [str(32.0+pitch_var), str(34.0+pitch_var), str(28.0+pitch_var)]

  file.write('    <gesture value="'+pitch[0]+'00000" slope="'+slopes[0]+'" duration_s="'+durations[0]+'" time_constant_s="'+time_constants[0]+'" neutral="0" />\n')
  file.write('    <gesture value="'+pitch[1]+'00000" slope="'+slopes[1]+'" duration_s="'+durations[1]+'" time_constant_s="'+time_constants[1]+'" neutral="0" />\n')
  file.write('    <gesture value="'+pitch[2]+'00000" slope="'+slopes[2]+'" duration_s="'+durations[2]+'" time_constant_s="'+time_constants[2]+'" neutral="0" />\n')

 else:                      # use predefined infant speaker

  pitch = [str(52.0+pitch_var), str(54.0+pitch_var), str(48.0+pitch_var)]

  file.write('    <gesture value="'+pitch[0]+'00000" slope="'+slopes[0]+'" duration_s="'+durations[0]+'" time_constant_s="'+time_constants[0]+'" neutral="0" />\n')
  file.write('    <gesture value="'+pitch[1]+'00000" slope="'+slopes[1]+'" duration_s="'+durations[1]+'" time_constant_s="'+time_constants[1]+'" neutral="0" />\n')
  file.write('    <gesture value="'+pitch[2]+'00000" slope="'+slopes[2]+'" duration_s="'+durations[2]+'" time_constant_s="'+time_constants[2]+'" neutral="0" />\n')


 file.write('  </gesture_sequence>\n')

 ########################
 # Lung pressure gestures
 ########################

 durations_lung = [str(0.01*len_var), str(0.528295*len_var), str(0.1*len_var)]
 time_constants_lung = [str(0.005*len_var), str(0.005*len_var), str(0.005*len_var)]

 file.write('  <gesture_sequence type="lung-pressure-gestures" unit="Pa">\n')
 file.write('    <gesture value="0.000000" slope="0.000000" duration_s="0.050000" time_constant_s="0.050000" neutral="0" />\n')
 file.write('    <gesture value="0.000000" slope="0.000000" duration_s="'+durations_lung[0]+'" time_constant_s="'+time_constants_lung[0]+'" neutral="0" />\n')
 file.write('    <gesture value="1000.000000" slope="0.000000" duration_s="'+durations_lung[1]+'" time_constant_s="'+time_constants_lung[1]+'" neutral="0" />\n')
 file.write('    <gesture value="0.000000" slope="0.000000" duration_s="'+durations_lung[2]+'" time_constant_s="'+time_constants_lung[2]+'" neutral="0" />\n')
 file.write('  </gesture_sequence>\n')

 file.write('</gestural_score>')



 print 'closing gesture file'
 file.close()

 return filename_
