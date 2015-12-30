from os import system			# for handling files and message output
from os import path as pth





def get_dic(p):             # convert input parameters to dictionary
 table = {'HX':p[0], 'HY':p[1], 'JA':p[2], 'LP':p[3], 'LD':p[4], 'VS':p[5], 'TCX':p[6], 'TCY':p[7], 'TTX':p[8], 'TTY':p[9], 'TBX':p[10], 'TBY':p[11], 'TS1':p[12], 'TS2':p[13], 'TS3':p[14], 'TS4':p[15]}
 return table



#########################################################################################
#
# Create speaker file
#
#########################################################################################


def create_speaker(speaker, params, name, verbose=False, rank=1):	# main function takes output file name and returns output file name with output path
					                    #  params: float[16] vector for vocal tract configuration
 if (speaker!='adult') and (speaker!='infant'):
    print 'Error: no valid speaker detected!'
    print 'Choose either "adult" or "infant" speaker.'
    return True

 filename_ = pth.relpath('VTL_API/speakers/'+name+'_input_'+str(rank)+'.speaker')
                                        # correct file name with respect to output path

 if verbose:
  print 'creating speaker file '+filename_

 system('rm '+filename_)        		# delete previous version of speaker file
 system('touch '+filename_)             # create empty speaker file

 if verbose:
  print 'opening speaker file'
 file = open(filename_, 'w')		# access emtpy file for writing

 dic = get_dic(params)                  # get parameter dictionary

 if verbose:
  print 'writing speaker shapes'
 file.write('<speaker>\n')		# from here on: write speaker file -> vocal shapes

 ###########################
 # Vocal tract model
 ###########################
 file.write('  <vocal_tract_model>\n')

  ##########################
  # Anatomy
  ##########################
 file.write('    <anatomy>\n')
 if speaker == 'infant':
  file.write('''      <palate>
        <p0 x="0.0000" z="-1.3895" teeth_height="0.0000" top_teeth_width="0.8105" bottom_teeth_width="0.8105" palate_height="0.8734" palate_angle_deg="60.8000"/>
        <p1 x="0.0000" z="-1.3895" teeth_height="0.0000" top_teeth_width="0.8105" bottom_teeth_width="0.8105" palate_height="0.8734" palate_angle_deg="60.8000"/>
        <p2 x="0.0000" z="-1.3895" teeth_height="0.0000" top_teeth_width="0.7719" bottom_teeth_width="0.7719" palate_height="0.8734" palate_angle_deg="60.8000"/>
        <p3 x="0.8342" z="-1.3895" teeth_height="0.3000" top_teeth_width="0.7719" bottom_teeth_width="0.7719" palate_height="0.9611" palate_angle_deg="60.8000"/>
        <p4 x="1.5342" z="-1.2351" teeth_height="0.3600" top_teeth_width="0.6176" bottom_teeth_width="0.6176" palate_height="0.8409" palate_angle_deg="60.8000"/>
        <p5 x="2.1842" z="-1.0807" teeth_height="0.4200" top_teeth_width="0.5404" bottom_teeth_width="0.5404" palate_height="0.4205" palate_angle_deg="38.0000"/>
        <p6 x="2.5842" z="-0.8491" teeth_height="0.4800" top_teeth_width="0.5018" bottom_teeth_width="0.2316" palate_height="0.0901" palate_angle_deg="23.4000"/>
        <p7 x="2.7342" z="-0.4632" teeth_height="0.4800" top_teeth_width="0.6176" bottom_teeth_width="0.1544" palate_height="0.0000" palate_angle_deg="0.0000"/>
        <p8 x="2.7342" z="0.0000" teeth_height="0.4800" top_teeth_width="0.6561" bottom_teeth_width="0.1544" palate_height="0.0000" palate_angle_deg="0.0000"/>
      </palate>\n''')
  file.write('''      <jaw fulcrum_x="-6.5000" fulcrum_y="2.0000" rest_pos_x="0.0000" rest_pos_y="-0.8000" tooth_root_length="0.4800">
        <p0 x="0.0000" z="-1.4667" teeth_height="0.0000" top_teeth_width="0.8105" bottom_teeth_width="0.8105" jaw_height="0.9000" jaw_angle_deg="69.5000"/>
        <p1 x="0.0000" z="-1.4667" teeth_height="0.0000" top_teeth_width="0.8491" bottom_teeth_width="0.8491" jaw_height="0.9000" jaw_angle_deg="69.5000"/>
        <p2 x="0.3842" z="-1.4667" teeth_height="0.3000" top_teeth_width="0.8105" bottom_teeth_width="0.8105" jaw_height="0.9000" jaw_angle_deg="69.5000"/>
        <p3 x="1.3842" z="-1.2351" teeth_height="0.3000" top_teeth_width="0.6947" bottom_teeth_width="0.6947" jaw_height="0.9000" jaw_angle_deg="69.5000"/>
        <p4 x="2.0842" z="-1.0807" teeth_height="0.3000" top_teeth_width="0.5790" bottom_teeth_width="0.5790" jaw_height="0.6000" jaw_angle_deg="42.2000"/>
        <p5 x="2.4842" z="-0.8491" teeth_height="0.3300" top_teeth_width="0.4632" bottom_teeth_width="0.5404" jaw_height="0.2400" jaw_angle_deg="35.8000"/>
        <p6 x="2.6842" z="-0.5404" teeth_height="0.3600" top_teeth_width="0.2316" bottom_teeth_width="0.6176" jaw_height="0.0780" jaw_angle_deg="31.4000"/>
        <p7 x="2.7342" z="-0.3860" teeth_height="0.4200" top_teeth_width="0.1544" bottom_teeth_width="0.6947" jaw_height="0.0000" jaw_angle_deg="0.0000"/>
        <p8 x="2.7342" z="0.0000" teeth_height="0.4200" top_teeth_width="0.1544" bottom_teeth_width="0.6947" jaw_height="0.0000" jaw_angle_deg="0.0000"/>
      </jaw>\n''')
  file.write('<lips width="0.5348"/>\n')
  file.write('''      <tongue>
        <tip radius="0.1488"/>
        <body radius_x="1.3388" radius_y="0.9921"/>
        <root automatic_calc="1" trx_slope="0.8316" trx_intercept="-3.4529" try_slope="1.1254" try_intercept="-1.2473"/>
      </tongue>\n''')
  file.write('''      <velum uvula_width="0.3858" uvula_height="0.4961" uvula_depth="0.5404" max_nasal_port_area="1.6439" >
        <low points="-1.331 -0.7 -1.065 -0.3853 -0.7453 -0.1493 -0.3194 0.126 0 0.2834 "/>
        <mid points="-1.863 -0.1493 -1.65 0.08668 -1.224 0.244 -0.6389 0.4014 0 0.5351 "/>
        <high points="-1.863 0.244 -1.597 0.598 -1.171 0.7947 -0.6389 0.834 0 0.8734 "/>
      </velum>\n''')
  file.write('      <pharynx fulcrum_x="-2.5256" fulcrum_y="0.8057" rotation_angle_deg="-90.0000" top_rib_y="-1.4000" upper_depth="2.9334" lower_depth="2.6246" back_side_width="1.5000"/>\n')
  file.write('''      <larynx upper_depth="0.7719" lower_depth="0.7719" epiglottis_width="0.2756" epiglottis_height="0.8819" epiglottis_depth="1.0807" epiglottis_angle_deg="100.0000">
        <narrow points="1.378 0 0.8036 -0.125 1.08 -0.75 1.768 -2 1.133 -2 0.63 -0.75 0 -0.625 0 0 "/>
        <wide points="2.526 0 1.63 -0.125 1.348 -0.75 1.768 -2 1.133 -2 0.8978 -0.75 0 -0.625 0 0 "/>
      </larynx>\n''')
  file.write('      <subglottal_cavity length="14.4000"/>\n')
  file.write('      <nasal_cavity length="7.1000"/>\n')

 if speaker == 'adult':
  file.write('''      <palate>
        <p0 x="0.2000" z="-2.3000" teeth_height="0.5000" top_teeth_width="1.0500" bottom_teeth_width="1.0500" palate_height="1.3000" palate_angle_deg="39.5000"/>
        <p1 x="0.9000" z="-2.2000" teeth_height="0.5000" top_teeth_width="1.0500" bottom_teeth_width="1.0500" palate_height="1.1500" palate_angle_deg="39.5000"/>
        <p2 x="1.8000" z="-2.0000" teeth_height="0.5000" top_teeth_width="1.0000" bottom_teeth_width="1.0000" palate_height="1.4250" palate_angle_deg="60.8000"/>
        <p3 x="2.8000" z="-1.8000" teeth_height="0.5000" top_teeth_width="1.0000" bottom_teeth_width="1.0000" palate_height="1.6000" palate_angle_deg="60.8000"/>
        <p4 x="3.5000" z="-1.6000" teeth_height="0.6000" top_teeth_width="0.8000" bottom_teeth_width="0.8000" palate_height="1.4000" palate_angle_deg="60.8000"/>
        <p5 x="4.1500" z="-1.4000" teeth_height="0.7000" top_teeth_width="0.7000" bottom_teeth_width="0.7000" palate_height="0.7000" palate_angle_deg="38.0000"/>
        <p6 x="4.5500" z="-1.1000" teeth_height="0.8000" top_teeth_width="0.6500" bottom_teeth_width="0.3000" palate_height="0.1500" palate_angle_deg="23.4000"/>
        <p7 x="4.7000" z="-0.6000" teeth_height="0.8000" top_teeth_width="0.8000" bottom_teeth_width="0.2000" palate_height="0.0000" palate_angle_deg="0.0000"/>
        <p8 x="4.7000" z="0.0000" teeth_height="0.8000" top_teeth_width="0.8500" bottom_teeth_width="0.2000" palate_height="0.0000" palate_angle_deg="0.0000"/>
      </palate>
      <jaw fulcrum_x="-6.5000" fulcrum_y="2.0000" rest_pos_x="0.0000" rest_pos_y="-1.2000" tooth_root_length="0.8000">
        <p0 x="0.2000" z="-2.3000" teeth_height="0.5000" top_teeth_width="1.0500" bottom_teeth_width="1.0500" jaw_height="1.5000" jaw_angle_deg="69.5000"/>
        <p1 x="1.2000" z="-2.2000" teeth_height="0.5000" top_teeth_width="1.1000" bottom_teeth_width="1.1000" jaw_height="1.5000" jaw_angle_deg="69.5000"/>
        <p2 x="2.2000" z="-1.9000" teeth_height="0.5000" top_teeth_width="1.0500" bottom_teeth_width="1.0500" jaw_height="1.5000" jaw_angle_deg="69.5000"/>
        <p3 x="3.2000" z="-1.6000" teeth_height="0.5000" top_teeth_width="0.9000" bottom_teeth_width="0.9000" jaw_height="1.5000" jaw_angle_deg="69.5000"/>
        <p4 x="3.9000" z="-1.4000" teeth_height="0.5000" top_teeth_width="0.7500" bottom_teeth_width="0.7500" jaw_height="1.0000" jaw_angle_deg="42.2000"/>
        <p5 x="4.3000" z="-1.1000" teeth_height="0.5500" top_teeth_width="0.6000" bottom_teeth_width="0.7000" jaw_height="0.4000" jaw_angle_deg="35.8000"/>
        <p6 x="4.5000" z="-0.7000" teeth_height="0.6000" top_teeth_width="0.3000" bottom_teeth_width="0.8000" jaw_height="0.1300" jaw_angle_deg="31.4000"/>
        <p7 x="4.5500" z="-0.5000" teeth_height="0.7000" top_teeth_width="0.2000" bottom_teeth_width="0.9000" jaw_height="0.0000" jaw_angle_deg="0.0000"/>
        <p8 x="4.5500" z="0.0000" teeth_height="0.7000" top_teeth_width="0.2000" bottom_teeth_width="0.9000" jaw_height="0.0000" jaw_angle_deg="0.0000"/>
      </jaw>
      <lips width="1.3000"/>
      <tongue>
        <tip radius="0.2000"/>
        <body radius_x="1.8000" radius_y="1.8000"/>
        <root automatic_calc="1" trx_slope="0.9380" trx_intercept="-5.1100" try_slope="0.8310" try_intercept="-3.0300"/>
      </tongue>
      <velum uvula_width="0.7000" uvula_height="0.9000" uvula_depth="0.7000" max_nasal_port_area="2.0000" >
        <low points="-1.25 -0.7 -1 -0.3 -0.7 0 -0.3 0.35 0 0.55 "/>
        <mid points="-1.75 0 -1.55 0.3 -1.15 0.5 -0.6 0.7 0 0.87 "/>
        <high points="-1.75 0.5 -1.5 0.95 -1.1 1.2 -0.6 1.25 0 1.3 "/>
      </velum>
      <pharynx fulcrum_x="-2.3720" fulcrum_y="1.2140" rotation_angle_deg="-98.0000" top_rib_y="-1.4000" upper_depth="3.8000" lower_depth="3.4000" back_side_width="1.5000"/>
      <larynx upper_depth="1.0000" lower_depth="1.0000" epiglottis_width="0.5000" epiglottis_height="1.6000" epiglottis_depth="1.4000" epiglottis_angle_deg="100.0000">
        <narrow points="1.8 0 1.05 -0.2 1.55 -1.2 2.68 -3.2 1.48 -3.2 1.1 -1.2 0 -1 0 0 "/>
        <wide points="3.3 0 2.13 -0.2 1.9 -1.2 2.68 -3.2 1.48 -3.2 1.45 -1.2 0 -1 0 0 "/>
      </larynx>
      <subglottal_cavity length="23.0000"/>
      <nasal_cavity length="11.4000"/>\n''')

  ###########################
  # Parameters
  ###########################
 if speaker == 'infant':
  file.write('''      <param index="0"  name="HX"  min="0.000"  max="1.000"  neutral="1.000" />
      <param index="1"  name="HY"  min="-3.228"  max="-1.850"  neutral="-2.643" />
      <param index="2"  name="JX"  min="-0.372"  max="0.000"  neutral="0.000" />
      <param index="3"  name="JA"  min="-7.000"  max="0.000"  neutral="-2.000" />
      <param index="4"  name="LP"  min="-1.000"  max="1.000"  neutral="-0.070" />
      <param index="5"  name="LD"  min="-1.102"  max="2.205"  neutral="0.524" />
      <param index="6"  name="VS"  min="0.000"  max="1.000"  neutral="0.000" />
      <param index="7"  name="VO"  min="-0.100"  max="1.000"  neutral="-0.100" />
      <param index="8"  name="WC"  min="0.000"  max="1.000"  neutral="0.000" />
      <param index="9"  name="TCX"  min="-3.194"  max="2.327"  neutral="-0.426" />
      <param index="10"  name="TCY"  min="-1.574"  max="0.630"  neutral="-0.767" />
      <param index="11"  name="TTX"  min="0.873"  max="3.200"  neutral="2.036" />
      <param index="12"  name="TTY"  min="-1.574"  max="1.457"  neutral="-0.578" />
      <param index="13"  name="TBX"  min="-3.194"  max="2.327"  neutral="1.163" />
      <param index="14"  name="TBY"  min="-1.574"  max="2.835"  neutral="0.321" />
      <param index="15"  name="TRX"  min="-4.259"  max="1.163"  neutral="0.000" />
      <param index="16"  name="TRY"  min="-3.228"  max="0.079"  neutral="0.053" />
      <param index="17"  name="TS1"  min="-1.081"  max="1.081"  neutral="0.000" />
      <param index="18"  name="TS2"  min="-1.081"  max="1.081"  neutral="0.046" />
      <param index="19"  name="TS3"  min="-1.081"  max="1.081"  neutral="0.116" />
      <param index="20"  name="TS4"  min="-1.081"  max="1.081"  neutral="0.116" />
      <param index="21"  name="MA1"  min="0.000"  max="0.300"  neutral="0.000" />
      <param index="22"  name="MA2"  min="0.000"  max="0.300"  neutral="0.000" />
      <param index="23"  name="MA3"  min="0.000"  max="0.300"  neutral="0.000" />\n''')
 if speaker == 'adult':
  file.write('''      <param index="0"  name="HX"  min="0.000"  max="1.000"  neutral="1.000" />
      <param index="1"  name="HY"  min="-6.000"  max="-3.500"  neutral="-4.750" />
      <param index="2"  name="JX"  min="-0.500"  max="0.000"  neutral="0.000" />
      <param index="3"  name="JA"  min="-7.000"  max="0.000"  neutral="-2.000" />
      <param index="4"  name="LP"  min="-1.000"  max="1.000"  neutral="-0.070" />
      <param index="5"  name="LD"  min="-2.000"  max="4.000"  neutral="0.950" />
      <param index="6"  name="VS"  min="0.000"  max="1.000"  neutral="0.000" />
      <param index="7"  name="VO"  min="-0.100"  max="1.000"  neutral="-0.100" />
      <param index="8"  name="WC"  min="0.000"  max="1.000"  neutral="0.000" />
      <param index="9"  name="TCX"  min="-3.000"  max="4.000"  neutral="-0.400" />
      <param index="10"  name="TCY"  min="-3.000"  max="1.000"  neutral="-1.460" />
      <param index="11"  name="TTX"  min="1.500"  max="5.500"  neutral="3.500" />
      <param index="12"  name="TTY"  min="-3.000"  max="2.500"  neutral="-1.000" />
      <param index="13"  name="TBX"  min="-3.000"  max="4.000"  neutral="2.000" />
      <param index="14"  name="TBY"  min="-3.000"  max="5.000"  neutral="0.500" />
      <param index="15"  name="TRX"  min="-4.000"  max="2.000"  neutral="0.000" />
      <param index="16"  name="TRY"  min="-6.000"  max="0.000"  neutral="0.000" />
      <param index="17"  name="TS1"  min="-1.400"  max="1.400"  neutral="0.000" />
      <param index="18"  name="TS2"  min="-1.400"  max="1.400"  neutral="0.060" />
      <param index="19"  name="TS3"  min="-1.400"  max="1.400"  neutral="0.150" />
      <param index="20"  name="TS4"  min="-1.400"  max="1.400"  neutral="0.150" />
      <param index="21"  name="MA1"  min="0.000"  max="0.300"  neutral="0.000" />
      <param index="22"  name="MA2"  min="0.000"  max="0.300"  neutral="0.000" />
      <param index="23"  name="MA3"  min="0.000"  max="0.300"  neutral="0.000" />\n''')
 file.write('    </anatomy>\n')

  ###########################
  # Vocal shapes
  ###########################
 file.write('    <shapes>\n')			# this part will be modified during learning
 if speaker == 'infant':
  file.write('''      <shape name="initial">
        <param name="HX" value="1.0000" domi="100.0"/>
        <param name="HY" value="-2.6430" domi="100.0"/>
        <param name="JX" value="0.0000" domi="100.0"/>
        <param name="JA" value="-2.0000" domi="100.0"/>
        <param name="LP" value="-0.0700" domi="100.0"/>
        <param name="LD" value="0.5240" domi="100.0"/>
        <param name="VS" value="0.0000" domi="100.0"/>
        <param name="VO" value="-0.1000" domi="100.0"/>
        <param name="WC" value="0.0000" domi="100.0"/>
        <param name="TCX" value="-0.4260" domi="100.0"/>
        <param name="TCY" value="-0.7670" domi="100.0"/>
        <param name="TTX" value="2.0360" domi="100.0"/>
        <param name="TTY" value="-0.5780" domi="100.0"/>
        <param name="TBX" value="1.1630" domi="100.0"/>
        <param name="TBY" value="0.3210" domi="100.0"/>
        <param name="TRX" value="-1.8530" domi="100.0"/>
        <param name="TRY" value="-1.7267" domi="100.0"/>
        <param name="TS1" value="0.0000" domi="100.0"/>
        <param name="TS2" value="0.0460" domi="100.0"/>
        <param name="TS3" value="0.1160" domi="100.0"/>
        <param name="TS4" value="0.1160" domi="100.0"/>
        <param name="MA1" value="0.0000" domi="100.0"/>
        <param name="MA2" value="0.0000" domi="100.0"/>
        <param name="MA3" value="0.0000" domi="100.0"/>
      </shape>\n''')
 if speaker == 'adult':
  file.write('''      <shape name="initial">
        <param name="HX" value="0.6259" domi="100.0"/>
        <param name="HY" value="-4.8156" domi="100.0"/>
        <param name="JX" value="0.0000" domi="100.0"/>
        <param name="JA" value="-4.6286" domi="100.0"/>
        <param name="LP" value="0.1203" domi="100.0"/>
        <param name="LD" value="0.6552" domi="100.0"/>
        <param name="VS" value="0.5709" domi="100.0"/>
        <param name="VO" value="-0.1000" domi="100.0"/>
        <param name="WC" value="0.0000" domi="100.0"/>
        <param name="TCX" value="1.3037" domi="100.0"/>
        <param name="TCY" value="-1.9642" domi="100.0"/>
        <param name="TTX" value="4.8069" domi="100.0"/>
        <param name="TTY" value="-1.0190" domi="100.0"/>
        <param name="TBX" value="3.3425" domi="100.0"/>
        <param name="TBY" value="-0.3516" domi="100.0"/>
        <param name="TRX" value="-1.7212" domi="100.0"/>
        <param name="TRY" value="-1.9642" domi="100.0"/>
        <param name="TS1" value="0.7520" domi="100.0"/>
        <param name="TS2" value="0.6960" domi="100.0"/>
        <param name="TS3" value="0.9000" domi="100.0"/>
        <param name="TS4" value="0.2560" domi="100.0"/>
        <param name="MA1" value="0.0000" domi="100.0"/>
        <param name="MA2" value="0.0000" domi="100.0"/>
        <param name="MA3" value="0.0000" domi="100.0"/>
      </shape>\n''')
 file.write('''      <shape name="input">
        <param name="HX" value="{HX}" domi="100.0"/>
        <param name="HY" value="{HY}" domi="100.0"/>\n'''.format(**dic))
 if speaker == 'infant':
    file.write('        <param name="JX" value="-0.1488" domi="100.0"/>\n')
 else:
    file.write('        <param name="JX" value="-0.2000" domi="100.0"/>\n')
 file.write('''        <param name="JA" value="{JA}" domi="100.0"/>
        <param name="LP" value="{LP}" domi="100.0"/>
        <param name="LD" value="{LD}" domi="100.0"/>
        <param name="VS" value="{VS}" domi="100.0"/>
        <param name="VO" value="-0.1000" domi="100.0"/>
        <param name="WC" value="0.0000" domi="100.0"/>
        <param name="TCX" value="{TCX}" domi="100.0"/>
        <param name="TCY" value="{TCY}" domi="100.0"/>
        <param name="TTX" value="{TTX}" domi="100.0"/>
        <param name="TTY" value="{TTY}" domi="100.0"/>
        <param name="TBX" value="{TBX}" domi="100.0"/>
        <param name="TBY" value="{TBY}" domi="100.0"/>
        <param name="TRX" value="-1.8530" domi="100.0"/>
        <param name="TRY" value="-1.7267" domi="100.0"/>
        <param name="TS1" value="{TS1}" domi="100.0"/>
        <param name="TS2" value="{TS2}" domi="100.0"/>
        <param name="TS3" value="{TS3}" domi="100.0"/>
        <param name="TS4" value="{TS4}" domi="100.0"/>
        <param name="MA1" value="0.0000" domi="100.0"/>
        <param name="MA2" value="0.0000" domi="100.0"/>
        <param name="MA3" value="0.0000" domi="100.0"/>
      </shape>\n'''.format(**dic))
 file.write('    </shapes>\n')
 file.write('  </vocal_tract_model>\n')

 ############################
 # Glottis model (Titze only)
 ############################
 file.write('  <glottis_models>\n')
 file.write('''    <glottis_model type="Titze" selected="1">
      <static_params>
        <param index="0" name="Cord rest thickness" abbr="rest_thickness" unit="m" min="0.003000" max="0.010000" default="0.004500" value="0.004500"/>
        <param index="1" name="Cord rest length" abbr="rest_length" unit="m" min="0.005000" max="0.020000" default="0.016000" value="0.016000"/>
        <param index="2" name="Chink length" abbr="chink_length" unit="m" min="0.001000" max="0.005000" default="0.002000" value="0.002000"/>
      </static_params>
      <control_params>
        <param index="0" name="f0" abbr="f0" unit="Hz" min="40.000000" max="600.000000" default="120.000000" value="120.000000"/>
        <param index="1" name="Subglottal pressure" abbr="pressure" unit="Pa" min="0.000000" max="2000.000000" default="800.000000" value="800.000000"/>
        <param index="2" name="Lower displacement" abbr="x_bottom" unit="m" min="-0.000500" max="0.003000" default="0.000300" value="0.000300"/>
        <param index="3" name="Upper displacement" abbr="x_top" unit="m" min="-0.000500" max="0.003000" default="0.000300" value="0.000300"/>
        <param index="4" name="Extra arytenoid area" abbr="ary_area" unit="m^2" min="-0.000025" max="0.000025" default="0.000000" value="0.000000"/>
        <param index="5" name="Phase lag" abbr="lag" unit="rad" min="0.000000" max="3.141500" default="0.880000" value="0.880000"/>
        <param index="6" name="Aspiration strength" abbr="aspiration_strength" unit="dB" min="-35.000000" max="0.000000" default="-35.000000" value="-35.000000"/>
      </control_params>
      <shapes>
        <shape name="default">
          <control_param index="0" value="120.000000"/>
          <control_param index="1" value="800.000000"/>
          <control_param index="2" value="0.000300"/>
          <control_param index="3" value="0.000300"/>
          <control_param index="4" value="0.000000"/>
          <control_param index="5" value="0.880000"/>
          <control_param index="6" value="-35.000000"/>
        </shape>
        <shape name="open">
          <control_param index="0" value="77.760630"/>
          <control_param index="1" value="869.190299"/>
          <control_param index="2" value="0.001002"/>
          <control_param index="3" value="0.000998"/>
          <control_param index="4" value="0.000000"/>
          <control_param index="5" value="0.880000"/>
          <control_param index="6" value="0.000000"/>
        </shape>
        <shape name="modal">
          <control_param index="0" value="104.439484"/>
          <control_param index="1" value="999.999997"/>
          <control_param index="2" value="0.000151"/>
          <control_param index="3" value="0.000151"/>
          <control_param index="4" value="0.000000"/>
          <control_param index="5" value="0.885903"/>
          <control_param index="6" value="-35.000000"/>
        </shape>
        <shape name="breathy">
          <control_param index="0" value="120.000000"/>
          <control_param index="1" value="800.000000"/>
          <control_param index="2" value="0.000497"/>
          <control_param index="3" value="0.000504"/>
          <control_param index="4" value="0.000000"/>
          <control_param index="5" value="0.885903"/>
          <control_param index="6" value="-20.790000"/>
        </shape>
        <shape name="pressed">
          <control_param index="0" value="104.599174"/>
          <control_param index="1" value="999.999998"/>
          <control_param index="2" value="0.000004"/>
          <control_param index="3" value="0.000000"/>
          <control_param index="4" value="0.000000"/>
          <control_param index="5" value="0.885903"/>
          <control_param index="6" value="-35.000000"/>
        </shape>
        <shape name="stop">
          <control_param index="0" value="120.000000"/>
          <control_param index="1" value="800.000000"/>
          <control_param index="2" value="-0.000300"/>
          <control_param index="3" value="-0.000300"/>
          <control_param index="4" value="0.000000"/>
          <control_param index="5" value="0.885903"/>
          <control_param index="6" value="-35.000000"/>
        </shape>
        <shape name="slightly-breathy">
          <control_param index="0" value="104.439484"/>
          <control_param index="1" value="999.999997"/>
          <control_param index="2" value="0.000249"/>
          <control_param index="3" value="0.000253"/>
          <control_param index="4" value="0.000000"/>
          <control_param index="5" value="0.885903"/>
          <control_param index="6" value="-35.000000"/>
        </shape>
        <shape name="fully-open">
          <control_param index="0" value="77.760630"/>
          <control_param index="1" value="869.190299"/>
          <control_param index="2" value="0.001999"/>
          <control_param index="3" value="0.001995"/>
          <control_param index="4" value="0.000000"/>
          <control_param index="5" value="0.880000"/>
          <control_param index="6" value="0.000000"/>
        </shape>
        <shape name="half-open">
          <control_param index="0" value="107.356872"/>
          <control_param index="1" value="1000.000000"/>
          <control_param index="2" value="0.000998"/>
          <control_param index="3" value="0.000998"/>
          <control_param index="4" value="0.000000"/>
          <control_param index="5" value="0.885903"/>
          <control_param index="6" value="-20.790000"/>
        </shape>
        <shape name="slightly-pressed">
          <control_param index="0" value="103.945487"/>
          <control_param index="1" value="999.999829"/>
          <control_param index="2" value="0.000053"/>
          <control_param index="3" value="0.000053"/>
          <control_param index="4" value="0.000000"/>
          <control_param index="5" value="0.885903"/>
          <control_param index="6" value="-35.000000"/>
        </shape>
        <shape name="open75">
          <control_param index="0" value="103.826174"/>
          <control_param index="1" value="999.421139"/>
          <control_param index="2" value="0.000746"/>
          <control_param index="3" value="0.000746"/>
          <control_param index="4" value="0.000000"/>
          <control_param index="5" value="0.880000"/>
          <control_param index="6" value="0.000000"/>
        </shape>
      </shapes>
    </glottis_model>\n''')

 file.write('    <glottis_model type="Two-mass model" selected="0">\n')
 file.write('    </glottis_model>\n')
 file.write('    <glottis_model type="Triangular glottis" selected="0">\n')
 file.write('    </glottis_model>\n')
 file.write('    <glottis_model type="Four-mass model" selected="0">\n')
 file.write('    </glottis_model>\n')

 file.write('  </glottis_models>\n')
 file.write('</speaker>')


 if verbose:
  print 'closing speaker file'
 file.close()

 return filename_


# Example

#par = [1.0, -2.0, 0.0, -2.0, -0.07, 0.5, 0.0, -0.1, -0.4, -0.7, 2.0, -0.6, 1.1, 0.3, 0.0, 0.05, 0.1, 0.1]
#create_speaker('test', par)
