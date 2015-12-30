import os



"""
This tutorial will walk you through the basic steps of the model.
Here we simulate an infant speaker who attempts to imitate the
 three vowels [a], [i], and [u], which she hears in her environment.
To make this example particularly easy and fast, she has to figure
 out only the horizontal position of her tongue center.
This file is a self-consistent code; you can simply run it as a
 regular python script, and it will perform all steps from the
 beginning to the end.
    -- Max Murakami 15/12/30
"""



#######################################
#
# Step 1: Generate Ambient Language Samples
#
#######################################



    # The folders data/[vowel] are accessed during auditory learning.
    # We need to fill them with samples of the corresponding vowel.
    # Let's delete them and create empty ones to fill.
os.system('rm data/a -r')
os.system('rm data/i -r')
os.system('rm data/u -r')
os.system('mkdir data/a')
os.system('mkdir data/i')
os.system('mkdir data/u')

    # The folders data/temp/[vowel] will be filled with newly created samples.
    # Let's delete them to start fresh.
os.system('rm data/temp -r')

    # Now we start generating samples.
    # For each vowel, we'll create 6 samples with standard deviation 0.01 from 
    #  the vowel prototypes, i.e. samples that shouldn't deviate much from the
    #  prototypes.
    # So we expect those samples to represent the vowels fairly well.
os.system('python generatedata.py a --n_samples 6 --sigma 0.01')
os.system('python generatedata.py i --n_samples 6 --sigma 0.01')
os.system('python generatedata.py u --n_samples 6 --sigma 0.01')

    # Now we treat all of the generated samples as representations of the 
    #  corresponding vowels and move them into the respective data vowel folders.
    # This is a simplification for the sake of this tutorial, though!
    #
    # IMPORTANT: During actual experiments, listen to each of the samples and
    #  catagorize them individually! Exclude those samples that don't match any
    #  of the target vowels and save them for later (see below).
os.system('mv data/temp/a/* data/a/')
os.system('mv data/temp/i/* data/i/')
os.system('mv data/temp/u/* data/u/')

    # Now we do the same for the ambient infant speaker.
os.system('python generatedata.py a --n_samples 6 --infant --sigma 0.01')
os.system('python generatedata.py i --n_samples 6 --infant --sigma 0.01')
os.system('python generatedata.py u --n_samples 6 --infant --sigma 0.01')

    # Here we first rename the samples so we don't overwrite the adult samples
    #  in the data vowel folders.
    #
    # -> During experiments, categorize these vowels individually too.
os.system("rename 's/\./_in\./' data/temp/a/*")
os.system('mv data/temp/a/* data/a/')
os.system("rename 's/\./_in\./' data/temp/i/*")
os.system('mv data/temp/i/* data/i/')
os.system("rename 's/\./_in\./' data/temp/u/*")
os.system('mv data/temp/u/* data/u/')

    # Now comes the generation of the null samples.
    # The auditory system needs to learn that there are sounds that are none of
    #  of those vowels that we'd like it to recognize.
    # These sound samples we will call null samples.
    # In our case, the auditory system should learn to recognize the target vowels 
    #  [a], [i], and [u], so the speaker can judge how well she imitates those.
    # We also want the auditory system to recognize that [o], for example, doesn't
    #  match either of the target vowels, so the speaker won't end up producing
    #  [o]s when she actually wants to imitate [u].
    # Samples of [o] would be null samples, then.
    # So first off, we'll create fresh folders for the null samples.
os.system('rm data/null_a -r')
os.system('rm data/null_i -r')
os.system('rm data/null_u -r')
os.system('mkdir data/null_a')
os.system('mkdir data/null_i')
os.system('mkdir data/null_u')

    # For each of the target vowels, we'll create 6 samples with standard deviation
    #  0.1 from the prototypes.
    # Sigma is much larger than in the previous case, so we expect these samples
    #  to be less similar to the target vowels.
os.system('python generatedata.py a --n_samples 6 --sigma 0.1')
os.system('python generatedata.py i --n_samples 6 --sigma 0.1')
os.system('python generatedata.py u --n_samples 6 --sigma 0.1')

    # Now we treat all of the generated samples as null samples and move them 
    #  into the respective data folders.
    # Again, this is a simplification for the sake of this tutorial!
    #
    # IMPORTANT: During actual experiments, listen to each of the samples and
    #  catagorize them individually! These samples will include ones that
    #  actually do represent the target vowels. Put those into the regular data
    #  folders.
os.system('mv data/temp/a/* data/null_a/')
os.system('mv data/temp/i/* data/null_i/')
os.system('mv data/temp/u/* data/null_u/')

    # Finally, the ambient infant speaker also gets to produce null samples.
os.system('python generatedata.py a --n_samples 6 --infant --sigma 0.1')
os.system('python generatedata.py i --n_samples 6 --infant --sigma 0.1')
os.system('python generatedata.py u --n_samples 6 --infant --sigma 0.1')

    # Rename and move the infant null samples
    #
    # -> During experiments, categorize these vowels individually too.
os.system("rename 's/\./_in\./' data/temp/a/*")
os.system('mv data/temp/a/* data/null_a/')
os.system("rename 's/\./_in\./' data/temp/i/*")
os.system('mv data/temp/i/* data/null_i/')
os.system("rename 's/\./_in\./' data/temp/u/*")
os.system('mv data/temp/u/* data/null_u/')





#######################################
#
# Step 2: Let the Auditory System Learn
#
#######################################



    # With the ambient speech samples in place, we can now train the auditory
    #  system.
    # For that, we're specifying the number of target vowels (3), the number
    #  of samples per vowel (12) and the number of samples per vowel that we're
    #  using for training the auditory system (let's say 9). 
    # The leftover samples will then be used for testing the classification 
    #  performance of the auditory system.
    # Also, we're specifying the reservoir size (1000) and the subfolder where
    #  the results will be stored (here: /output/[date-no.]/tutorial).
os.system('python learndata.py 3 --n_samples 12 --n_training 9 --n_reservoirs 1000 --subfolder tutorial')
    # The trained auditory system will then be stored as current_auditory_system.flow
    #  in the data folder.





#######################################
#
# Step 3: Babble Away
#
#######################################



    # Given the trained auditory system, we can finally let our speaker babble.
    # In this tutorial, the speaker has to figure out only the horizontal
    #  position of her tongue center (TCX).
    # This babbling will likely take quite some time (and may in fact take
    #  infinitely long, see below).
    # You can observe the learning progress in the output/[date]/data folder,
    #  where you will find the sound files of the current (and previous
    #  successful) articulations as well as their auditory system responses
    #  to judge their quality.
os.system('python rl_agent_mpi.py --parameters TCX --default_settings')





#######################################
#
# Comments
#
#######################################



    # Even this simple example may take a very long time to finish, or it may
    #  never stop.
    # The most critical factor for this is the quality of the auditory learning.
    # So your most important task is to make sure the auditory system is trained
    #  properly:
    #   - Make sure that every single ambient speech sample is placed in the
    #      correct folder. If it sounds like [a] for you, put it in data/a and
    #      so on. If it doesn't sound like any of the target vowels, put it into
    #      one of the null folders. All null folders are treated in the same way,
    #      so it doesn't exactly make a big difference where you put it. 
    #      The idea behind the different null folders is this: If we generate 
    #      speech samples in the vicinity of a given prototypical vowel, then 
    #      null samples may show up that don't really sound like that vowel but 
    #      show some similarity to it. So in a sense, they represent constraints 
    #      of the acoustical properties of that vowel, which greatly helps the 
    #      auditory system to learn a model of that vowel.
    #      In the ideal case, all null folders contain the same number of samples,
    #      which should be a third of what each vowel folder holds. The reason
    #      is that we'd like to train the auditory system in an unbiased fashion
    #      such that the trained auditory system shows no a priori classification
    #      preference for any class. Use these rules to achieve this:
    #       n_samples is a multiple of the number of vowels,
    #       n_training is a multiple of the number of vowels,
    #       each vowel folder contains at least n_samples samples,
    #       each null folder contains at least n_samples/3 samples.
    #      An alternative is to bias auditory learning in favor of the null class.
    #      This will have the effect that the trained auditory system is more
    #      likely to classify a given sample as a null sample, so the speech
    #      sample needs to provide stronger evidence that it is one the target
    #      vowels. So introducing such a training bias creates "stricter" auditory
    #      systems, which is viable.
    #   - Increase the number of ambient speech samples for auditory learning. The
    #      number of samples that we generated in this example is way too small for
    #      efficient learning. Raise that number by at least one order of magnitude
    #      to see reasonable learning progress.
    # Other ways to improve/accelerate learning:
    #   - Make use of parallelization. Especially when you're moving to problems that
    #      involve more than one articulator, being able to crank out tens or hundreds
    #      of speech samples per babbling iteration is a huge advantage and makes
    #      these problems feasible in the first place.
    #   - Run statistics. Because the reservoir of the auditory system is based on
    #      random numbers, you can always end up with one that has trouble recognizing
    #      one or the other vowel, even if your training parameters are good. Train
    #      multiple auditory systems and pick one that performs well.
    #   - Lower the reward threshold during babbling. Setting the reward threshold 
    #      to 0.5 is rather ad hoc. If you find that speech samples with 0.47 are
    #      consistently good enough imitations, don't hesitate to lower that threshold.
    #      This will make speech evaluation more lenient and the whole imitation
    #      process much faster.
