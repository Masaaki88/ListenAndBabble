# ListenAndBabble
Attention: This repository is still work in progress and contains largely undocumented code.

This repository contains the code for the model published in 'Murakami et al. (2015): "Seeing [u] aids vocal learning: babbling and imitation of vowels using a 3D vocal tract model, reinforcement learning, and reservoir computing." International Conference on Development and Learning and on Epigenetic Robotics 2015 (in press).'

The code is written in Python 2.7 and was tested in Ubuntu 14.04. Apart from standard Python libraries, it requires:
- [numpy](http://sourceforge.net/projects/numpy/files/NumPy/)
- [scipy](http://sourceforge.net/projects/scipy/files/scipy/)
- [matplotlib](http://matplotlib.org/downloads.html)
- [brian](http://brian.readthedocs.org/en/latest/installation.html)
- [Oger](http://reservoir-computing.org/installing_oger)
- [mpi4py](https://pypi.python.org/pypi/mpi4py)

Additionally, you'll need the VocalTractLab API for Linux, which you can download [here](http://vocaltractlab.de/index.php?page=vocaltractlab-download).
After downloading it, extract the content into the VTL_API subfolder.

example.py provides a step-by-step tutorial of how to use the code.
