# Suas Literacy Intervention Analysis

Suas literacy analysis in Python

Contact: Adelaide Nic Chartaigh adelaide@suas.ie
Author: Jonathan Sedar jon.sedar@applied.ai
Date: Spring / Summer 2014


Basic analyses in Python, intending to use a PyMC approach to allow hierarchical bayesian regression


## Development

Assuming you're using Anaconda distro on OSX, you can simply create a new 
environment using the `requirements_conda.txt` file as a build recipe:


    conda create -n suas --file requirements_conda.txt

PyMC is not (yet) in the conda repos so you'll need to get that separately using:

    conda install -c https://conda.binstar.org/pymc pymc

And if you're using the latest gcc from homebrew, you may need to instead use

    conda install -c https://conda.binstar.org/jonsedar pymc

... which was built on OSX Mavericks 10.9.4 with homebrew gcc 4.8.3.


To use/exit the Anaconda environment:
    
    source activate suas
    source deactivate

