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

The code uses PyMC3 which is still in Alpha, but has some great syntax and 
processing benefits over v2.3. You can `pip install` it via:

    pip install git+https://github.com/pymc-devs/pymc


To use/exit the Anaconda environment:
    
    source activate suas
    source deactivate

