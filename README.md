# SCORE

The Shape COnstraint REstoration algorithm ([SCORE](#SCORE)) is a proximal algorithm based on sparsity and shape constraints to restore images.

- [Getting Started](#Getting-Started)
  * [Prerequisites](###Prerequisites)
  * [Installing](###Installing)
- [Running the examples](##Running-the-examples)
  * [Example 1](###Example-1)
  * [Example 2](###Example-2)
- [Reproducible Research](##Reproducible-Research)
- [Authors](##Authors)
- [License](##License)
- [Acknowledgments](##Acknowledgments)

## Getting Started


### Prerequisites


These instructions will get you a copy of the project up and running on your local machine. One easy way to install the prerequisites is using Anaconda. To install Anaconda see : https://docs.conda.io/projects/conda/en/latest/user-guide/install/

* Numpy

```sh
conda install -c anaconda numpy
```
* Scipy

```sh
conda install -c anaconda scipy
```

* Skimage

```sh
conda install -c conda-forge scikit-image
```

* α-shearlet Transform

&nbsp;&nbsp;&nbsp;&nbsp;Clone or download the library using the following link : 
https://github.com/dedale-fet/alpha-transform

&nbsp;&nbsp;&nbsp;&nbsp;Add the path of the α-shearlet Transform library to the PYTHONPATH variable in the bash profile

```sh
export PYTHONPATH="$HOME/path/to/alpha-transform-master:$PYTHONPATH"
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_Replace `path/to` by the corresponding path_

* GalSim [optional] (for research reproducibility)

```sh
conda install -c conda-forge galsim 
```

* Matplotlib [optional]

```sh
conda install -c conda-forge matplotlib
```

### Installing

After install the prerequisites, clone or download `score` repository. And to be able to access from any working directory, use the following command to add the path to `score` to PYTHONPATH variable in the bash profile :

```sh
export PYTHONPATH="$HOME/path/to/alpha-transform-master:$PYTHONPATH"
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_Replace `path/to` by the corresponding path_

## Running the examples

This repository contains two examples. They respectively illustrate a denoising and a deconvolution case.

### Example 1

In this simple denoising case, we restore a galaxy image corrupted by noise. The core of the code is:

```python
#instantiate score and, for example, set the value of gamma the other parameters will take their default values
denoiser = score(gamma=0.5)
#denoise
denoiser.denoise(obs=gal_obs) #the result will be in denoiser.solution
```

### Example 2

In this deconvolution case, we compare the score algorithm with a value of γ = 1 (which is close to its optimal computed value) and the Sparse Restoration Algorithm (γ = 0 and no Removal of Isolated Pixels). We loop on a stack of galaxy images and perfom both deconvolution operation on each image:

```python
#loop
for obs, psf, gt in zip(gals_obs,psfs,gals):
    #deconvolve
    g1.deconvolve(obs=obs,ground_truth=gt,psf=psf)
    g0.deconvolve(obs=obs,ground_truth=gt,psf=psf)
    #update ellipticity error lists
    g1_error_list += [g1.relative_ell_error]
    g0_error_list += [g0.relative_ell_error]
```

## Reproducible Research


The code `generate_dataset.py` allows to recreate the exactly the same dataset used for the numerical experiments of the original paper[INSERT LINK] of `score`. +ADD INFO ABOUT COSMOS CATALOG AND PARAMETER CHOICE.

## Authors 
## [SECTION UNFINISHED]

* [**Fadi Nammour**](http://www.cosmostat.org/people/fadi-nammour) - *Initial work* -
* [**Morgan Schmitz**](http://www.cosmostat.org/people/mschmitz) - *Initial work* -

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments 
## [SECTION UNFINISHED]

* Samuel Farrens
* Ming
* Felix Voigtlaender for alphaShearlet library
* etc
