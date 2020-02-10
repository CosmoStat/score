# OSCAR
Observed Shape Constraint Applied to Restoration.
The Shearlet version of OSCAR relies on a slightly modified version of Felix
Voigtlaender's alphaShearlet library. The original can be found here:
https://github.com/dedale-fet/alpha-transform
# SCORE

The Shape COnstraint REstoration algorithm (SCORE) is a proximal algorithm based on sparsity and shape constraints to restore
images.

- [Getting Started](#Getting-Started)
  * [Sub-heading](#sub-heading)
    + [Sub-sub-heading](#sub-sub-heading)
- [Heading](#heading-1)
  * [Sub-heading](#sub-heading-1)
    + [Sub-sub-heading](#sub-sub-heading-1)
- [Heading](#heading-2)
  * [Sub-heading](#sub-heading-2)
    + [Sub-sub-heading](#sub-sub-heading-2)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

> One easy way to install the prerequisites is using Anaconda. To install Anaconda see : https://docs.conda.io/projects/conda/en/latest/user-guide/install/

* Numpy

```
conda install -c anaconda numpy
```
* Scipy

```
conda install -c anaconda scipy
```

* Skimage

```
conda install -c conda-forge scikit-image
```

* α-shearlet Transform

&nbsp;&nbsp;&nbsp;&nbsp;Clone or download the library using the following link : 
https://github.com/dedale-fet/alpha-transform

&nbsp;&nbsp;&nbsp;&nbsp;Add the path of the α-shearlet Transform library to the PYTHONPATH variable in the bash profile

```
export PYTHONPATH="$HOME/path/to/alpha-transform-master:$PYTHONPATH"
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_Replace `path/to` by the corresponding path_

* Matplotlib [optional]

```
conda install -c conda-forge matplotlib
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
