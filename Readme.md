# Beetle Classifier Robot

- *Author* Naor Scheinowitz
- *Version* 0.6
- *Last edit* 2021-11-29

## Introduction

The goal of this project is to create a physical measurement setup and accompanying Python code that characterizes the optical properties of *Coleoptera*. Specifically, we are interested in the polarization and luminosity of reflected light from a beetles shell. This project was inspired by the mind-boggling fact that the [Protaetia speciosa jousselini](https://en.wikipedia.org/wiki/Protaetia_speciosa) almost exclusively reflects left-handed circularly polarized light.
![Image taken from abovementioned link](https://upload.wikimedia.org/wikipedia/commons/0/04/Scarabaeidae_-_Protaetia_speciosa.JPG)

### Measurement procedure (rough outline)

1. Illuminate spot on beetle shell with collimated beam of light. We will use narrow-band color filters to create a polychromatic dataset.
2. Measure reflected light as function of angle with a CCD combined with polarization filters.

This will create a beetle-dataset of the form [beetle species, &lambda;, &theta;, S] or `[beetle species, wavelength, angle, Stokes vector]`.

#### Hardware list
- *Newport ESP300 motion controller*. Controls the angle-dependent movement of the CCD.
- *Mikropack Halogen Light Source HL-2000-FHSA*
- *ThorLabs DCC1645C*. Note: ThorLabs documentation refers to this class of camera as a 'DCx' camera, but it is equivalent to the classification 'UC480', which occasionally shows up in other documentation.

### Python code structure

The code is built up of one `class` per piece of computer-controllable hardware. The measurement procedure is done through the *Python Notebook* file `master.ipynb`, which combines the necessary functions into a sequential procedure that sends commands to the different devices. Documentation for each `class` is included in their files.

## Installation and setup with Anaconda

Unfortunately all of this only works on Windows due driver support from hardware manufacturers.

Start by installing [ThorCam](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam), the software used to interface with our CCD. Make sure to select *DCx USB* support during installation. The rest of the required packages can be installed using [Anaconda](https://www.anaconda.com/products/individual). For this project we create a new Anaconda environment and install all required packages there.
```
conda create --name BCR
conda activate BCR
conda install matplotlib jupyter pip h5py
conda install -c conda-forge pylablib
pip install pyvisa pyvisa-py
```
Navigate to this folder through your terminal of choice and start a *Jupyter Notebook* session. Next, open the file `master.ipynb`. From here, you can load all necessary modules and start giving commands. Further documentation and sample measurements are included in this file.


## Extra Documentation
| Package | Used For | Links |
| :------------- | :------------- |:------------- |
| PyVisa      | Sending commands to  and receiving status codes from the ESP300| [Docs](https://pyvisa.readthedocs.io/en/latest/introduction/index.html) |
| pylablib | Interfacing with the ThorLabs camera through Python | [Basics](https://pylablib.readthedocs.io/en/latest/devices/cameras_basics.html)[Docs](https://pylablib.readthedocs.io/en/latest/devices/uc480.html#cameras-uc480) [Commands](https://pylablib.readthedocs.io/en/latest/.apidoc/pylablib.devices.uc480.html) [VISA](https://pylablib.readthedocs.io/en/latest/devices/generic_awgs.html?highlight=VISA)
| ThorCam | Driver necessary for operating camera | [Driver](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam) |
| h5py | data storage interface HDF5 | [Docs](https://docs.h5py.org/en/latest/build.html)
