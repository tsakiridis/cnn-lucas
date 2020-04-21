A multi-channel 1-D Convolutional Neural Network to predict the soil properties of the Land Use / Cover Area Frame Survey of the European Union from VNIR--SWIR spectra
=======================================================================================================================================================================

# Preliminaries

This repository contains a modified version of the python code which was used in the paper ["Simultaneous prediction of soil properties from VNIR--SWIR spectra using a localized multi-channel 1-D convolutional neural network"](https://dx.doi.org/10.1016/j.geoderma.2020.114208).

The CNNs proposed therein were applied to the [LUCAS 2009 topsoil database](https://esdac.jrc.ec.europa.eu/content/lucas-2009-topsoil-data), to predict the soil properties from the VNIR--SWIR (400--2500 nm) spectra. They are the following ones:

* **single input -- single output** where a single spectral source is used to predict a single soil property (called **1-1** in the paper)
* **multiple input -- single output** where a combination of spectral sources (initial + spectral pre-treatments) is used to predict a single soil property (called **m-1** in the paper)
* **single input -- multiple output** where a single spectral source is used to predict a single soil property (not examined in the paper)
* **multiple input -- multiple output** where a combination of spectral sources (initial + spectral pre-treatments) is used to predict multiple soil properties *simultaneously* (called **m-m** in the paper)

You will have to download the LUCAS data from the ESDAC platform after filling the request form. In the download data, you will find an *.Rdata file containing the soil spectral library.

In the original version that we used for the paper, the code was reading data from a local folder --- we had pre-calculated all spectral pre-treatments using R and the [prospectr](https://cran.r-project.org/web/packages/prospectr/index.html) package. However, as the re-distribution of LUCAS is not allowed, in this python version the .Rdata is loaded and then the spectral pre-treatments are calculated on the fly, which incurs a slight computational cost each time you call the model, as data are exchanged back-and-forth between R and python.

The code is written in python. To load the .Rdata file and to calculate the pre-treatments, R will be called via python. This will be done with the [rpy2](https://rpy2.github.io/doc/latest/html/introduction.html) package.

Finally, in addition to the code, the splits used to build the models are also provided in the folder data.


# Installation

I strongly recommend using [anaconda](https://www.anaconda.com/). Go ahead and download & install it, and we'll create a new environment for all the packages that are needed.

Obviously, you are probably already planning on running this with a GPU, so I'll assume that you've installed the drivers for your graphics card.

We'll create a new environment (named tf_gpu) to install tensorflow-gpu and keras.
```sh
conda create --name tf_gpu tensorflow-gpu
conda activate tf_gpu
conda install -c conda-forge keras scikit-learn
```

Next we'll install some helpful python packages:
```sh
conda install pandas numpy
```

I'll also install ipykernel which allows me to use a jupyter notebook from my base environment, where the package [nb_conda_kernels](https://anaconda.org/conda-forge/nb_conda_kernels) is installed.
You may skip this if you don't plan on using jupyter, but keep in mind that a couple jupyter notebook are provided in this repository to help you get started.
```sh
conda install -c anaconda ipykernel 
```

Bear in mind that [pyreadr](https://github.com/ofajardo/pyreadr), a package that doesn't require R to load an .Rdata file, can't load the LUCAS file --- so we'll have to install R and rpy2 to load the file. Currently, anaconda supplies v2.9.4 of rpy2 which is not the latest, and doesn't properly identify tzlocal as a dependency.
```sh
conda install -c conda-forge tzlocal 
conda install -c r r rpy2 r-prospectr
```

Also, if you want to visualize the data we'll need a plotting library:
```sh
conda install -c conda-forge matplotlib 
conda install -c anaconda seaborn
```

Now, [clone this repository](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository), and edit the config.json file using your preferred text editor to select the path to the .Rdata file you downloaded from LUCAS, as well as the path to the data folder (of this repository) where the data split is stored.


# Sanity check

* Make sure you've edited the config.json to include the path to the .Rdata you've downloaded from ESDAC
* Run the read\_Rdata\_with\_rpy2 notebook to ensure that you can transform the data into a pandas dataframe. This is a crucial step as this confirms that the R <-> python interface works. 
* You can also inspect the data splits using the visualize-split notebook
* Activate the tf_gpu conda environment and ensure that keras can be loaded (e.g. import the package via ipython)


# Run 

Open up a terminal and activate the conda environment we are using:
```sh
conda activate tf_gpu
```
and cd to the directory where you cloned this repo.

You can run the models through the command line, and most of the parameters can be passed as arguments via the command line. To see a list of available arguments, simply type:
```sh
python cnn.py --help
```

The command takes a positional argument which is the path to where you want the results to be stored. By default, both the model (.hdf5) and the predictions in the independent test set (*.csv) are stored in subfolders under the specified folder; you'll also find a small json file containing all arguments used to produce the models for future reference. The error metrics (RMSE, Rsquared, and RPIQ) are outputted in the terminal, but not stored anywhere; you can easily re-calculate them from the csv files and using the functions defined in metrics.py.

To run e.g. the **m-m** model described in the paper, you need to call it as follows:
```sh
python cnn.py -d ~/cnn-results -i all -o all -f 32 64 -fc 100 40
```
where the first argument is the path to the results, the following specify to use all available inputs and outputs, the penultimate argument controls the number of convolutional filters, while the final argument the number of units of the two dense layers.

Some additional examples are provided in the shell script (script.sh).


# Acknowledgments

The initial version of the code was written almost entirely by K. Keramaris, and was cleaned and updated by me.

The LUCAS topsoil dataset used in this work was made available by the European Commission through the European Soil Data Centre managed by the Joint Research Centre (JRC), [http://esdac.jrc.ec.europa.eu/](http://esdac.jrc.ec.europa.eu/).
