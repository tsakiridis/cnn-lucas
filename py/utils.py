from enum import Enum

import pandas
import numpy as np
from scipy.stats import boxcox

from reader_Rdata import read_Rdata
from preprocess import calculate_preprocessing


# Input standardization

class InputStandardization(str, Enum):
    NONE = "no"
    PER_BAND = "per-band"
    STANDARD_SCORE = "standard"

    def __str__(self):
        return self.value


def standardizePerBand(data):
    for i in range(0, len(data[0])):
        data[:, i] = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])
    return data


def standardizeFull(data):
    return (data - np.mean(data)) / np.std(data)


# Zero padding definitions

def zeroPadding(data, size):
    zeros = np.zeros(len(data))
    zeros = zeros.reshape(len(data), 1)

    while len(data[0]) != size:
        data = np.append(data, zeros, axis=1)
    return data


# Output normalizations

def normalizeOutput(output, maxNorm=1):
    return (maxNorm * 2) * ((output - min(output)) / float(max(output) - min(output))) - maxNorm


def unnormalizeOutput(output, minOut, maxOut, maxNorm=1):
    return ((output + maxNorm) * float(maxOut - minOut)) / (2 * maxNorm) + minOut


class Data:
    def __init__(
            self,
            standardizeInput,
            output='OC',
            maxNormalization=1):
        self.properties = ["OC", "clay", "silt", "sand", "N", "P", "K", "CEC", "CaCO3", "pH.in.H2O"]
        self.multipleOutput = output == "all"
        if not self.multipleOutput:
            assert output in self.properties

        # Path to .Rdata is stored in the config.json
        config = pandas.read_json("../config.json", typ="series")
        self.df_properties, self.df_spectra = read_Rdata(config.LUCAS_RDATA)
        # Get only the samples specified in the data_partition file
        self.split = pandas.read_csv(config.DATA_PARTITION)

        # First deal with the spectra
        self.df_spectra = self.df_spectra.loc[self.split.ID]
        self.spectra = calculate_preprocessing(self.df_spectra)
        if standardizeInput == InputStandardization.PER_BAND:
            for key in ["Absorbances", "Absorbances-SG1", "Absorbances-SNV-DT", "CR"]:
                self.spectra[key] = standardizePerBand(self.spectra[key])
        elif standardizeInput == InputStandardization.STANDARD_SCORE:
            for key in self.spectra.keys():
                self.spectra[key] = standardizeFull(self.spectra[key])

        for key in ["Absorbances-SG0-SNV", "Absorbances-SG1", "Absorbances-SG1-SNV", "CR"]:
            self.spectra[key] = zeroPadding(
                self.spectra[key], len(self.spectra["Absorbances"][0]))

        # Now deal with the output
        if self.multipleOutput:
            self.out_original = {}
            self.output = {}
            for key in self.properties:
                self.out_original[key] = self.df_properties.loc[self.split.ID, key].to_numpy()
                if key == "P":
                    # Box-Cox defined for positive values. We'll take the limit as y approaches 0.
                    self.out_original[key] = np.where(self.out_original[key]==0, 1e-10, self.out_original[key]) 
                    self.out_original[key] = boxcox(self.out_original[key], 0.4)
                elif key == "K":
                    # Box-Cox defined for positive values. We'll take the limit as y approaches 0.
                    self.out_original[key] = np.where(self.out_original[key]==0, 1e-10, self.out_original[key]) 
                    self.out_original[key] = boxcox(self.out_original[key], 0.3)
                self.output[key] = normalizeOutput(self.out_original[key], maxNormalization)
        else:
            self.out_original = self.df_properties.loc[self.split.ID, output].to_numpy()
            self.output = normalizeOutput(self.out_original, maxNormalization)

    # For single input data

    def get_trn_val(self, f, key):
        assert key in self.spectra.keys()
        assert f in list(set(self.split.fold)) and f != -1

        trn = self.split.index[(self.split.fold != -1) & (self.split.fold != f)].to_numpy()
        val = self.split.index[self.split.fold == f].to_numpy()

        trnX = self.spectra[key][trn, :]
        valX = self.spectra[key][val, :]

        trnX = trnX.reshape((trnX.shape[0], trnX.shape[1], 1))
        valX = valX.reshape((valX.shape[0], valX.shape[1], 1))

        if self.multipleOutput:
            trnY = self.createOutputData(trn)
            valY = self.createOutputData(val)
        else:
            trnY = self.output[trn]
            valY = self.output[val]

        return trnX, trnY, valX, valY

    def get_tst(self, key):
        assert not self.multipleOutput
        assert key in self.spectra.keys()

        tst = self.split.index[self.split.split == "test"]

        tstX = self.spectra[key][tst, :]
        tstX = tstX.reshape((tstX.shape[0], tstX.shape[1], 1))

        tstY = self.createOutputData(tst) if self.multipleOutput else self.output[tst]

        return tstX, tstY

    # For multiple input data

    def get_m_trn_val(self, f):
        assert f in list(set(self.split.fold)) and f != -1

        trn = self.split.index[(self.split.fold != -1) & (self.split.fold != f)].to_numpy()
        val = self.split.index[self.split.fold == f].to_numpy()

        trnX = np.zeros((len(trn), len(self.spectra["Absorbances"][0]), 6))
        for i, key in enumerate(self.spectra.keys()):
            trnX[:, :, i] = self.spectra[key][trn]

        valX = np.zeros((len(val), len(self.spectra["Absorbances"][0]), 6))
        for i, key in enumerate(self.spectra.keys()):
            valX[:, :, i] = self.spectra[key][val]

        if self.multipleOutput:
            trnY = self.createOutputData(trn)
            valY = self.createOutputData(val)
        else:
            trnY = self.output[trn]
            valY = self.output[val]

        return trnX, trnY, valX, valY

    def get_m_tst(self):
        tst = self.split.index[self.split.split == "test"]

        tstX = np.zeros((len(tst), len(self.spectra["Absorbances"][0]), 6))
        for i, key in enumerate(self.spectra.keys()):
            tstX[:, :, i] = self.spectra[key][tst]

        tstY = self.createOutputData(tst) if self.multipleOutput else self.output[tst]

        return tstX, tstY

    def createOutputData(self, ind):
        y = np.zeros(shape=(len(self.output), len(ind)))
        for i, key in enumerate(self.output):
            y[i, :] = [self.output[key][j] for j in ind]
        return y.transpose()

    # Unnormalize output

    def unnormalizeOutput(self, y, output, maxNorm):
        if self.multipleOutput:
            minY, maxY = min(self.out_original[output]), max(self.out_original[output])
        else:
            minY, maxY = min(self.out_original), max(self.out_original)
        return unnormalizeOutput(y, minY, maxY, maxNorm)

