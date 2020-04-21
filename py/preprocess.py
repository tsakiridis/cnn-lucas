import rpy2

from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

import numpy as np


def calculate_preprocessing(spc_df):
    
    numpy2ri.activate()
    base = importr('base')
    utils = importr('utils')
    prospectr = importr('prospectr')

    # Ignore first 200 * 0.5 = 100 nm, pick every 20 * 0.5 = 10 nm
    subsample = list(range(200, spc_df.shape[1], 20)) 
    # This can be used only if the original spectral wavelengths are retained,
    # i.e. not for SG*-based spectra
    
    # SG0
    sg0 = np.array(
        prospectr.savitzkyGolay(spc_df.to_numpy(), m = 0, w = 101, p = 3))
    
    # SG1
    sg1 = np.array(
        prospectr.savitzkyGolay(spc_df.to_numpy(), m = 1, w = 101, p = 3))
    
    # Common for both SG filters because they use the same width
    sg_subsample = list(range(150, sg1.shape[1], 20))
    
    return {
        "Absorbances": spc_df.iloc[:, subsample].to_numpy(),
        "Absorbances-SG0-SNV": np.array(
            prospectr.standardNormalVariate(sg0))[:, sg_subsample],
        "Absorbances-SG1": sg1[:, sg_subsample],
        "Absorbances-SG1-SNV": np.array(
            prospectr.standardNormalVariate(sg1))[:, sg_subsample],
        "CR": np.array(
            prospectr.continuumRemoval(spc_df.to_numpy(), type="A"))[:, subsample],
        "Absorbances-SNV-DT": np.array(
            prospectr.detrend(spc_df.to_numpy(), 
                              wav=rpy2.robjects.FloatVector(spc_df.columns.astype('float').to_numpy())))[:, subsample]
    }