import pandas
import rpy2.robjects as robjects


""" Read the .Rdata from ESDAC and return the properties and spectra df """
def read_Rdata(path_to_file):
    # Load the .Rdata
    robjects.r['load'](path_to_file)
    Rdf = robjects.r('LUCAS.SOIL')
    # Identify the columns to extract for properties
    colnames = list(Rdf.colnames)
    cols_to_extract = [
        "sample.ID", "ID", "clay", "silt", "sand", "pH.in.H2O", "OC", 
        "N", "P", "K", "CEC", "CaCO3"
    ]
    cols_dict = {key: colnames.index(key) for key in cols_to_extract}
    # Generate the properties df
    df = pandas.DataFrame(
        {key: list(Rdf[cols_dict[key]]) for key in cols_to_extract}).set_index("ID")
    # And now generate the spectral df (hard column index 3 for "spc")
    # this may take a while!
    spc_df = pandas.DataFrame({key: list(Rdf[3][i]) for i, key in enumerate(list(Rdf[3].colnames))})
    spc_df["ID"] = df.index
    spc_df.set_index("ID", inplace=True)
    return df, spc_df
