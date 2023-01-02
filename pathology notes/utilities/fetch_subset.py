# Author: Matthew Rossi

def fetch_subset(data, code, mode='include'):
    """
	data is a pandas table with the following columns: 'c.icd10_after_spilt', 'c.path_notes'
	code is the icd10 code to be included/excluded in the new dataset
	mode indicates whether to include or exclude entries with the specified icd10code

	ex. fetch_subset(data, "C50", mode='include') returns all entries whose icd10 codes contain C50
	fetch_subset(data, "C50.9", mode='exclude') returns all entries that don't contain C50.9
"""
    cols = data.columns
    if mode == 'include':
        return data[data[cols[-2]].str.contains(code)]
    if mode == 'exclude':
        return data[data[cols[-2]].str.contains(code) == False]
    
    raise Exception("Keyword 'mode' must be either 'include' or 'exclude'. Instead received '%s'." % mode)

if __name__ == "__main__":
   
    import sys
    import pandas as pd
    args = (sys.argv)
    if len(args) < 3:
        raise Exception("Usage: python fetch_subset.py <data tsv> <code> [<savefile>] [<include/exclude>]")
    filepath = args[1]
    code = args[2]
    mode = 'include'
    savefile = "fetched_subset.tsv"
    if len(args) > 3:
        savefile = args[3]
        if savefile[-4:] != '.tsv':
            savefile = savefile+'.tsv'
    if len(args) > 4:
        mode = args[4]
    data = pd.read_csv(filepath,sep='\t')
    if data.columns[0] == 'c.reident_key':
        data = data.drop(data.columns[0], axis=1)

    savefile = "~/pathologynotes/" + savefile

    subset = fetch_subset(data, code, mode)
    subset.to_csv(savefile, sep='\t', index=False)
