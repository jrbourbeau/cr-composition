
import json
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


def serialize_SFS(sfs, outfile):
    params = sfs.get_params()
    with open(outfile, 'wb') as f_obj:
        json.dumps(params, f_obj)


def deserialize_SFS(infile):
    sfs = SFS()
    with open(infile, 'rb') as f_obj:
        params = json.loads(f_obj)
    sfs.set_params(params)
    return sfs
