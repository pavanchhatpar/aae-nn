"""This module converts the aae-form-data from JSON to MAT files"""
import json
import os
from mat4py import savemat
import numpy as np

with open('aae-form-root-canal-ml-export.json') as json_data:
    d = json.load(json_data)
# o = {'X_data': [], 'y_data': []}
o = {'raw':[]}
i = 0

def mkdir_if_not_exists(path):
    """This function will create a directory <path> if it does not exist"""
    if not os.path.exists(path):
        os.makedirs(path)

for key in d['forms']:
    # o['X_data'].append([])
    r = []
    k = 0
    for j in range(len(d['forms'][key]['responses'])):
        # o['X_data'][i].append(0)
        tmp = d['formResponses'][d['forms'][key]['responses'][j]]
        # if d['formResponses'][d['forms'][key]['responses'][j]]['section'] == 'A':
        #     o['X_data'][i].extend([1,0,0])
        # else:
        # o['X_data'][i].extend(tmp['min'])
        # o['X_data'][i].extend(tmp['mod'])
        # o['X_data'][i].extend(tmp['high'])
        r.extend(tmp['min'])
        r.extend(tmp['mod'])
        r.extend(tmp['high'])
            # o['X_data'][i][k] += sum(tmp['min']) * 1.0
            # o['X_data'][i][k] += sum(tmp['mod']) * 2.0
            # o['X_data'][i][k] += sum(tmp['high']) * 5.0
        k += 1
    # o['y_data'].append([1] if d['forms'][key]['referredToSpl'] else [0])
    r.append(1 if d['forms'][key]['referredToSpl'] else 0)
    o['raw'].append(r)
    i += 1
mkdir_if_not_exists('octave')
savemat('octave/aae-raw-data.mat', o)
np.save('aae-raw-data', o)
