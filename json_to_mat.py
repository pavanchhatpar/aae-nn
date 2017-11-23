"""This module converts the aae-form-data from JSON to MAT files"""
import json
import os
from mat4py import savemat

with open('aae-form-root-canal-ml-export.json') as json_data:
    d = json.load(json_data)
o = {'X_data': [], 'y_data': []}
i = 0

def mkdir_if_not_exists(path):
    """This function will create a directory <path> if it does not exist"""
    if not os.path.exists(path):
        os.makedirs(path)

for key in d['forms']:
    o['X_data'].append([])
    k = 0
    for j in range(len(d['forms'][key]['responses'])):
        o['X_data'][i].append(0)
        tmp = d['formResponses'][d['forms'][key]['responses'][j]]
        if d['formResponses'][d['forms'][key]['responses'][j]]['section'] == 'A':
            o['X_data'][i][k] = 1
        else:
            o['X_data'][i][k] += sum(tmp['min']) * 1.0
            o['X_data'][i][k] += sum(tmp['mod']) * 2.0
            o['X_data'][i][k] += sum(tmp['high']) * 5.0
        k += 1
    o['y_data'].append([2.0] if d['forms'][key]['referredToSpl'] else [1.0])
    i += 1
mkdir_if_not_exists('octave')
savemat('octave/aae-data.mat', o)
