#!/user/bin/env python
# (1) parse feature
# (2) generate svm input
# (3) generate lmdb

import os
import argparse
import sys
import numpy as np
from random import shuffle
import lmdb
from caffe.io import array_to_datum

traj_len = 15

INFO = 1 # show info

feat_dir = '/home/kv/research/improved_trajectory_release/feature/'
feat_lst_name = sys.argv[1]
label_lst_name = sys.argv[2]
try: 
    lmdb_filename = sys.argv[3]+ '_iDT_lmdb'
    svm_filename  = sys.argv[3]+ 'iDT_svm_input'
except:
    lmdb_filename = 'iDT_lmdb'
    svm_filename  = 'iDT_svm_input'

print '[IDTfeatParser] Read features from...', feat_lst_name
print '[IDTfeatParser] Label according to...', label_lst_name

global LABEL_DATA
LABEL_DATA = {}
global file_handles
file_handles = []
global num_of_dims

def _create_files_():
    try: 
        svm_handle = open(svm_filename, 'w+')
    except:
        print '[IDTfeatParser] open svm_input failed.'
    file_handles.append(svm_handle)
    return svm_handle

def _close_files_():
    for handle in file_handles:
        try:
            handle.close()
        except:
            pass

def _process_(lst_name):
    with open(lst_name) as f:
        lst = f.readlines()
    for each_line in lst:
        video = each_line.split(' ')[0].split('/')[-2]
        frame = each_line.split(' ')[1]
        label = each_line.split(' ')[2].strip('\n')
        vptr  = video+'-'+frame
        LABEL_DATA[vptr] = label

def _index_label_(string):
    vptr = string.split('.gz')[0].split('/')[-1]
    return LABEL_DATA[vptr], vptr

def _parse_info_(data):
    frameNum, mean_x, mean_y, var_x, var_y, length, scale, x_pos, y_pos, t_pos = [x for x in data]
    return frameNum, mean_x, mean_y, var_x, var_y, length, scale, x_pos, y_pos, t_pos

def _parse_feature_(string):
    filename = feat_dir + string + '.gz'
    features = np.genfromtxt(filename, dtype=float,  autostrip=True, invalid_raise=False)
    each_feature = features[-1]
    if features.size == 436:
        each_feature = features
    if INFO == 1:
        print "** Feature: ", string, "/** Number of features: ", len(features)
    info= _parse_info_(each_feature[0:10])  # The first 10 elements
    Traj= each_feature[10:40]      # default: 30 dimension
    HOG = each_feature[40:136]     # default: 96 dimension
    HOF = each_feature[136:244]    # default: 108 dimension
    MBHx= each_feature[244:340]    # default: 96 dimension
    MBHy= each_feature[340:436]    # default: 96 dimension
    if INFO == 1:
        print "** Dimension of features: ", len(each_feature), "/** Frame: ", info[0]
    return each_feature[10:436]

def _svm_feature_(feature):
    tmpS = ''
    i = 0
    # convert to svm format
    for i in range(0, len(feature)):
        tmpS += str(i+1) + ':' + str(feature[i]) + ' '
    return tmpS

def _lmdb_feature_(feature):
    tokens = feature.rstrip().split()
    arr = [float(dim.split(':')[1]) for dim in tokens[1:]]
    return arr

def _create_lmdb_(X,Y):
    num = np.prod(X.shape)
    itemsize = np.dtype(X.dtype).itemsize
    # set a reasonable upper limit for database size
    map_size = 10240 * 1024 + num * itemsize * 2
    print 'save {} instances...'.format(num)
    env = lmdb.open(lmdb_filename, map_size=map_size)
    for i, (x, y) in enumerate(zip(X, Y)):
        datum = array_to_datum(x, y)
        str_id = '{:08}'.format(i)
        with env.begin(write=True) as txn:
            txn.put(str_id, datum.SerializeToString())

# def valid_filename(string):
#    try:
#        if(os.path.isfile(string)):
#            return string
#        else:
#            raise argparse.ArgumentTypeError('Must be valid files')
#    except Exception as e:
#        print 'ERROR: {}'.format(e)
#        raise argparse.ArgumentTypeError('Must be valid files')

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Parser for iDT features')
    # parser.add_argument('-f', type=valid_filename)
    # args = parser.parse_args()
    # feat_filename = args.f

    #* store video2labels as an index dictionary *#
    _process_(label_lst_name)
    #* output: LABEL_DATA[vptr] = label *#
    handles = _create_files_()

    with open(feat_lst_name) as f:
        feat_lst = f.readlines()
    items = []
    for each_feature in feat_lst:
        index = _index_label_(each_feature)
        label = index[0]
        feature= _parse_feature_(index[1])
        num_of_dims = len(feature)
        svm_format = label +' '+ _svm_feature_(feature) + '\n'
        items.append( (int(label), _lmdb_feature_(svm_format)) )
        handles.write(svm_format)
        #  break
    # shuffle
    Y = np.array([y for y, _ in items])
    X = np.array([x for _, x in items])
    X = X.reshape( (len(Y), 1, 1, num_of_dims) )
    _create_lmdb_(X, Y)
    
    _close_files_()


