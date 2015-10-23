import sys
import struct
import os
from random import shuffle
import numpy as np

import Profile
from Profile import prof
import Indexer
from Indexer import index

def _upack_feature_(filename):
    fd = open(filename, 'rb')
    n = struct.unpack('i', fd.read(4))[0]
    c = struct.unpack('i', fd.read(4))[0]
    l = struct.unpack('i', fd.read(4))[0]
    h = struct.unpack('i', fd.read(4))[0]
    w = struct.unpack('i', fd.read(4))[0]
    fc = []
    for i in range(n * c):
        feat = struct.unpack('f', fd.read(4))[0]
        fc.append(feat)
    return fc

def _convert_svm_format_(feat_list):
    tmpS = ''
    i = 0
    # convert to svm format and save as the file
    for i in range(0, len(feat_list)):
        tmpS += str(i+1) + ':' + str(feat_list[i]) + ' '
    return tmpS

# traverse the dir and extract the feature name
output_name = 'STR_svm_input'
output = open(output_name, 'w+')
output_list = []

#score_list_name = sys.argv[1]
#feat_dir = sys.argv[2]
feature_list = 'input_16023.txt'
feature_dir  = '/home/kv/research/C3D/examples/c3d_tr_feature_extraction/output/alov_feat/16023/'

feat_type = 'fc6'
tracker_type = 'STR'

print 'Read features from', feature_dir , 'with format:', feat_type
print 'genderate score according to', feature_list
count = 0

if __name__ == '__main__':

    with open(feature_list) as f:
        feat_lst = np.genfromtxt(f, dtype=[('name', np.str_, 1024), ('frame', np.str_, 1024), ('label', int)], delimiter=' ')

    for item in feat_lst:
        attr  = item['name'].split('/')[-2].split('_')[0].split('-')[-1]
        video = item['name'].split('/')[-2].split('_')[1]
        frame = item['frame']
        # feature -> scores
        feature_key = '_'.join([attr, video, frame])
        try:
            score_key   = '_'.join([attr, video, index.data[feature_key]])
            # The score
            score = prof.data[tracker_type][score_key]
            # The feature
            feat_name   = feature_dir+ feat_type+'/'+attr+'_'+video+'-'+frame+'.'+feat_type
            feature     = _upack_feature_(feat_name)
            svm_feature = _convert_svm_format_(feature)
            # Gen_svm
            line = np.str_(score[0]) + ' ' + svm_feature + '\n'
            output_list.append(line)
            output.write(line)
            count = count +1
        except: 
            pass
        #break

print 'Done,', count, 'features saved in', output_name
output.close()


output_train = open(output_name + '_train', 'w+')
output_test  = open(output_name + '_test', 'w+')

shuffle(output_list)
total_num = len(output_list)
test_num = total_num /10
train_num = total_num - test_num
train = output_list[:train_num]
test  = output_list[train_num:]
for item in train:
    output_train.write('%s' % item)
for item in test:
    output_test.write('%s' % item)
output_train.close(), output_test.close()
print 'Done, train/test set: ', train_num, '-', test_num



