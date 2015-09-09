import sys
import struct
import os
from random import shuffle
import numpy as np

score_list_name = sys.argv[1]
feat_dir = sys.argv[2]

score_dict = {}
feat_type = '.pool5'

print 'Read features from', feat_dir , 'with format:', feat_type
print 'genderate score according to', score_list_name

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
output_name = 'svm_input'
output = open(output_name, 'w+')
output_list = []

with open(score_list_name) as f:
    lst = np.genfromtxt(f, dtype=[('name', np.str_, 1024), ('label', int)], delimiter=' ')

for item in lst:
    video = item['name'].split('.')[0].split('/')[-2].split('-')[-1]
    frame = item['name'].split('.')[0].split('/')[-1]
    feat_name = feat_dir+video+'-'+frame+feat_type
    feature = _upack_feature_(feat_name)

    svm_feature = _convert_svm_format_(feature)
    label = np.str_(item['label'])

    line = label + ' ' + svm_feature + '\n'
    output_list.append(line)
    output.write(line)


print 'Done, all features saved in', output_name
output.close()

output_train = open(output_name + '_train', 'w+')
output_test  = open(output_name + '_test', 'w+')

shuffle(output_list)
total_num = len(output_list)
test_num = total_num /4
train_num = total_num - test_num
train = output_list[:train_num]
test  = output_list[train_num:]
for item in train:
    output_train.write('%s' % item)
for item in test:
    output_test.write('%s' % item)
output_train.close(), output_test.close()



