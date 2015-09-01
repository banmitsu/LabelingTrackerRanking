#! /usr/bin/env python
#! FelisJ release. 2015-08-13
import numpy as np
import os, sys
from collections import namedtuple
import operator
from random import shuffle
import argparse

#dir_GT = '/home/kv/alov300++GT_txtFiles/alov300_rectangleAnnotation_full/'
dir_GT   = sys.argv[1]
# dir_DATA = '/home/kv/research/C3D/examples/c3d_tr_feature_extraction/dataset/ALOV/alov300++_frames/imagedata'
dir_DATA = 'dataset/ALOV/alov300++_frames/imagedata' # string for input.lst (bug: opencv dep.)
# dir_FEAT = '/home/kv/research/features/c3d_1Msport/'
dir_FEAT = 'output/alov_feat/'                       # string for prefix.lst (bug: opencv dep.)
# dir_TRK  = '/home/kv/tracker_demo/'
dir_TRK = sys.argv[2]

nameto_lst_DATA = 'ALOVvideo.lst'       # The whole list of the dataset
nameto_c3d_datalyr_source = 'input.lst' # The input-list for the c3d feature extraction
nameto_c3d_output_prefix = 'prefix.lst' # The output-prefix for the c3d
nameto_lst_LABEL = 'start2end.index'    # The index file of start-end frame
global lst_TRK 


try:
    fileto_lst_DATA = open(nameto_lst_DATA, 'w+')
    fileto_lst_LABEL = open(nameto_lst_LABEL, "w+")
    fileto_c3d_datalyr_source = open(nameto_c3d_datalyr_source, "w+")
    fileto_c3d_output_prefix = open(nameto_c3d_output_prefix, "w+")
except:
    print "Could not create the files: ALOVvideo.lst, start2end.index, input.lst, prefix.lst"
    sys.exit()



def _append_labels(str1):
    return str1 + ' 1\n'

def _c3d_prefix(str1): 
    str1_token = str1.split('/')
    video_name = str1_token[-2].split('-')[-1]
    frame_name = str1_token[-1].split(' ')[-1]
    return dir_FEAT + video_name + '-' + frame_name + '\n'

def _process_c3d_(pathto_each_file):
    nameto_each_file_token = pathto_each_file.split('.')[0].split('/')
    nameto_each_file = '/'.join([dir_DATA, nameto_each_file_token[-2], nameto_each_file_token[-1] ])
    nameto_each_frame = '/ '.join([nameto_each_file, numof_each_frame]) 
    fileto_c3d_datalyr_source.write( _append_labels(nameto_each_frame) )
    fileto_c3d_output_prefix.write(  _c3d_prefix(nameto_each_frame) )
    return

def _close_c3d_():
    fileto_c3d_datalyr_source.close()
    fileto_c3d_output_prefix.close()

def _process(pathto_each_file):
    nameto_each_file_token = pathto_each_file.split('.')[0].split('/')
    nameto_each_file = '/'.join([dir_DATA, nameto_each_file_token[-2], nameto_each_file_token[-1] ])
    nameto_each_frame = '/ '.join([nameto_each_file, numof_each_frame])
    # generate the start2end.index
    fileto_lst_LABEL.write( nameto_each_frame + ':' + keyto_label + '\n')
    # generate the ALOV.lst
    fileto_lst_DATA.write(  nameto_each_frame + '\n')
    return

def _traverse_exit():
    fileto_lst_LABEL.close()
    fileto_lst_DATA.close()



#parser = argparse.ArgumentParser(description='Parse ALOV dataset')
#parser.add_argument('dirname', type=valid_dirname, help='dirname to the ground truth files')
#parser.add_argument('dirname', type=valid_dirname, help='dirname to the tracker results')
#args = parser.parse_args()


KEYto_FEAT = {}
# -----------------------------------------------------------------------------------------------
# traverse all the video in the dataset
# according to the ground truth file to create the index "start2end.index" [start frame:end frame]
# -----------------------------------------------------------------------------------------------
for root, dirs, files in os.walk(dir_GT):
    for each_file in files:
        pathto_each_file = '/'.join([root, each_file])
        with open(pathto_each_file) as f:
            lines = f.readlines()
            # for each_frame in lines:
            for index in range(len(lines)-1):
                # for each frame in an .ann, format it as 00001111
                each_frame = lines[index]
                numof_frame = each_frame.split(' ')[0].rstrip('\n')
                numof_each_frame =  "{:0>8}".format(numof_frame)
                # generate the "key" to the label of each frame
                numof_label = lines[index+1].split(' ')[0].rstrip('\n')
                keyto_label = "{:0>8}".format(numof_label)
                # To check the length of the last frame is enough for c3d feature extraction
                if (index == len(lines)-2): 
                    if (int(numof_label)-int(numof_frame) < 5) :
                        pass
                    else:
                        KEYto_FEAT[keyto_label] = numof_each_frame
                        _process( pathto_each_file )
                        _process_c3d_( pathto_each_file ) # process the input file of _c3d_
                else:
                    KEYto_FEAT[keyto_label] = numof_each_frame
                    _process( pathto_each_file)
                    _process_c3d_( pathto_each_file ) # process the input file of _c3d_

_traverse_exit()
_close_c3d_()

# --------------------------------------------------------------------------------------------------
# traverse all the tracker's result and all score values
# -------------------------------------------------------------------------------------------------
lst_TRK = []
CLUSTER_DATAS = {} # ALL scores for ALL trackers
CLUSTER_LABELS = {}# ALL ranks for ALL trackers
for root, dirs, files in os.walk(dir_TRK):
    lst_TRK = dirs
    break
print "The tracker list:", lst_TRK
for trk in lst_TRK:
    CLUSTER_DATAS[trk] = {}
    CLUSTER_LABELS[trk] = {}

for root, dirs, files in os.walk(dir_TRK):
    for f in files:
        nameto_tracker = root.lstrip(dir_TRK).split('/')[0]
        # current version:
        if f.endswith('txt'):
            type_ = f.split('-')[1].rstrip('.txt').split('_')
            video_name = f.split('-')[1].rstrip('.txt')
            ATTR = type_[0]       #Attribute = Transparency
            VIDEO = type_[1]      #video00001
            score_path = '/'.join([root, f]) 
            # read the score file
            lst_profile = np.genfromtxt(score_path, dtype=[('index', int), ('score', float)], delimiter=',')
            # To check the score list is not an empty file
            if not lst_profile['score'].size:
                print score_path, 'is empty file'
                continue
            #if score['score'].size >1:
            #    total_scores.extend(score['score'].tolist())
            #else:
            #    total_scores.append(score['score'].tolist())
            # set the dict which key=attribute, value is a tuple list
            # T.setdefault(attr, []).append(score_statistic(score_path, video_num, avg_, std_, max_, min_))
            if lst_profile['score'].size >1:
                for profile in lst_profile:
                    key = '{0:08d}'.format(profile['index'])
                    idx_feature = KEYto_FEAT[key]
                    # print "Tracker:", nameto_tracker
                    # print "End frame:", key
                    # print "Index feature:", idx_feature
                    # print "Score:", profile['score']
                    ptr = ATTR+"_"+VIDEO+"_"+idx_feature # the key to score for current tracker
                    CLUSTER_DATAS[nameto_tracker][ptr] = [profile['score']]


# ---------------------------------------------------------------------------------------------------
# Create Labels
# ---------------------------------------------------------------------------------------------------

global file_handles
file_handles = []

def _gen_label_(handle, each_video, label):
    if handle['data_name'] == 'c3d':
        video = each_video.split(':')[0]
    if handle['data_name'] == 'caffe':
        prefix = '/home/kv/research/C3D/examples/c3d_tr_feature_extraction/'
        video_ = each_video.split(':')[0]
        video  = prefix + video_.split(' ')[0]+ video_.split(' ')[-1] + '.jpg'
    handle['fileptr'].write( video + " " + label + '\n' )
    return


def _open_file_(class_typ, data_name):
    nameto_datalyr_source = data_name + '_' + class_typ + '_input.lst'
    try:
        fileto_datalyr_source = open(nameto_datalyr_source, "w+")
    except:
        print "Could not open source", data_name, '_', class_typ, '_input.lst'
    file_handles.append(fileto_datalyr_source)
    return {'fileptr': fileto_datalyr_source, 'filename' : nameto_datalyr_source, 'class_typ': class_typ, 'data_name': data_name}


def _close_all_():
    for fp in file_handles:
        try:
            fp.close()
        except:
            pass

def _gen_source_(inputfile, prefix):
    nameto_datalyr_train_ = prefix+'_train_input.lst'
    nameto_datalyr_test_  = prefix+'_test_input.lst'
    try:
        fileto_train = open( nameto_datalyr_train_, "w+" )
        fileto_test  = open( nameto_datalyr_test_, "w+" )
    except:
        print "Fail: _gen_source_()"
    with open(inputfile) as f:
        content = f.readlines()
    total_num = len(content)
    test_num  = total_num/4
    train_num = total_num - test_num
    print prefix, "_gen_source_():", test_num, "-", train_num
    shuffle(content)
    train = content[:train_num]
    test  = content[train_num:]
    for item in train:
        fileto_train.write('%s' % item)
    for item in test:
        fileto_test.write('%s' % item)
    fileto_train.close()
    fileto_test.close()
    return



scores_trk = {}
scores_lst = []

handle_c3d_trkrank = _open_file_('trkrank', 'c3d')
handle_caffe_trkrank = _open_file_('trkrank', 'caffe')


# Create labels for each video clips
with open(nameto_lst_LABEL,'r') as f: # start2end.index
    for each_video in f:
        try:
            prefix_ = each_video.split(' ')[1]
        except:
            print "NO INTRE:", each_video
            sys.exit()
        # generate the key
        suffix_ = each_video.split(' ')[0]
        ATTR_VIDEO = suffix_.split('/')[-2].split('-')[-1]
        START_FRAME = prefix_.split(':')[0]
        ptr = ATTR_VIDEO + "_" + START_FRAME
        for each_tracker in lst_TRK:
            try:
                scores_trk[each_tracker] = CLUSTER_DATAS[each_tracker][ptr]
            except:
                pass
                # print "NO DATA:", each_tracker, ptr
            try:
                scores_lst.append(CLUSTER_DATAS[each_tracker][ptr])
            except:
                scores_lst.append(-1)
        # Rank the scores among trakcers and product labels (classification)
        sorted_trk = sorted(scores_trk.items(), key=operator.itemgetter(1), reverse=True)
        for ind, each_trk in enumerate(sorted_trk):
            CLUSTER_LABELS[each_trk[0]][ptr] = ind

        # Classification #1 (by the best tracker name)
        label_ = lst_TRK.index(sorted_trk[0][0])  
        label = "{:0>1}".format(label_)
        # Classification #2 (by success rate of all trackers)

        # Classification #3 (by attribute)

        # avg_ = np.mean(score_lst)
        # std_ = np.std(score_lst)
        # max_ = np.max(score_lst)
        # min_ = np.min(score_lst)
        _gen_label_(handle_c3d_trkrank, each_video, label)
        _gen_label_(handle_caffe_trkrank, each_video, label)

        scores_lst = []

# close all opened files
_close_all_()

print "Done, "

# generate train/test input.lst for training network
_gen_source_( handle_c3d_trkrank['filename'], 'c3d_trkrank' )
_gen_source_( handle_caffe_trkrank['filename'], 'caffe_trkrank')


# travel through all attribute

# create the dict of score
sys.exit()

# ----------------------------

# output_file = open(score_list_name, 'w+')
with open(video_list_name) as f:
    file_content = f.readlines()
for line in file_content:
    line_list = line.split(' ')
    path = line_list[0]
    prefix = path.split('/')[-2].split('-')[1]
    clip_num = line_list[1].rstrip('\n')
    key = prefix + '_' + clip_num
    try:
        output_score = TR[key]
    except KeyError:
        print key, 'missed video clip'
        output_score = -1.0
        pass
    new_line = ' '.join([path, clip_num, str(output_score)])
    output_file.write(new_line +'\n')
output_file.close()
