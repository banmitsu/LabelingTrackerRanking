#! /usr/bin/env python
#! FelisJ release. 2015-09-16
import numpy as np
import os, sys
import operator
from random import shuffle

global file_handles
file_handles = []

def _video_filename_parser_(item):
    NAME  = item.split(' ')[0].split('/')[-2]
    ATTR  = NAME.split('_')[0].split('-')[-1]
    VIDEO = NAME.split('_')[-1].split('-')[-1]
    START_F = item.split(' ')[-1].split(':')[0].rstrip('\n')
    END_F   = item.split(' ')[-1].split(':')[-1].rstrip('\n')
    return {'ATTR': ATTR, 'VIDEO': VIDEO, 'START_F': START_F, 'END_F': END_F}
        
def _process_inxfile_(inxfile):
    hashScoreInx = {}
    for item in inxfile:
    	token = _video_filename_parser_(item)
	KEY = token['ATTR']+'_'+token['VIDEO']+'_'+token['START_F']
	hashScoreInx[KEY] = token['END_F']
    return hashScoreInx

def _open_file_(label_typ):
    nameto_datalyr_source = 'caffe_' + label_typ + '_input.lst'
    try:
        fileto_datalyr_source = open(nameto_datalyr_source, "w+")
    except:
        print "Could not open source", 'caffe_', label_typ, '_input.lst'
    file_handles.append(fileto_datalyr_source)
    return {'fileptr': fileto_datalyr_source, 'filename' : nameto_datalyr_source, 'label_typ': label_typ}

def _close_all_():
    for fp in file_handles:
        try:
            fp.close()
        except:
            pass

def _gen_label_(handle, each_video, label):
    video_ = each_video.rstrip('\n')
    video  = video_.split(' ')[0]+ video_.split(' ')[-1] + '.jpg'
    handle['fileptr'].write( video + " " + label + '\n' )



# --------------------------------------------------------------------------------------------------
# traverse all the tracker's result and all score values
# -------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    args_name = ['command', 'dir_TRK', 'label_typ']
    args = dict(zip(args_name, sys.argv))
    if len(args) < 3:
    	print "Error command:..."
	print "ipython ScoresCaffe.py /path/to/tracker_demo label_type" 
	sys.exit()

    global lst_TRK
    lst_TRK = []
    CLUSTER_DATAS = {} # ALL scores for ALL trackers
    CLUSTER_LABELS ={} # ALL ranks for ALL trackers
    for root, dirs, files in os.walk(args['dir_TRK']):
    	lst_TRK = dirs
    	break
    print "The tracker list:", lst_TRK
    for trk in lst_TRK:
    	CLUSTER_DATAS[trk] = {}
    	CLUSTER_LABELS[trk] = {}

    for root, dirs, files in os.walk(args['dir_TRK']):
    	for f in files:
            nameto_tracker = root.lstrip(args['dir_TRK']).split('/')[0]
            if f.endswith('txt'):
            	type_ = f.split('-')[1].rstrip('.txt').split('_')
            	video_name = f.split('-')[1].rstrip('.txt')
            	ATTR  = type_[0]       #Attribute = Transparency
            	VIDEO = type_[1]       #video00001
            	score_path = '/'.join([root, f]) 
            	# read the score file
            	lst_profile = np.genfromtxt(score_path, dtype=[('index', int), ('score', float)], delimiter=',')
            	# To check the score list is not an empty file
            	if not lst_profile['score'].size:
                     print score_path, 'is empty file'
                     continue
            	if lst_profile['score'].size >1:
                    for profile in lst_profile:
                    	key = '{0:08d}'.format(profile['index'])
                    	ptr = ATTR+"_"+VIDEO+"_"+key # Clutter_video00006_00000014
                    	CLUSTER_DATAS[nameto_tracker][ptr] = [profile['score']]
			# print nameto_tracker, ptr, CLUSTER_DATAS[nameto_tracker][ptr]

# --------------------------------------------------------------------------------------------------
# generate labels
# -------------------------------------------------------------------------------------------------
    with open("start2end.index") as f:
    	indexfile = f.readlines()
    hashScoreInx = _process_inxfile_(indexfile)

    with open("ALOVvideo.lst") as f:
    	videolist = f.readlines()

    handle_caffe = _open_file_(args['label_typ'])

    scores_trk = {}
    scores_lst = []
    for item in videolist:
	token = _video_filename_parser_(item)
	END_F  = hashScoreInx[token['ATTR']+'_'+token['VIDEO']+'_'+token['START_F']]
	vptr   = token['ATTR']+'_'+token['VIDEO']+'_'+END_F
	# ---
    	for each_tracker in lst_TRK:
             try:
             	  scores_trk[each_tracker] = CLUSTER_DATAS[each_tracker][vptr]
             except:
             	  print "NO DATA:", each_tracker, vptr
             try:
             	  scores_lst.append(CLUSTER_DATAS[each_tracker][vptr])
             except:
             	  scores_lst.append(-1.0)
        # Rank the scores among trakcers and product labels (classification)
        sorted_trk = sorted(scores_trk.items(), key=operator.itemgetter(1), reverse=True)
        for ind, each_trk in enumerate(sorted_trk):
             CLUSTER_LABELS[each_trk[0]][vptr] = ind
        # Classification #1 "trkrank" (by the best tracker name)
	if (args['label_typ']=='rank'):
             label_ = lst_TRK.index(sorted_trk[0][0])  
             label = "{:0>1}".format(label_)
             _gen_label_(handle_caffe, item, label)
        # Classification #2 (by success rate of all trackers)
	if (args['label_typ']=='comx'):
             for ind, each_score in enumerate(scores_lst):
	     	 try:
                     [float(i) for i in each_score]
                 except:
                     i = each_score
                 if( i >= 0.99):
                     scores_lst[ind] = 1
                 else:
                     scores_lst[ind] = 0
   	     label_ = np.sum(scores_lst)
	     label = "{:0>1}".format(label_)
	     _gen_label_(handle_caffe, item, label)
        # Classification #3 (by attribute)
        # avg_ = np.mean(score_lst)
        # std_ = np.std(score_lst)
        # max_ = np.max(score_lst)
        # min_ = np.min(score_lst)
        scores_lst = []

# close all opened files
_close_all_()

print "Done, generate caffe input list:...", 'caffe_'+args['label_typ']+'_input.lst'

            

