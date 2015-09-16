#! /usr/bin/env python
#! FelisJ release. 2015-09-16
import numpy as np
import os, sys

def _process(pathto_each_file):
    nameto_each_file_token = pathto_each_file.split('.')[0].split('/')
    nameto_each_file = '/'.join([args['dir_DATA'], nameto_each_file_token[-2], nameto_each_file_token[-1] ])
    nameto_each_frame = '/ '.join([nameto_each_file, index_start_frame ])
    # generate the start2end.index
    fileto_lst_LABEL.write( nameto_each_frame + ':' + index_end_frame + '\n')
    # generate the ALOV.lst
    fileto_lst_DATA.write(  nameto_each_frame + '\n')

# -----------------------------------------------------------------------------------------------
# traverse all the video in the dataset
# according to the ground truth file to create the index "start2end.index" [start frame:end frame]
# -----------------------------------------------------------------------------------------------
if __name__ == '__main__':

    nameto_lst_DATA = 'ALOVvideo.lst'       # The whole list of the dataset
    nameto_lst_LABEL = 'start2end.index'    # The index file of start-to-end frame according to GT files
    global lst_TRK 
    try:
    	fileto_lst_DATA = open(nameto_lst_DATA, 'w+')
	fileto_lst_LABEL = open(nameto_lst_LABEL, "w+")
    except:
    	print "Could not create the files: ALOVvideo.lst, start2end.index, input.lst, prefix.lst"
    	sys.exit()
    arg_names = ['command', 'dir_GT', 'dir_DATA']
    args = dict(zip(arg_names, sys.argv))
    if len(args) < 3:
    	print "Error command:..."
	print "ipython ALOVparser.py /path/to/alov300++GT_txtFiles/alov300_rectangleAnnotation_full /path/to/alov300++_frames/imagedata"
	sys.exit()

    for root, dirs, files in os.walk(args['dir_GT']):
    	for each_file in files:
            pathto_each_file = '/'.join([root, each_file])
	    # open .ann
      	    with open(pathto_each_file) as f:
                lines = f.readlines()
	        for index in range(len(lines)-1):
	 	    # for each frame in an .ann, format it as 00001111
		    start_frame = lines[index].split(' ')[0].rstrip('\n')
		    index_start_frame = "{:0>8}".format(start_frame)
		    end_frame   = lines[index+1].split(' ')[0].rstrip('\n')
		    index_end_frame   = "{:0>8}".format(end_frame)
		    _process( pathto_each_file )

    fileto_lst_LABEL.close()
    fileto_lst_DATA.close()


# ----------------------------
