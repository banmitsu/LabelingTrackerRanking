# LabelingTrackerRanking

This is a script to generate caffe/c3d input fo tracker ranking task.

(1) Dataset: Amsterdam Library of Ordinary Videos for evaluating visual trackers robustness
    http://www.alov300.org/
 
(2) Tracked results: (tracker_demo)


steps-1

ALOVparser.py: 
Parse the alov300 dataset, generate two files.
- The video list of alov300
- The index file (start-to-end frame) according to the Ground truth files of alov300

	- ipython ALOVparser.py /path/to/alov300++GT_txtFiles/alov300_rectangleAnnotation_full /path/to/alov300++_frames/imagedata

ScoresCaffe.py:	
Generate caffe input (label is according to tracker results)

	- ipython ScoresCaffe.py /path/to/tracker_demo label_type

	label_type = ['rank', 'comx']

genSet.py:	
Output train/test_caffe_input.lst

	- ipython genSet.py caffe_input.lst




