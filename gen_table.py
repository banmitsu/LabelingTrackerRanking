import sys
import struct
import os
import numpy as np

import Profile
from Profile import prof

Attributes = ('Light','SurfaceCover','Specularity','Transparency','Shape','MotionSmoothness', 'MotionCoherence','Clutter','Confusion','LowContrast','Occlusion','MovingCamera','ZoomingCamera','LongDuration')
Trackers  = ('STR','CSK','TLD','ASLA')

if __name__ == '__main__':
	for trk in Trackers:
		print trk
		scores = np.array([0.0])
		for attr in Attributes:
			#print attr
			#print prof.statis[trk][attr][0]
			scores = scores + prof.statis[trk][attr][3][1]
		print scores, scores/14



