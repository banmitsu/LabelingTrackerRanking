import sys
import struct
import os
import numpy as np

# traverse the dir and extract the feature name

class Profile:
    pass

Attributes = {'Light', 'SurfaceCover', 'Specularity', 'Transparency', 'Shape', 'MotionSmoothness', 'MotionCoherence', 'Clutter', 'Confusion', 'LowContrast', 'Occlusion', 'MovingCamera', 'ZoomingCamera', 'LongDuration'}
dt = np.dtype([('Name', np.str_, 16), ('Score', float, (1,))])

#if __name__ == "__main__":
def service():
    score_dir     = '/home/kv/tracker_demo/'
    CLUSTER_DATAS = {}
    CLUSTER_PROFILES = {}
    # preprocessing the tracker_demo
    for root, dirs, files, in os.walk(score_dir):
        lst_TRK = dirs
        break
    print 'The tracker list:', lst_TRK
    for trk in lst_TRK:
        CLUSTER_DATAS[trk] = {}
        CLUSTER_PROFILES[trk] = {}
        for attr in Attributes:
            CLUSTER_PROFILES[trk][attr] = np.array([('MAX',sys.float_info.min),('MIN', sys.float_info.max),('MEAN', .0), ('FIAL', .0)], dtype=dt)

    for root, dirs, files in os.walk(score_dir):
        for f in files:
            tracker = root.lstrip(score_dir).split('/')[0]
            type_      = f.split('-')[1].rstrip('.txt').split('_')
            video_name = f.split('-')[1].rstrip('.txt')
            ATTR       = type_[0]
            VIDEO      = type_[1]
            score_path = '/'.join([root, f])
            if not (f.endswith('_all.txt') | f.endswith('.jpg')):
                try:
                    lst_profile = np.genfromtxt(score_path, dtype=[('index', int), ('score', float)], delimiter=',')
                except:
                    pass
                if not lst_profile['score'].size:
                    print score_path, "is empty file"
                    continue
                if lst_profile['score'].size > 1:
                    for profile in lst_profile:
                        FRAME = '{0:08d}'.format(profile['index'])
                        idx   = '_'.join([ATTR, VIDEO, FRAME])
                        #CLUSTER_DATAS[tracker][idx] = [profile['score']]
            if f.endswith('_all.txt'):
                #print score_path
                try: 
                    lst_profile = np.genfromtxt(score_path, dtype=None, delimiter=',')
                except:
                    pass
                if not lst_profile['f0'].size:
                    print score_path, "is empty file"
                    continue
                if lst_profile['f1'].size > 1:
                    for profile in lst_profile:
                        FRAME = '{0:08d}'.format(profile['f0'])
                        idx   = '_'.join([ATTR, VIDEO, FRAME])
                        lst   = [x for x in profile]
                        count = 0
                        scores= []
                        for index, score in enumerate(lst[1:]):
                            if score == -1:
                                count += 1
                            else:
                                scores.append(score)
                        scores= np.array(scores)

                        if not scores.size:
                            print tracker, ATTR, VIDEO, profile['f0'], "is failed"
                            if CLUSTER_PROFILES[tracker][ATTR][3]['Name'] == "FIAL":
                                CLUSTER_PROFILES[tracker][ATTR][3]['Score'] += 1;
                            break

                        #print scores.max(), scores.min(), scores.mean(), scores.std()
                        if CLUSTER_PROFILES[tracker][ATTR][0]['Name'] == "MAX":
                            x = CLUSTER_PROFILES[tracker][ATTR][0]['Score'];
                            CLUSTER_PROFILES[tracker][ATTR][0]['Score'] = x > scores.max() and x or scores.max()

                        if CLUSTER_PROFILES[tracker][ATTR][1]['Name'] == "MIN":
                            x = CLUSTER_PROFILES[tracker][ATTR][1]['Score'];
                            CLUSTER_PROFILES[tracker][ATTR][1]['Score'] = x < scores.min() and x or scores.min()

                        if CLUSTER_PROFILES[tracker][ATTR][2]['Name'] == "MEAN":
                            x = CLUSTER_PROFILES[tracker][ATTR][2]['Score'];
                            CLUSTER_PROFILES[tracker][ATTR][2]['Score'] = x == 0 and scores.mean() or (scores.mean()+x)/2;
                        CLUSTER_DATAS[tracker][idx] = scores.mean()

    return CLUSTER_DATAS, CLUSTER_PROFILES

#if __name__ == "__main__":
prof = Profile() 
prof.data, prof.statis = service()

