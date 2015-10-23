import sys
import struct
import os
import numpy as np

class Indexer:
    pass
    
def service():
    index_list     = 'start2end.index'
    FEAT_SCORE_TABLE = {}
    with open(index_list) as f:
        s2e_lst = np.genfromtxt(f, dtype=[('name', np.str_, 1024), ('s2e', np.str_, 1024)], delimiter=' ')
    for item in s2e_lst:
        attr = item['name'].split('/')[-2].split('_')[0].split('-')[-1]
        video= item['name'].split('/')[-2].split('_')[1]
        featuref= item['s2e'].split(':')[0]
        scoref  = item['s2e'].split(':')[1]
        idx     = ('_').join([attr, video, featuref])
        FEAT_SCORE_TABLE[idx] = scoref

    return FEAT_SCORE_TABLE
        #break

#if __name__ == "__main__":
index = Indexer()
index.data = service()
