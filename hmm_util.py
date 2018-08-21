#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:49:24 2018

@author: weiqing.yu@groundtruth.com
"""

def state_change_segment(states, cut):
    def smap(s):
        return int(s > cut)
    res = [[smap(states[0]),0]]
    for i in range(1,len(states)):
        s = states[i]
        if smap(s)!=res[-1][0]:
            res[-1][1] = i-1
            res.append([smap(s),0])
    res[-1][1] = len(states)-1
    return res

def duration(seq,cs):
    res, lc = {0:[],1:[]}, 0
    for c in cs:
        if c[0]==0:
            res[0].append((seq[c[1]]-seq[lc])*1.0/60000)
        else:
            res[1].append((seq[c[1]]-seq[lc])*1.0/60000)
        lc = c[1]
    return res
