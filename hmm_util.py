#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:49:24 2018

@author: weiqing.yu@groundtruth.com
"""

from scipy.stats import norm
from math import log
import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
from math import sin, cos, sqrt, atan2, radians
from datetime import datetime

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

def join_prob(dur, means, covs):
    res = 0.0
    for i in range(len(means)):
        for val in dur[i]:
            res += log(norm.pdf(val, means[i], covs[i]))
    return res

def choice(topK,cut):
    sc = state_change_segment([x[1] for x in topK],cut)
    scl = map(len,sc)
    ml, mp = min(scl), max(topK, key = lambda x : x[0])
    prob = -100000.0
    for i in range(len(topK)):
        if scl==ml:
            if topK[i][0]>prob:
                res = topK[i][1]
    if prob<mp[0]*1.01:
        return mp
    else:
        return (prob,res)

def init_model(n,m,d,feats,mstd=0.01):
    atmp = np.random.random_sample((n, n))+2*np.eye(n)
    row_sums = atmp.sum(axis=1)
    a = np.array(atmp / row_sums[:, np.newaxis], dtype=np.double)
#    wtmp = np.random.random_sample((n, m))
#    row_sums = wtmp.sum(axis=1)
#    w = np.array(wtmp / row_sums[:, np.newaxis], dtype=np.double)
    means_dict = {
            'dspd':[0,3,3,10,10,20],
            'angle':[0.2,2,0,1,2.5,3],
            'proximity_mode':[0.8,0.6,0.6,0.5,0.1,0.1],
            'geo':[1,1,1,1,0,0],
            'sden':[3,2,2,1.5,1.5,1],
            'dtime':[5,6,7,3,2,1],
            'ba':[0,0,0,1,1,1],
            'ddd':[0.01,0.01,0.015,0.1,0.05,0.75]}
    means = [means_dict[x] if x in means_dict \
             else np.random.uniform(0,1,m*n).tolist() \
             for x in feats]
    means = np.array(means, dtype=np.double).T\
            .reshape((n,m,d))

    covars = np.zeros((n,m,d,d))
    covs = {0:20, 1:1 ,2:0.3, 3:10, 4:10, 5:10}
    for i in xrange(n):
        for j in xrange(m):
            for k in xrange(d):
                covars[i][j][k][k] = covs[k]
    
    pitmp = np.random.random_sample((n))
    pi = np.array(pitmp / sum(pitmp), dtype=np.double)
    weights = np.ones((n,m))
    weights = (weights.T / weights.sum(axis=1)).T
    model = {'A':a, 'means':means, 'cov':covars, 'pi':pi, 'mstd':mstd, "weights":weights}
    return model

def ll_dist(l1, l2):
    R = 6373.0
    lat1, lon1, lat2, lon2 = radians(round(l1[0],5)), radians(round(l1[1],5)), radians(round(l2[0],5)), radians(round(l2[1],5))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c * 0.621371

def jumpIdent(rec,thre):
    res, dist, dist1, dist2 = [0.0], [], [], []
    for i in range(len(rec)-1):
        dist.append(ll_dist(rec[i],rec[i+1]))
    for i in range(len(rec)-2):
        dist1.append(ll_dist(rec[i],rec[i+2]))
    for i in range(len(dist)-1):
        dist2.append(dist[i]+dist[i+1])
    dif = [dist2[i]-dist1[i] for i in range(len(dist1))]
#    avgdist = sum(dif)*1.0/len(dif)
    for d in dif:
        if d>thre:
            res.append(1.0)
        else:
            res.append(0.0)
    res.append(0.0)
    return res

def jumpFlg(samp1,thre):
    sa = samp1[['lat','long','hs']].values
    s, res = 0, []
    while s < sa.shape[0]:
        if sa[s][2]==0:
            rec = []
            while sa[s][2]==0:
                rec.append(sa[s][0:2])
                s+=1
                if s==len(sa):
                    break
            if len(rec)>4:
                res+=jumpIdent(rec,thre)
            else:
                res+=[0.0]*len(rec)
        else:
            s+=1
            res.append(0.0)
    return res
    
def plotState(dat,uid,myhmm):
    try:
        samp1 = dat.loc[(dat.uid==uid)&(np.isnan(dat.haflg))].sort_values('ts')
    except:
        samp1 = dat.loc[(dat.uid==uid)].sort_values('ts')
#    obs1 = samp1.iloc[::max(1,int(samp1.shape[0]/50)),:].copy()
    obs1 = samp1.copy()
    obs = np.array(obs1[['dspd','angle','td','dd','geo','cflg']])
#    statemap = {0:"Stationary",1:"Walking",2:'Slow Moving',3:"Parking",4:"Traffic Light",5:"Driving"}
    statemap = {0:"Stationary",1:"Driving"}

    mapbox_access_token='pk.eyJ1Ijoid2VpcWluZ3kiLCJhIjoiY2ppZG56N3lnMGRtMzN3cXZ4bm95dmtxNyJ9.fe7Jmdf44QWOnGCLKsTReg'
    samp1['hs'] = choice(myhmm.viterbi_k_log(obs,20),0)[1]
    try:
        samp1_g = samp1.loc[np.isnan(samp1.haflg)]
    except:
        samp1_g = samp1
    samp1_m = samp1_g[['lat','long']].copy().values
    rec = {}
    for i in range(samp1_m.shape[0]):
        ll = tuple(samp1_m[i])
        if ll not in rec:
            rec[ll] = tuple([ll[0]+0.00003,ll[1]])
        else:
            samp1_m[i] = rec[ll]
            rec[ll] = tuple([rec[ll][0]+0.00003,rec[ll][1]])
    
    data = [
        go.Scattermapbox(
            lat=samp1_g['lat'],
            lon=samp1_g['long'],
            mode='lines+markers',
            marker = dict(
                size = 5,
                opacity = 1,
                color = samp1.hs
            ),
            hovertext = samp1_g.hs.map(statemap)+" "+samp1_g['text']
        ),
        go.Scattermapbox(
            lat=samp1_m[:,0]+0.00002,
            lon=samp1_m[:,1],
            mode='text',
            text=[str(x)+" "+samp1.time.values[x] for x in range(samp1_m.shape[0])],
            marker = dict(
                size = 5,
                opacity = 1,
                color = 'darkblue'
            ),
        ),
        go.Scattermapbox(
            lat=samp1.loc[samp1.flg==1]['lat'],
            lon=samp1.loc[samp1.flg==1]['long'],
            mode='markers',
            text="Map Plot",
            marker = dict(
                size = 10,
                opacity = 1,
                color = 'burlywood'
            ),
        )
    ]
    layout = go.Layout(
        autosize=True,
        hovermode='closest',
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=dict(
                lat=37.42,
                lon=-122.0945
            ),
            pitch=0,
            zoom=11
        ),
    )
    
    fig = dict(data=data, layout=layout)
    py.iplot(fig, filename='manual inspection')

def hmmPlot(results,uid,feats,br1,br2,br3,filename=''):
    for res in results:
        if uid in res.uid.unique():
            samp1 = res.loc[res.uid==uid].sort_values('ts').copy()
            break
#    obs1 = samp1.iloc[::max(1,int(samp1.shape[0]/50)),:].copy()
#    statemap = {0:"Stationary",1:"Walking",2:'Slow Moving',3:"Parking",4:"Traffic Light",5:"Driving"}
    samp1.reset_index(drop = True, inplace=True)
    ft = np.vectorize(datetime.fromtimestamp)
    def sft(dt):
        return datetime.strftime(dt,'%H:%M:%S')
    vsft = np.vectorize(sft)
    samp1['time'] = vsft(ft(samp1.ts/1000))
    br1_flg = np.array([any(x in y for x in br1)*5.0 for y in samp1.fp_brand.values])
    br2_flg = np.array([any(x in y for x in br2)*5.0 for y in samp1.fp_brand.values])
    br1_txt = [[br1[x] for x in br1 if x in y] for y in samp1.fp_brand.values]
    br1_txt = pd.Series([x[0] if x else "" for x in br1_txt])
    br2_txt = np.array([[br2[x] for x in br2 if x in y] for y in samp1.fp_brand.values])
    br2_txt = pd.Series([x[0] if x else "" for x in br2_txt])
    br3_txt = np.array([[br3[x] for x in br3 if x in y] for y in samp1.fp_brand.values])
    br3_txt = pd.Series([x[0] if x else "" for x in br3_txt])
    samp1['text']=samp1.time+" "+br1_txt+br2_txt+br3_txt+" "
    for i in range(len(feats)):
        samp1['text'] = samp1['text']+feats[i]+":"+samp1[feats[i]].map(str)+" "
    statemap = {0:"Stationary",1:"Driving"}
    cdict = {0:'darkgoldenrod',1:'royalblue'}

    mapbox_access_token='pk.eyJ1Ijoid2VpcWluZ3kiLCJhIjoiY2ppZG56N3lnMGRtMzN3cXZ4bm95dmtxNyJ9.fe7Jmdf44QWOnGCLKsTReg'
    samp1_m = samp1[['lat','long']].copy().values

    rec = {}
    for i in range(samp1_m.shape[0]):
        ll = tuple(samp1_m[i])
        if ll not in rec:
            rec[ll] = tuple([ll[0]+0.00003,ll[1]])
        else:
            samp1_m[i] = rec[ll]
            rec[ll] = tuple([rec[ll][0]+0.00003,rec[ll][1]])
    
    data = [
        go.Scattermapbox(
            lat=samp1['lat'],
            lon=samp1['long'],
            mode='lines+markers',
            marker = dict(
                size = 5+br1_flg+br2_flg,
                opacity = 1,
                color = samp1.hs.map(cdict)
            ),
            hovertext = samp1.hs.map(statemap)+" "+samp1['text']
        ),
        go.Scattermapbox(
            lat=samp1_m[:,0]+0.00002,
            lon=samp1_m[:,1],
            mode='text',
            text=[str(x)+" "+samp1.time.values[x] for x in range(samp1_m.shape[0])],
            marker = dict(
                size = 5,
                opacity = 1,
                color = 'darkblue'
            ),
        ),
        go.Scattermapbox(
            lat=samp1_m[:,0]+0.00002,
            lon=samp1_m[:,1],
            mode='text',
            text=[str(x)+" "+samp1.time.values[x] for x in range(samp1_m.shape[0])],
            marker = dict(
                size = 5,
                opacity = 1,
                color = 'darkblue'
            ),
        )
    ]
    layout = go.Layout(
        autosize=True,
        hovermode='closest',
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=dict(
                lat=samp1.lat.mean(),
                lon=samp1.long.mean()
            ),
            pitch=0,
            zoom=11
        ),
    )
    
    fig = dict(data=data, layout=layout)
    py.iplot(fig, filename=filename+" "+uid)