#!/usr/bin/python3

import json
import pandas as pd
import getpass
import os

machines = range(1,5)
#apps = ['bt.C.x', 'cg.C.x', 'dc.B.x', 'ft.B.x', 'lu.C.x', 'mg.B.x', 'sp.D.x', 'ua.C.x']
apps = ['blackscholes', 'canneal', 'ferret', 'freqmine', 'bodytrack', 'bt.C.x', 'cg.C.x', 'dc.B.x', 'ft.B.x', 'lu.C.x', 'mg.B.x', 'sp.D.x', 'ua.C.x']
#apps_train = ['bt.C.x', 'cg.C.x', 'dc.B.x', 'ep.D.x', 'ft.B.x', 'is.C.x', 'lu.C.x']
#apps_validation = ['mg.B.x', 'sp.C.x', 'ua.C.x']

fanGs = { 'FanGroup1' : ['Fan1A', 'Fan1B', 'Fan2A', 'Fan2B', 'Fan3A', 'Fan3B'], 'FanGroup2' : ['Fan4A', 'Fan4B'], 'FanGroup3' : ['Fan5A', 'Fan5B', 'Fan6A', 'Fan6B', 'Fan7A', 'Fan7B'] }
appattrs = ['cyc', 'inst', 'llcref', 'llcmiss', 'br', 'brmiss', 'l2lin']
phyattrs = ['power', 'fanpower']
apprates = ["%s_rate" % x for x in appattrs]
homedir = os.environ['HOME']

def getFanPwr(rpm, pwr):
	return lambda x: pwr * ((x / rpm) ** 3.)

def flatten(x):
    if isinstance(x, dict):
        # pass 
        yield from [z for _, v in x.items() for z in flatten(v)]
    elif isinstance(x, list):
        # pass
        yield from [z for v in x for z in flatten(v)]
    else:
        yield x
fans = list(flatten(fanGs))
# print(fans)

def json2df(jfile):
    objs = [json.loads(line) for line in jfile]
    return pd.DataFrame(objs)

def procdf(df):
    df['irate'] = df['inst'] / df['intv']
    df['power'] = df['energy'] / (2 ** df['eunit']) / df['intv'] * 1e9
   # df['dtemp'] = df['tpkg'] - df['tpkg'].shift(1)
    for attr in appattrs:
        df[attr + '_rate'] = df[attr] / df['intv']
    df['fanpower'] = df[['%s' % x for x in fans]].apply(getFanPwr(12000.,7.)).sum(axis=1) 
#    for g in fanGs:
#        df[g] = df[fanGs[g]].mean(axis=1)
    return df

def merge2df(df1, df2, ts = 'ts'):
    fdts = lambda x: abs((df1[ts] - df2[ts].shift(x)).mean())

    # print(df1.columns, df2.columns)
    dts = fdts(0)
    idts = 0
    i = -1
    while True:
        tmp = fdts(i)
        if tmp < dts:
            dts = tmp
            idts = i
            i -= 1
            continue
        break
    i = 1
    while True:
        tmp = fdts(i)
        if tmp < dts:
            dts = tmp
            idts = i
            i += 1
            continue
        break

    df1 = df1.copy()
    df2 = df2.copy()
    df1.columns = ["%s_0" % x for x in df1.columns] 
    df2.columns = ["%s_1" % x for x in df2.columns]
    df1.rename(columns = {'fanpower_0':'fanpower'}, inplace = True)
    df2.drop('fanpower_1', axis=1, inplace=True)
  
    df = pd.concat([df1, df2.shift(idts)], axis=1).dropna()
    df['ts'] = df[['ts_0', 'ts_1']].max(axis=1)
    df['dts'] = df['ts_0'] - df['ts_1']
    
    return df

def pickapp(df, nid, sts, ets):
    ts = "ts_%d" % nid
    ists = df[ts].searchsorted(sts, side='left')
    iets = df[ts].searchsorted(ets, side='left')
    return (ists[0], iets[0])

def parseApp(f):
    sts = None
    ets = None
    start = True
    for line in f:
        if line[0] == '{':
            if start:
                sts = json.loads(line)['ts']
            else:
                ets = json.loads(line)['ts']
        elif 'finish' in line:
            start = False
    
    if sts is not None or ets is not None:
        if sts is None:
            sts = ets - 300 * 1e9
        if ets is None:
            ets = sts + 300 * 1e9
        return (sts, ets)
    else:
        return None

    return (js[0]['ts'], js[1]['ts'])

def loaddb(tag, machines):
    db = {}
    for hi  in machines:
        db[hi] = {}
        for ni in range(2):
            db[hi][ni] = {}
            fname = "%s/data/%s/run-%d/stat-node%d.log" % (homedir, tag, hi, ni)
            db[hi][ni]['df'] = procdf(json2df(open(fname, 'r')))
            for app in apps:
                fname = "%s/data/%s/run-%d/%s-node%d-stat.log" % (homedir, tag, hi, app, ni)
                db[hi][ni][app] = parseApp(open(fname, 'r'))
        db[hi]['df'] = merge2df(db[hi][0]['df'], db[hi][1]['df'])
    return db

if __name__ == '__main__':
    db = loaddb('May07', machines)
