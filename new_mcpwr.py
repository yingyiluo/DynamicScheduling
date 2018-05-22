#!/usr/bin/python3

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import multiprocessing as mp
import random as rand
import numpy.random as random
import json
import pandas as pd
import model_xgb
import model_lr
import model_svr
import model_gp
import model_mlp
import numpy as np
import os
import libdata
import datetime
# import mclearn
import getpass
from itertools import permutations
from argparse import ArgumentParser
# import matplotlib.pyplot as plt

#machines = range(1,17)
machines = range(1, 2)
#apps = ['blackscholes', 'canneal', 'ferret', 'freqmine', 'bodytrack', 'bt.C.x', 'cg.C.x', 'dc.B.x', 'ft.B.x', 'lu.C.x', 'mg.B.x', 'sp.D.x', 'ua.C.x']
apps = ['bt.C.x', 'cg.C.x', 'dc.B.x', 'ft.B.x', 'lu.C.x', 'mg.B.x', 'sp.D.x', 'ua.C.x']
homedir = os.environ['HOME']
#apps_validation = ['bt.C.x', 'cg.C.x']

def stdWorkloads(workloads):
    if isinstance(workloads, dict):
        for hi, apps in iter(workloads.items()):
            yield apps
    elif isinstance(workloads, (list, np.ndarray)):
        i = 0
        while i * 2 + 1 < len(workloads):
            yield (workloads[i * 2], workloads[i * 2 + 1])
            i += 1

tgtlist = ["power_0", "power_1", "fanpower"]
applist = ["%s_%d" % (x, i) for x in libdata.apprates for i in [0,1]]

def getdata(apphist, tgthist, dt):
    dfapp = pd.DataFrame(apphist)[applist]
    try:
        dftgt = pd.DataFrame(tgthist)[tgtlist]
    except Exception as e:
        print(tgthist, tgtlist)
        raise(e)

    dfapp['idx'] = range(len(dfapp))
    dfapp = dfapp.set_index('idx')
    dftgt['idx'] = range(len(dftgt))
    dftgt = dftgt.set_index('idx')
    dftgt = dftgt.reindex(dfapp.index)

    df = dfapp
    #print(df.columns)
    for t in range(1, dt+1):
        for sdf in [dfapp, dftgt]:
            shifted = sdf.shift(t)
            shifted.columns = ["%s_%d" % (x, t) for x in sdf.columns]
            df = pd.concat([df, shifted], axis=1)
    # print(df, dftgt.shift(1))
    # print(len(df.dropna()), len(dfapp), len(dftgt))

    return df.dropna()

def evolve(model, df, dfapp, dt):
    apphist = []
    phyhist = []
    dfapp_len = len(dfapp)-1
#    est1 = fanest.FSMFanEst(df.iloc[dt],key='tpkg_0',fankey='FanGroup1_0')
#    est2 = fanest.FSMFanEst(df.iloc[dt],key='tpkg_1',fankey='FanGroup3_0')
    for t in range(0, dt):
       # r = rand.randint(0, dfapp_len)
        r = t % dfapp_len
        apphist.append(dfapp[applist].iloc[r])
        phyhist.append(df[tgtlist].iloc[t])
    for i in range(dt, len(df)):
      #  r = rand.randint(0, dfapp_len)
        r = i % dfapp_len
        apphist.append(dfapp[applist].iloc[r])
        tsample = getdata(apphist[-dt-1:], phyhist[-dt:], dt).iloc[-1]
        ttgt = model.predict(tsample.values.reshape(1,-1))
        # fix ttgt = ttgt' + dtemp
        # ttgt[1] = phyhist[-1][1] + ttgt[0]
        # print phyhist[-1], ttgt
        ttgt = pd.DataFrame(data=ttgt, columns = tgtlist).iloc[0]
#        fan1est = est1.estFanTarget(ttgt)
#        fan2est = est2.estFanTarget(ttgt)

#        ttgt['FanGroup1_0'] = est1.updateTarget(fan1est)
#        ttgt['FanGroup3_0'] = est2.updateTarget(fan2est)

#        ttgt['tpkg_0'] = ttgt['dtemp_0'] + phyhist[-1]['tpkg_0']
#        ttgt['tpkg_1'] = ttgt['dtemp_1'] + phyhist[-1]['tpkg_1']

        # print ttgt.values.shape
        phyhist.append(ttgt)
    return pd.DataFrame(phyhist)

def pickdf(hi, app0, app1):
    s0, e0 = db[hi][0][app0][app1]
    s1, e1 = db[hi][1][app0][app1]
    s0, e0 = libdata.pickapp(db[hi]['df'], 0, s0, e0)
    s1, e1 = libdata.pickapp(db[hi]['df'], 1, s1, e1)
    sts = max(s0, s1)
    ets = min(e0, e1)
    # print(app0, app1, ets - sts)
    return db[hi]['df'][sts:ets]

def naiveOpt(workloads):
    return workloads

def mcOpt(workloads):
    size = len(workloads)
    apps = list(set(flatten(workloads)))

def _pair_predict(model, hi, app0, app1, dt = 1, cache = {}):
    if (hi, app0, app1) in cache:
        return cache[hi, app0, app1]
    train = pickdf(hi, app0, app1)[300:300+100]
    phyhist = evolve(model, train, dt)

    cache[hi, app0, app1] = phyhist[['power_0', 'power_1']].mean(axis=0)
    return cache[hi, app0, app1]

def trainModel(arg):
    model, hi, nTrain, dt, apps_train = arg 
    app0 = apps_train[0]
    app1 = apps_train[1] 
#    for app0 in apps_train:
#        for app1 in apps_train:
#            if app0 == app1:
#                continue
#            df = pickdf(hi, app0, app1)[:nTrain]
    if True:
        if True:
            df = db[hi]['%s-%s' % (app0, app1)][:nTrain]
            
            #print(df.columns)
            #print(df.shape)
            data = getdata(df, df, dt)
            data.to_csv('traindata.csv')
            #print('data shape')
            #print(data.shape)
            target = df[dt:][tgtlist]
           # print(len(data))
            #print(data.columns.values)
            model.train(data.values, target.values)
    model.fit()
    return model.clean()

def pairOpt(workloads):
    nStep = 1000
    nTry = 10
    dt = 1
    machines = range(1, int(len(workloads)/2) + 1)
    idx = lambda hi, ni: ni + hi * len(machines)
    print(len(machines))
    ridx = lambda x : (x // len(machines), x % len(machines))
    models = {}
    param = {
        'loss': 'ls',
        'learning_rate': .1,
        'alpha': 0.9,
        'max_depth': 4,
        'n_estimators': 100,
        'm': 10000,
    }
    perflog = {
        'realized'  : [],
        'predicted' : [],
    }
    totrain = []
    
    for hi in machines:
        model = model_xgb.XGBoost()
        model.init(**param)
        totrain.append((model, hi, nTrain_new, dt, libdata.apps))

    lmodels = pool.map(trainModel, totrain)
    for i in range(len(lmodels)):
        models[i+1] = lmodels[i]

    cache = {}
    stdwl = list(stdWorkloads(workloads))
    table = []
    for i in range(len(stdwl)):
        hi = i + 1
        app0, app1 = stdwl[i]
        p = _pair_predict(models[hi], hi, app0, app1, cache=cache, dt=dt)
        table.append({'power':p['power_0'], 'app':app0})
        table.append({'power':p['power_1'], 'app':app1})
    tableidx = [idx(hi, ni) for hi in machines for ni in range(2)]
    table = pd.DataFrame(table, index = tableidx)

    while nStep > 0:
        nStep -= 1
        perflog['realized'].append(oneRun([table['app'].loc[idx(hi, ni)] for hi in machines for ni in range(2)])['pkgpwr'])
        perflog['predicted'].append(table['power'].sum())
        tsorted = table.sort_values(by='power', ascending=False)
        # print(tsorted)
        ok = False
        for i, j in [(k, len(tsorted) - l - 1) for k in range(nTry) for l in range(nTry)]:
            h0, n0 = ridx(tsorted.index[i])
            h1, n1 = ridx(tsorted.index[j])
            if tsorted.loc[idx(h0,n0), 'app'] == tsorted.loc[idx(h1,n1), 'app']:
                continue
            #print(tsorted.columns.values)
            pbefore = tsorted['power'].loc[[idx(h0, n0),idx(h0,1-n0),idx(h1,n1),idx(h1,1-n1)]].sum()
            a00, a01 = tsorted.loc[idx(h0, 0), 'app'], tsorted.loc[idx(h0,1), 'app']
            a10, a11 = tsorted.loc[idx(h1, 0), 'app'], tsorted.loc[idx(h1,1), 'app']
            a0before = [a00, a01]
            a1before = [a10, a11]
            a0after = a0before[:]
            a1after = a1before[:]
            a0after[n0] = a1before[n1]
            a1after[n1] = a0before[n0]
            if h0 == h1:
                a0after[n1] = a1after[n1]
                a1after[n0] = a0after[n0]
            # print(a0before, a0after, a1before, a1after)
            pa0, pa1 = _pair_predict(models[h0], h0, a0after[0], a0after[1], cache=cache, dt=dt)
            pb0, pb1 = _pair_predict(models[h1], h1, a1after[0], a1after[1], cache=cache, dt=dt)
            if pa0 + pa1 + pb0 + pb1 < pbefore:
                tsorted.set_value(idx(h0, 0), 'app', a0after[0])
                tsorted.set_value(idx(h0, 1), 'app', a0after[1])
                tsorted.set_value(idx(h1, 0), 'app', a1after[0])
                tsorted.set_value(idx(h1, 1), 'app', a1after[1])
                tsorted.set_value(idx(h0, 0), 'power', pa0)
                tsorted.set_value(idx(h0, 1), 'power', pa1)
                tsorted.set_value(idx(h1, 0), 'power', pb0)
                tsorted.set_value(idx(h1, 1), 'power', pb1)
                ok = True
                break
        if not ok:
            break
        table = tsorted
    perflog = pd.DataFrame(perflog)
    fig, ax = plt.subplots()
    ax.plot(perflog.index, perflog['realized'], 'r')
    ax.plot(perflog.index, perflog['predicted'], 'b')
    fig.suptitle(','.join(workloads))
    #fig.savefig(optpdf, format='pdf')
    return [table['app'].loc[idx(hi, ni)] for hi in machines for ni in range(2)]

def _pair_predict_fan(model, hi, app0, app1, dt = 1, cache = {}):
    if (hi, app0, app1) in cache:
        return cache[hi, app0, app1]
    train = pickdf(hi, app0, app1)[300:300+100]
    phyhist = evolve(model, train, dt)

    cache[hi, app0, app1] = np.mean(phyhist[['%s_0' % x for x in libdata.fans]].apply(getFanPwr(12000., 7.)).sum(axis = 1))
    return cache[hi, app0, app1]

def pairOptFan(workloads):
    nTrain = 3000
    nStep = 1000
    nTry = 6
    machines = range(1, int(len(workloads)/2) + 1)
    idx = lambda hi, ni: ni + hi * len(machines)
    ridx = lambda x : (x // len(machines), x % len(machines))
    perflog = {
        'realized'  : [],
        'predicted' : [],
    }

    cache = {}
    stdwl = list(stdWorkloads(workloads))
    table = []
    for i in range(len(stdwl)):
        hi = i + 1
        app0, app1 = stdwl[i]
        p = _pair_predict_fan(models[hi], hi, app0, app1, cache=cache, dt=dt)
        table.append({'power':p, 'app0':app0, 'app1':app1})
    tableidx = [hi for hi in machines]
    table = pd.DataFrame(table, index = tableidx)

    while nStep > 0:
        nStep -= 1
        perflog['realized'].append(oneRun([table[app].loc[hi] for hi in machines for app in ['app0', 'app1']])['fanpwr'])
        perflog['predicted'].append(table['power'].sum())
        tsorted = table.sort_values(by='power', ascending=False)
        # print(tsorted)
        ok = False
        for i, j in [(k, len(tsorted) - l - 1) for k in range(nTry) for l in range(nTry)]:
            h0 = tsorted.index[i]
            h1 = tsorted.index[j]
            pbefore = tsorted['power'].loc[[h0, h1]].sum()
            appset = [tsorted[app].loc[hi] for hi in [h0, h1] for app in ['app0', 'app1']]
            for totest in permutations(appset):
                a00, a01, a10, a11 = totest
                pa = _pair_predict_fan(models[h0], h0, a00, a01, cache=cache, dt=dt)
                pb = _pair_predict_fan(models[h1], h1, a10, a11, cache=cache, dt=dt)
                if pa + pb < pbefore:
                    tsorted.set_value(h0, 'app0', a00)
                    tsorted.set_value(h0, 'app1', a01)
                    tsorted.set_value(h0, 'power', pa)
                    tsorted.set_value(h1, 'app0', a10)
                    tsorted.set_value(h1, 'app1', a11)
                    tsorted.set_value(h1, 'power', pb)
                    ok = True
                    break
            if ok:
                break
        if not ok:
            break
        table = tsorted
    perflog = pd.DataFrame(perflog)
    fig, ax = plt.subplots()
    ax.plot(perflog.index, perflog['realized'])
    ax.plot(perflog.index, perflog['predicted'])
    fig.suptitle(','.join(workloads))
   # fig.savefig(optpdffan, format='pdf')
    return [table[app].loc[hi] for hi in machines for app in ['app0', 'app1']]

def optWorkloads(workloads):
    # return naiveOpt(workloads)
    return pairOpt(workloads)
    #return pairOptFan(workloads)

def getdfs(workloads):
    if isinstance(workloads, dict):
        for hi, apps in iter(workloads.items()):
            app0, app1 = apps
            yield pickdf(hi, app0, app1)
    elif isinstance(workloads, (list, np.ndarray)):
        i = 0
        while i * 2 + 1 < len(workloads):
            yield pickdf(i + 1, workloads[i * 2], workloads[i * 2 + 1])
            i += 1

def aggr(series, method):
    if method == 'max':
        return series.max(axis=0)
    elif method == 'mean':
        return series.mean(axis=0)
    else:
        # sum is the default
        return series.sum(axis=0)

def oneRun(workloads):
    res = [{k : func['func'](df) for k, func in iter(funcs.items())} for df in getdfs(workloads)]
    res = pd.DataFrame(res)
    # print(funcs, workloads, res)
    res = [ {k : aggr(res[k], func['aggr']) for k, func in iter(funcs.items())} ]
    # print(pd.DataFrame(res).iloc[0])
    return pd.DataFrame(res).iloc[0]

def mcRuns(workloads, nRuns = 1000):
    # onerun = lambda x: oneRun(x, funcs)
    works = [random.permutation(workloads) for i in range(nRuns)]
    df = pd.DataFrame(pool.map(oneRun, works))
    res = pd.DataFrame([df.mean(axis=0), df.max(axis=0), df.min(axis=0), df.std(axis=0)], \
                       index = ['mean', 'max', 'min', 'std'])
    return res, works

def getFanPwr(rpm, pwr):
    return lambda x: pwr * ((x / rpm) ** 3.)

def evalAccuracy(nTests = 1000, targets = ['fanpower', 'power_0', 'power_1'], apps_validation = []):
    app0 = apps_validation[0]
    app1 = apps_validation[1]
    err = { j + 1 : { x : [] for x in targets } for j in range(len(machines)) }
    for i in range(nTests):
        for j in range(len(machines)):
            hi = j + 1 
            df = db[hi]['%s-%s' % (app0, app1)][nTrain:nTrain+interval]
            dfapp = db[hi]['%s-%s' % (app0, app1)][nTrain-120:nTrain]
            phyhist = evolve(models[hi], df, dfapp, dt)
            for x in targets:
                #print('phyhist')
                #print(phyhist[x])
                #print('df')
                #print(df[x])
                #print(phyhist[x].values - df[x].values)
                err[hi][x].append(np.mean(np.abs(phyhist[x].values - df[x].values)))
    res = { j + 1 : { x : np.mean(err[j + 1][x]) for x in targets } for j in range(len(machines)) }
    #print(res)
    #totres = np.mean(list(flatten(res)))
    totres = np.mean([np.mean(list(e.values())) for e in list(res.values())])
    #print(totres)
    prediction = pd.concat([db[1]['%s-%s' % (app0, app1)][targets][:nTrain], phyhist], sort='True')
    actual = db[hi]['%s-%s' % (app0, app1)][:nTrain+interval]
    for t in targets:
        plt.figure(figsize = (10,2))
        plt.plot([x/60.0 for x in range(1, nTrain+interval+1)], prediction['%s' % t], 'r', label='predict')
        plt.plot([x/60.0 for x in range(1, nTrain+interval+1)], actual['%s' % t], 'b', label='actual')
        plt.legend(loc=1)
        print('%s/coolr/prediction/run-%d-%s-%s-%s.png' % (homedir, hi, t, app0, app1))
        plt.savefig('%s/coolr/prediction/run-%d-%s-%s-%s.png' % (homedir, hi, t, app0, app1))
        plt.close()
    return res

def testPop(nTests = 1000):
    df = []
    optcmp = []
    for i in range(nTests):
        workloads = random.choice(libdata.apps, size=len(machines) * 2)

        optwl = optWorkloads(workloads)
        optres = oneRun(optwl)

        if i % 10 == 0:
            print("%s\t%d\t%s" % (str(datetime.datetime.now()), i, str(workloads)))
        res, works = mcRuns(workloads, nRuns = NRUNS)
        df.append(res)
        optcmp.append(optres - res.loc['mean'])

        # sanity checks
        if optres['pkgpwr'] < res.loc['min', 'pkgpwr']:
            # Unlikely, check
            print(workloads, optwl, optres['pkgpwr'], res.loc['min', 'pkgpwr'])

            # for i in len(works):
            #    if all([optwl[j] == works[i][j] for j in len(optwl)]):
            #        print('Found identical MC run')
            #        print(res.iloc(i))
            #        break

    # df = [mcRuns(random.choice(libdata.apps, size=len(machines) * 2)) for _ in range(nTests)]
    dfmean = pd.DataFrame([x.loc['mean'] for x in df])
    # print(dfmean)
    gdf = pd.DataFrame([x.loc['max'] / x.loc['mean'] - 1. for x in df])
    df = pd.DataFrame([1 - x.loc['min'] / x.loc['mean'] for x in df])
    # print(df)
    res = pd.DataFrame([dfmean.mean(axis=0), df.mean(axis=0), df.max(axis=0), df.min(axis=0), df.std(axis=0), gdf.mean(axis=0), gdf.max(axis=0), gdf.min(axis=0), gdf.std(axis=0)], \
                       index = ['valmean', 'min_mean', 'min_max', 'min_min', 'min_std', 'max_mean', 'max_max', 'max_min', 'max_std'])
    optcmp = pd.DataFrame(optcmp)
    optres = pd.DataFrame([optcmp.mean(axis=0), optcmp.max(axis=0), optcmp.min(axis=0), optcmp.std(axis=0)], \
                       index = ['mean', 'max', 'min', 'std'])
    return res, optres

def testDual():
    idx = []
    df = []
    for i in range(len(libdata.apps) - 1):
        for j in range(i + 1, len(libdata.apps)):
            print('Running %s %s' % (libdata.apps[i], libdata.apps[j]))
            idx.append((libdata.apps[i], libdata.apps[j]))
            workloads = [libdata.apps[i]] * len(machines) + [libdata.apps[j]] * len(machines)
            df.append(mcRuns(workloads, nRuns = NRUNS))
    dfmean = pd.DataFrame([x.loc['mean'] for x, works in df], index = idx)
    dfdiff = pd.DataFrame([x.loc['mean'] - x.loc['min'] for x, works in df], index = idx)
    # print(dfmean)
    # print(dfdiff)
    gdf = pd.DataFrame([x.loc['max'] / x.loc['mean'] - 1. for x, works in df])
    df = pd.DataFrame([1 - x.loc['min'] / x.loc['mean'] for x, works in df])
    res = pd.DataFrame([dfmean.mean(axis=0), df.mean(axis=0), df.max(axis=0), df.min(axis=0), df.std(axis=0), gdf.mean(axis=0), gdf.max(axis=0), gdf.min(axis=0), gdf.std(axis=0)], \
                       index = ['valmean', 'min_mean', 'min_max', 'min_min', 'min_std', 'max_mean', 'max_max', 'max_min', 'max_std'])
    return res

def logperf(x):
    print(x['inst_rate_0'][offset:])
    print(np.log(1. / x['inst_rate_0'][offset:].values.astype(float)))
    return np.sum(np.log(1. / x['inst_rate_0'][offset:].values.astype(float)) + np.log(1. / x['inst_rate_1'][offset:].values.astype(float)))

offset = -120
endoffset = -10
targets = {
    'maxfanpwr' : { 'aggr' : 'max', 'func' : lambda x : np.max(x[['%s_0' % x for x in libdata.fans]].apply(getFanPwr(12000., 7.)).max(axis = 1)) },
    'maxpkgpwr' : { 'aggr' : 'max', 'func' : lambda x : x[['power_0', 'power_1']][offset:endoffset].max(axis=1).sort_values()[-10:].mean() },
    'pkgpwr'    : { 'aggr' : 'sum', 'func' : lambda x : np.mean(x['power_0'][offset:endoffset] + x['power_1'][offset:endoffset]) },
    'fanpwr'    : { 'aggr' : 'sum', 'func' : lambda x : np.mean(x[['%s_0' % x for x in libdata.fans]][offset:endoffset].apply(getFanPwr(12000., 7.)).sum(axis = 1)) },
    'logperf'   : { 'aggr' : 'sum', 'func' : lambda x : np.sum(np.log(x['inst_rate_0'][offset:endoffset].values.astype(float)) + np.log(x['inst_rate_1'][offset:endoffset].values.astype(float))) },
    'instperf'  : { 'aggr' : 'mean', 'func' : lambda x : np.mean(x['inst_rate_0'][offset:endoffset] + x['inst_rate_1'][offset:endoffset]) },
}
targets['pkg+fan pwr'] = { 'aggr' : 'sum', 'func' : lambda x: targets['pkgpwr']['func'](x) + targets['fanpwr']['func'](x) }
funcs = targets

# print(oneRun({1:('bt.C.x', 'cg.C.x'), 2:('bt.C.x', 'cg.C.x')}, targets))

# MC parameters
NRUNS = 600
NTESTS = 10

dt = 1

if __name__ == '__main__':
    rand.seed(0)
    parser = ArgumentParser()
    parser.add_argument('-ml', '--model', dest='ml_method', action='store', help='ml method', default='xgb')
    parser.add_argument('-l', '--loss', dest='loss', action='store', help='loss function', default='quantile')
    parser.add_argument('-a', '--alpha', dest='alpha', type=float, action='store', help='alpha', default=0.9)
    parser.add_argument('-r', '--learning_rate', dest='learning_rate', type=float, action='store', help='learning_rate', default=0.1)
    parser.add_argument('-d', '--depth', dest='depth', type=int, action='store', help='max depth', default=5)
    parser.add_argument('-t', '--tag', dest='tag', action='store', help='tag', default='May13-2018')
    parser.add_argument('-n', '--n_estimators', dest='n_estimators', type=int, action='store', help='n estimators', default=100)
    parser.add_argument('-m', '--max_samples', dest='max_samples', type=int, action='store', help='max samples', default=5000)
    parser.add_argument('-o', '--optimizer', dest='opt', action='store', default='naive')
    parser.add_argument('-nt', '--nTrain', dest='nTrain', type=int, action='store', help='n trains', default=600)
    parser.add_argument('-i', '--interval', dest='interval', type=int, action='store', help='internal', default=300)
    args = parser.parse_args()

    tag = args.tag
    nTrain = args.nTrain
    interval = args.interval
    ml_method = args.ml_method

    #app0 = 'ft.B.x'
    #app1 = 'dc.B.x'
    db = {}
    for hi in machines:
        db[hi] = {}
        for app0 in apps:
            for app1 in apps:
#                if app0 == app1:
#                    continue
                for ni in range(2):
                    db[hi][ni] = {}
                    dbni = db[hi][ni]
                    if ni == 0:
                        appx = app0
                    if ni == 1:
                        appx = app1
                    fname = '%s/coolr/data/stats/%s/run-%d/coolr1-1000000-%s-%s-node%d-stat.log' % (homedir, tag, hi, app0, app1, ni)
                    dbni['df'] = libdata.procdf(libdata.json2df(open(fname, 'r')))

                db[hi]['%s-%s' % (app0, app1)] = libdata.merge2df(db[hi][0]['df'], db[hi][1]['df'])
      #  db[hi]['bt.C.x-ft.B.x'].to_csv('test.csv')
    
    models = {}
    params = {
        "loss": "quantile",
        "learning_rate": 0.1,
		"n_estimators": 100,
		"max_depth": 3,
		"min_samples_split": 2,
		"min_samples_leaf": 1,
		"min_weight_fraction_leaf": 0.,
		"subsample": 1.,
		"max_features": "sqrt",
		"max_leaf_nodes": None,
		"alpha": 0.9,
		"init": None,
		"verbose": 0,
		"warm_start": False,
		"random_state": 0,
		"presort": "auto",
    }
    if ml_method == 'lr':
        params = {
            'fit_intercept': True,
            'normalize': True,
            'm': args.max_samples,
        }
    elif ml_method == 'svr':
        params = {
            'm': args.max_samples,
        }
    elif ml_method == 'gp':
        params = {
            'n_restarts_optimizer': 0,
            'm': args.max_samples,
        }
    elif ml_method == 'mlp':
        params = {
            'hidden_layer_sizes': (100, ),
            'activation': 'relu',
            'solver': 'adam',
            'max_iter': 1000,
            'm': args.max_samples,
        }

    for eval_times in range(1):
        allres = [] 
        for app0 in apps:
            for app1 in apps:  
               # if app0 == app1:
               #     continue
                pool = mp.Pool(mp.cpu_count()) 
                appPair = [app0, app1]
                print(appPair)
                totrain = []
                for hi in machines:
                    if ml_method == 'xgb':
                        model = model_xgb.XGBoost()
                    if ml_method == 'lr':
                        model = model_lr.LR()
                    if ml_method == 'svr':
                        model = model_svr.SVM()
                    if ml_method == 'gp':
                        model = model_gp.GPR()
                    if ml_method == 'mlp':
                        model = model_mlp.MLP()
                    model.init(**params)
                    totrain.append((model, hi, nTrain, dt, appPair))

                lmodels = pool.map(trainModel, totrain)
                for i in range(len(lmodels)):
                    models[i+1] = lmodels[i]
                pool.close()
                pool.join()
                res = evalAccuracy(nTests = 1, apps_validation = appPair)
                allres.append(res)
        errs = {}
        for hi in machines:
            errs[hi] = {}
            errs[hi]['fanpower'] = 0
            errs[hi]['power_0'] = 0
            errs[hi]['power_1'] = 0
        for res in allres:
             for hi in machines:
                # print(res[hi])
                 errs[hi]['fanpower'] += res[hi]['fanpower']
                 errs[hi]['power_0'] += res[hi]['power_0']
                 errs[hi]['power_1'] += res[hi]['power_1']
        num_appPairs = len(allres)
        print(num_appPairs)
        errs = { hi : {k : v/num_appPairs for k, v in errs[hi].items()} for hi in machines}
        print(errs)
#    pool = mp.Pool(mp.cpu_count())
#    print(testPop(nTests = NTESTS))
#    # print(testDual())
#    pool.close()
#    pool.join()
