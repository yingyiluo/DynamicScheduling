import json
import pandas as pd
#import libdata
import os
import numpy as np
import matplotlib.pyplot as plt
import os.path

tag = 'May13-2018'
csvdir = os.environ['HOME']
apps = ['bt.C.x', 'cg.C.x', 'dc.B.x', 'ft.B.x', 'lu.C.x', 'mg.B.x', 'sp.D.x', 'ua.C.x']
machines = range(1, 2)
nodes = range(0, 2)
fans = ['Fan1A', 'Fan1B', 'Fan2A', 'Fan2B', 'Fan3A', 'Fan3B', 'Fan4A', 'Fan4B', 'Fan5A', 'Fan5B', 'Fan6A', 'Fan6B', 'Fan7A', 'Fan7B']
appattrs = ['cyc', 'inst', 'llcref', 'llcmiss', 'br', 'brmiss', 'l2lin']
fanGs = { 'FanGroup1' : ['Fan1A', 'Fan1B', 'Fan2A', 'Fan2B', 'Fan3A', 'Fan3B'], 'FanGroup2' : ['Fan4A', 'Fan4B'], 'FanGroup3' : ['Fan5A', 'Fan5B', 'Fan6A', 'Fan6B', 'Fan7A', 'Fan7B'] }

def getFanPwr(rpm, pwr):
	return lambda x: pwr * ((x / rpm) ** 3.)

def json2df(jfile):
	objs = [json.loads(line) for line in jfile]
	return pd.DataFrame(objs)

def procdf(df):
	df['irate'] = df['inst'] / df['intv']
	df['power'] = df['energy'] / (2 ** df['eunit']) / df['intv'] * 1e9
	df['dtemp'] = df['tpkg'] - df['tpkg'].shift(1)
	for attr in appattrs:
		df[attr + '_rate'] = df[attr] / df['intv']
	for g in fanGs:
		df[g] = df[fanGs[g]].mean(axis=1)
	return df

if __name__ == '__main__':
	for hi in machines:
		for app1 in apps:
			for app2 in apps:
				if app1 == app2:
					continue
				appPair = [app1, app2]
				app_fname1 = '%s/coolr/data/stats/%s/run-%d/coolr1-1000000-%s-%s-node0-stat.log' % (csvdir, tag, hi, appPair[0], appPair[1])
				app_fname2 = '%s/coolr/data/stats/%s/run-%d/coolr1-1000000-%s-%s-node1-stat.log' % (csvdir, tag, hi, appPair[0], appPair[1])
				df_app1 = procdf(json2df(open(app_fname1, 'r'))).iloc[0:1500]
				df_app2 = procdf(json2df(open(app_fname2, 'r'))).iloc[0:1500]
			#if df_app1.shape[0] < df_app2.shape[0]:
			#	df_app = df_app2
			#else:
			#	df_app = df_app1
			# get the data for the first half hour
				plt.plot([x/60.0 for x in range(1, 1501)], df_app1['tpkg'], 'r', label=app1)
				plt.plot([x/60.0 for x in range(1, 1501)], df_app2['tpkg'], 'b', label=app2)
				plt.legend(loc=1)
				plt.savefig('%s/coolr/analyzeddata/temp/run-%d-temp-%s-%s.png' % (csvdir, hi, app1, app2))
				plt.close()
				#df_app1.to_csv('/home/cc/exp/csvdata/run-%d-%s-%s-node0-stat.csv' % (hi, appPair[0], appPair[1]))
				#df_app2.to_csv('/home/cc/exp/csvdata/run-%d-%s-%s-node1-stat.csv' % (hi, appPair[0], appPair[1]))
