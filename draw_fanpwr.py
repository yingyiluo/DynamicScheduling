import json
import pandas as pd
#import libdata
import os
import numpy as np

#homedir = '/home/cc'
tag = 'May8-2018'
csvdir = os.environ['HOME']
#apps1 = [['bt.C.x', 'ft.B.x'], ['ft.B.x', 'ua.C.x']]
#apps1 = [['mg.B.x', 'ua.C.x'], ['bt.C.x', 'mg.B.x']] 
#apps1 = [['cg.C.x', 'ft.B.x']]
apps1 = [['lu.C.x', 'ep.D.x']]
#machines = range(1, 2)
machines = range(2, 3)
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
		for appPair in apps1:
			app_fname1 = '%s/exp/data/%s/run-%d/coolr1-1000000-%s-%s-node0-stat.log' % (csvdir, tag, hi, appPair[0], appPair[1])
			app_fname2 = '%s/exp/data/%s/run-%d/coolr1-1000000-%s-%s-node1-stat.log' % (csvdir, tag, hi, appPair[0], appPair[1])
			df_app1 = procdf(json2df(open(app_fname1, 'r')))
			df_app2 = procdf(json2df(open(app_fname2, 'r')))
			#if df_app1.shape[0] < df_app2.shape[0]:
			#	df_app = df_app2
			#else:
			#	df_app = df_app1
			print df_app1.shape[0]
			# get the data for the first half hour
			df_app = df_app1.iloc[0:1200]
			fan = df_app[['%s' % x for x in fans]].apply(getFanPwr(12000.,7.)).sum(axis=1)
			#time = df_app['ts']
			#frames = [fan, time]
			#result = pd.concat(frames)
			fan.to_csv('/home/cc/exp/fandata/run-%d-%s-%s-stat.csv' % (hi, appPair[0], appPair[1]))
