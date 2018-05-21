import json
import pandas as pd
import os
import numpy as np
from argparse import ArgumentParser
import itertools

homedir = os.environ['HOME']
#apps1 = [['bt.C.x', 'ft.B.x'], ['ft.B.x', 'ua.C.x']]
#apps1 = [['mg.B.x', 'ua.C.x'], ['bt.C.x', 'mg.B.x']] 
#apps1 = [['cg.C.x', 'ft.B.x']]
#apps1 = [['lu.C.x', 'ep.D.x']]
#machines = range(1, 2)
#machines = range(2, 3)
#apps = ['bt.C.x', 'cg.C.x', 'dc.B.x', 'ep.D.x', 'ft.B.x', 'is.C.x', 'lu.C.x', 'mg.B.x', 'sp.C.x', 'ua.C.x']
npb_apps = ['bt.C.x', 'cg.C.x', 'dc.B.x', 'ft.B.x', 'lu.C.x', 'mg.B.x', 'sp.D.x', 'ua.C.x'] 
parsec_apps = ['blackscholes', 'canneal', 'ferret', 'freqmine', 'bodytrack']

machines = range(0)
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

def pwr_placement(apps, num_machines):
	#indices = range(len(apps))
	possCombs = list(itertools.combinations(apps, num_machines*2)) #each machine has two sockets
	bigTable = []
	labels = ['combination', 'opt_placement', 'pkg_power', 'fan_power', 'min_fan_power', 'max_fan_power', 'avg_fan_power', 'max_power', 'avg_power']
	for comb in possCombs:
		print comb
		placements = list(itertools.permutations(comb))
		min_avg_pwr = 5000
		max_avg_pwr = 0
		avg_avg_pwr = 0
		min_avg_fanpwr = 5000
		max_avg_fanpwr = 0
		avg_avg_fanpwr = 0
		min_p = []
		this_avg_fanpwr = 0
		for p in placements:
			avg_pwr = 0
			avg_fanpwr = 0
			for hi in range(1, num_machines+1):
				i = hi - 1	
				appPair = [p[2*i], p[2*i+1]]	
				app_fname1 = '%s/coolr/data/stats/%s/run-%d/coolr1-1000000-%s-%s-node0-stat.log' % (homedir, tag, hi, appPair[0], appPair[1])
				app_fname2 = '%s/coolr/data/stats/%s/run-%d/coolr1-1000000-%s-%s-node1-stat.log' % (homedir, tag, hi, appPair[0], appPair[1])
				# get first 25mins data
				df_app1 = procdf(json2df(open(app_fname1, 'r'))).iloc[0:1500]
				df_app2 = procdf(json2df(open(app_fname2, 'r'))).iloc[0:1500]
				
				# fan[hi]['%s-%s' % (appPair[0], appPair[1])] = 
				avg_pwr += np.mean(df_app1['power']) + np.mean(df_app2['power'])
				avg_fanpwr += np.mean(df_app1[['%s' % x for x in fans]].apply(getFanPwr(12000.,7.)).sum(axis=1))
			avg_avg_pwr += avg_pwr
			avg_avg_fanpwr += avg_fanpwr
			if avg_pwr < min_avg_pwr:
				min_avg_pwr = avg_pwr
				this_avg_fanpwr = avg_fanpwr
				min_p = p
			if avg_fanpwr < min_avg_fanpwr:
				min_avg_fanpwr = avg_fanpwr
			if avg_pwr > max_avg_pwr:
				max_avg_pwr = avg_pwr
			if avg_fanpwr > max_avg_fanpwr:
				max_avg_fanpwr = avg_fanpwr

		comb_str = '-'.join(comb)
		p_str = '-'.join(min_p)
		avg_avg_pwr = avg_avg_pwr/len(placements)
		avg_avg_fanpwr = avg_avg_fanpwr/len(placements)
		aRow = (comb_str, p_str, min_avg_pwr, this_avg_fanpwr, min_avg_fanpwr, max_avg_fanpwr, avg_avg_fanpwr, max_avg_pwr, avg_avg_pwr)
		bigTable.append(aRow)
	print "finished loop"	
	df_bigTable = pd.DataFrame.from_records(bigTable, columns=labels)
	df_bigTable.to_csv('%s/coolr/analyzeddata/min-power-stat.csv' % (homedir))
	
def fanpwr_placement(apps, num_machines):
	#indices = range(len(apps))
	possCombs = list(itertools.combinations(apps, num_machines*2)) #each machine has two sockets
	bigTable = []
	labels = ['combination', 'opt_placement', 'pkg_power', 'min_fan_power', 'max_fan_power', 'avg_fan_power', 'min_power', 'max_power', 'avg_power']
	for comb in possCombs:
		print comb
		placements = list(itertools.permutations(comb))
		min_avg_pwr = 5000
		max_avg_pwr = 0
		avg_avg_pwr = 0
		min_avg_fanpwr = 5000
		max_avg_fanpwr = 0
		avg_avg_fanpwr = 0
		min_p = []
		this_avg_pwr = 0
		for p in placements:
			avg_pwr = 0
			avg_fanpwr = 0
			for hi in range(1, num_machines+1):
				i = hi - 1	
				appPair = [p[2*i], p[2*i+1]]	
				app_fname1 = '%s/coolr/data/stats/%s/run-%d/coolr1-1000000-%s-%s-node0-stat.log' % (homedir, tag, hi, appPair[0], appPair[1])
				app_fname2 = '%s/coolr/data/stats/%s/run-%d/coolr1-1000000-%s-%s-node1-stat.log' % (homedir, tag, hi, appPair[0], appPair[1])
				# get first 25mins data
				df_app1 = procdf(json2df(open(app_fname1, 'r'))).iloc[0:1500]
				df_app2 = procdf(json2df(open(app_fname2, 'r'))).iloc[0:1500]
				
				# fan[hi]['%s-%s' % (appPair[0], appPair[1])] = 
				avg_pwr += np.mean(df_app1['power']) + np.mean(df_app2['power'])
				avg_fanpwr += np.mean(df_app1[['%s' % x for x in fans]].apply(getFanPwr(12000.,7.)).sum(axis=1))
			avg_avg_pwr += avg_pwr
			avg_avg_fanpwr += avg_fanpwr
			if avg_pwr < min_avg_pwr:
				min_avg_pwr = avg_pwr
			if avg_fanpwr < min_avg_fanpwr:
				min_avg_fanpwr = avg_fanpwr
				this_avg_pwr = avg_pwr
				min_p = p

			if avg_pwr > max_avg_pwr:
				max_avg_pwr = avg_pwr
			if avg_fanpwr > max_avg_fanpwr:
				max_avg_fanpwr = avg_fanpwr

		comb_str = '-'.join(comb)
		p_str = '-'.join(min_p)
		avg_avg_pwr = avg_avg_pwr/len(placements)
		avg_avg_fanpwr = avg_avg_fanpwr/len(placements)
		aRow = (comb_str, p_str, this_avg_pwr, min_avg_fanpwr, max_avg_fanpwr, avg_avg_fanpwr, min_avg_pwr, max_avg_pwr, avg_avg_pwr)
		bigTable.append(aRow)
	print "finished loop"	
	df_bigTable = pd.DataFrame.from_records(bigTable, columns=labels)
	df_bigTable.to_csv('%s/coolr/analyzeddata/min-fanpower-stat.csv' % (homedir))

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('-t', '--tag', dest='tag', action='store', help='tag', default='May13-2018')
	parser.add_argument('-b', '--benchmark', dest='benchmark', action='store', help='benchmark', default='npb')
	args = parser.parse_args()

	tag = args.tag
	benchmark = args.benchmark
	
	apps = npb_apps
	if benchmark == 'parsec':
		apps = parsec_apps

	fanpwr_placement(apps, 2)
	fan = {}
	for hi in machines:
		fan[hi] = {}
		for app1 in apps:
			for app2 in apps:
				if app1 == app2:
					continue
				appPair = [app1, app2]	
				app_fname1 = '%s/coolr/data/stats/%s/run-%d/coolr1-1000000-%s-%s-node0-stat.log' % (homedir, tag, hi, appPair[0], appPair[1])
				app_fname2 = '%s/coolr/data/stats/%s/run-%d/coolr1-1000000-%s-%s-node1-stat.log' % (homedir, tag, hi, appPair[0], appPair[1])
				# get first 25mins data
				df_app1 = procdf(json2df(open(app_fname1, 'r'))).iloc[0:1500]
				df_app2 = procdf(json2df(open(app_fname2, 'r'))).iloc[0:1500]
				print df_app1.shape[0]
				# df_app = df_app1.iloc[0:1500]
				fan[hi]['%s-%s' % (app1, app2)] = df_app1[['%s' % x for x in fans]].apply(getFanPwr(12000.,7.)).sum(axis=1)
				#fan.to_csv('%s/coolr/analyzeddata/run-%d-%s-%s-stat.csv' % (homedir, hi, appPair[0], appPair[1]))
				
		fan[hi].to_csv('%s/coolr/analyzeddata/run-%d-fan-stat.csv' % (homedir, hi))
