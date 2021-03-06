import json
import pandas as pd
import os
import numpy as np
from argparse import ArgumentParser
import itertools
import glob

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
apps = ['blackscholes', 'canneal', 'ferret', 'freqmine', 'bodytrack', 'bt.C.x', 'cg.C.x', 'dc.B.x', 'ft.B.x', 'lu.C.x', 'mg.B.x', 'sp.D.x', 'ua.C.x']
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
		print(comb)
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
	print("finished loop")	
	df_bigTable = pd.DataFrame.from_records(bigTable, columns=labels)
	df_bigTable.to_csv('%s/coolr/analyzeddata/min-power-stat.csv' % (homedir))
	
def fanpwr_placement(apps, num_machines):
	#indices = range(len(apps))
	possCombs = list(itertools.combinations(apps, num_machines*2)) #each machine has two sockets
	bigTable = []
	labels = ['combination', 'opt_placement', 'pkg_power', 'perf', 'min_fan_power', 'max_fan_power', 'avg_fan_power', 'min_power', 'max_power', 'avg_power', 'worst_perf', 'best_perf']
	for comb in possCombs:
		print(comb)
		placements = list(itertools.permutations(comb))
		min_avg_pwr = 5000
		max_avg_pwr = 0
		avg_avg_pwr = 0
		min_avg_fanpwr = 5000
		max_avg_fanpwr = 0
		avg_avg_fanpwr = 0
		min_p = []
		max_p = []
		this_avg_pwr = 0
		max_perf = 0
		min_perf = 5000000
		this_perf = 0
		for p in placements:
			avg_pwr = 0
			avg_fanpwr = 0
			perf = 0
			for hi in range(1, num_machines+1):
				i = hi - 1	
				appPair = [p[2*i], p[2*i+1]]	
				app_fname1 = '%s/coolr/data/stats/%s/run-%d/coolr1-1000000-%s-%s-node0-stat.log' % (homedir, tag, hi, appPair[0], appPair[1])
				app_fname2 = '%s/coolr/data/stats/%s/run-%d/coolr1-1000000-%s-%s-node1-stat.log' % (homedir, tag, hi, appPair[0], appPair[1])
				# get first 25mins data
				df_app1 = procdf(json2df(open(app_fname1, 'r'))).iloc[600:1500]
				df_app2 = procdf(json2df(open(app_fname2, 'r'))).iloc[600:1500]
				
				# fan[hi]['%s-%s' % (appPair[0], appPair[1])] = 
				avg_pwr += np.mean(df_app1['power']) + np.mean(df_app2['power'])
				avg_fanpwr += np.mean(df_app1[['%s' % x for x in fans]].apply(getFanPwr(12000.,7.)).sum(axis=1))
				perf += appPair_perf(appPair[0], appPair[1], hi)	
			avg_avg_pwr += avg_pwr
			avg_avg_fanpwr += avg_fanpwr
			if avg_pwr < min_avg_pwr:
				min_avg_pwr = avg_pwr
			if avg_fanpwr < min_avg_fanpwr:
				min_avg_fanpwr = avg_fanpwr
				this_avg_pwr = avg_pwr
				min_p = p
				this_perf = perf
			if perf < min_perf:
				min_perf = perf

			if avg_pwr > max_avg_pwr:
				max_avg_pwr = avg_pwr
			if avg_fanpwr > max_avg_fanpwr:
				max_avg_fanpwr = avg_fanpwr
				max_p = p
			if perf > max_perf:
				max_perf = perf

		comb_str = '-'.join(comb)
		p_str = '-'.join(min_p)
		p_max_str = '-'.join(max_p)
		avg_avg_pwr = avg_avg_pwr/len(placements)
		avg_avg_fanpwr = avg_avg_fanpwr/len(placements)
		aRow = (p_max_str, p_str, this_avg_pwr, this_perf, min_avg_fanpwr, max_avg_fanpwr, avg_avg_fanpwr, min_avg_pwr, max_avg_pwr, avg_avg_pwr, min_perf, max_perf)
		bigTable.append(aRow)
	print("finished loop")
	df_bigTable = pd.DataFrame.from_records(bigTable, columns=labels)
	df_bigTable.to_csv('%s/coolr/analyzeddata/min-fanpower-stat-plc.csv' % (homedir))

def appPair_perf(app1, app2, hi):
	fn_pat1 = '%s/coolr/data/perf/%s/run-%d/%s-%s-%s*.out' % (homedir, tag, hi, app1, app2, app1)
	fn_pat2 = '%s/coolr/data/perf/%s/run-%d/%s-%s-%s*.out' % (homedir, tag, hi, app1, app2, app2)
	app_fname1 = glob.glob(fn_pat1)[0]
	app_fname2 = glob.glob(fn_pat2)[0]
	#fd1 = open(app_fname1, 'r')
	#fd2 = open(app_fname2, 'r')
	#num_lines1 = len(fd1.readlines())
	#num_lines2 = len(fd2.readlines())
	num_lines1 = int(os.popen('wc -l %s ' % app_fname1).read().split( )[0])
	num_lines2 = int(os.popen('wc -l %s' % app_fname2).read().split( )[0])
	#print app1, ': ', num_lines1,'; ', app2, ': ', num_lines2
	#fd1.close()
	#fd2.close()	
	return (num_lines1 + num_lines2)

def perf(apps, num_machines):
	perfTable = []
	labels = []
	for hi in range(1, num_machines+1):
		idx = hi - 1
		for ni in nodes:
			labels.append('S%s-N%s' % (hi, ni))
			fn_pat1 = '%s/coolr/data/perf/%s/run-%d/%s-%s-%s*.out' % (homedir, tag, hi, appPair[0], appPair[1], appPair[0])
			app_fname1 = glob.glob(fn_pat1)[0]
			num_lines = len(open(app_fname1).readlines())

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('-t', '--tag', dest='tag', action='store', help='tag', default='May18-2018')
	parser.add_argument('-b', '--benchmark', dest='benchmark', action='store', help='benchmark', default='npb')
	args = parser.parse_args()

	tag = args.tag
	benchmark = args.benchmark
	
	apps = parsec_apps
	#if benchmark == 'parsec':
	#	apps = parsec_apps

	fanpwr_placement(apps, 1)
	fan = {}
	for hi in machines:
		fan[hi] = {}
		for app1 in ["bt.C.x"]:
			for app2 in ["mg.B.x"]:
				appPair = [app1, app2]	
				app_fname1 = '%s/exp/stats/coolr1-1000000%s-%s-%s-node0-stat.log' % (homedir, tag, appPair[0], appPair[1])
				app_fname2 = '%s/exp/stats/coolr1-1000000%s-%s-%s-node1-stat.log' % (homedir, tag, appPair[0], appPair[1])
				# get first 25mins data
				df_app1 = procdf(json2df(open(app_fname1, 'r'))).iloc[0:1500]
				df_app2 = procdf(json2df(open(app_fname2, 'r'))).iloc[0:1500]
				print(df_app1.shape[0])
				# df_app = df_app1.iloc[0:1500]
				#fan[hi]['%s-%s' % (app1, app2)] 
				fan = df_app1[['%s' % x for x in fans]].apply(getFanPwr(12000.,7.)).sum(axis=1)
				print(fan[600:1500].mean())
				fan.to_csv('%s/coolr/analyzeddata/run-%d-%s-%s-%s-stat.csv' % (homedir, hi, tag, appPair[0], appPair[1]))
				
		#fan[hi].to_csv('%s/coolr/analyzeddata/run-%d-fan-stat.csv' % (homedir, hi))
