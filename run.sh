#!/bin/bash

apps="bt.C.x cg.C.x ft.B.x sp.D.x mg.B.x ua.C.x lu.C.x"

for x in $apps
do
	for y in $apps
	do
		if [[ "$x" = "$y" ]];
		then
			echo "equal"
			continue
		fi
		#echo "not equal"
   		python exp1.py $x $y
		sleep 300s
	done
done
