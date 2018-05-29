#!/bin/bash

x="2 60 120 180 240 300 360"
model="xgb lr svr gp mlp"
model="mlp"
for m in $model
do
  for i in $x
  do
    echo ${i} >> evaluate_acc/${m}.log
    python3.6 correct_mcpwr.py -ml $m -i $i >> evaluate_acc/${m}.log 
  done
done
