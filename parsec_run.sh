#!/bin/bash
exs=/home/cc/parsec_prebuilt

cat >run-stat.sh <<EOF
sfile=/sys/class/xstat/stat\$1
ofile=\$HOME/exp/parsecdata/coolr1-1000000-\${2}-node\${1}-stat.log

rm -f \$ofile
echo 1 > /sys/class/xstat/reset\$1
while true; do
  cat \$sfile >>\$ofile
  sleep 1s
done
EOF

chmod +x run-stat.sh

function startstat() {
  ./run-stat.sh 0 $1 &
  statpid0=$!
  ./run-stat.sh 1 $1 &
  statpid1=$!
  sleep 2s
}

function stopstat() {
  kill -9 $statpid0
  kill -9 $statpid1
  sleep 30s
}

apps="blackscholes canneal ferret freqmine bodytrack"
declare -A optmap
optmap[blackscholes]="24 /home/cc/parsec_prebuilt/in_10M.txt black_out"
optmap[canneal]="24 80000 2000 /home/cc/parsec_prebuilt/2500000.nets 300000"
optmap[ferret]="/home/cc/parsec_prebuilt/corel/ lsh /home/cc/parsec_prebuilt/queries/ 3000 40 24 ferrer_out"
optmap[freqmine]="/home/cc/parsec_prebuilt/webdocs_250k.dat 7900"
optmap[bodytrack]="/home/cc/parsec_prebuilt/sequenceB_261/ 4 261 16000 220 0 24"

for x in $apps
do
   for y in $apps
   do
       if [[ "$x" = "$y" ]];
       then
           echo "equal"
           continue
       fi
       echo "sleep"
       sleep 300s

       ts=$(date +%F-%H-%M)
       name=${x}-${y}
       logfx=/home/cc/exp/parseclog/${x}-${y}-${x}-${ts}.out
       logfy=/home/cc/exp/parseclog/${x}-${y}-${y}-${ts}.out

       startstat ${x}-${y}
       
       echo ${x}-on-node0
       echo ${y}-on-node1  
       (time numactl --cpunodebind=0 --membind=0 $exs/$x ${optmap[$x]}) > $logfx 2>&1 &
       xpid=$!

       (time numactl --cpunodebind=1 --membind=1 $exs/$y ${optmap[$y]}) > $logfy 2>&1 &
       ypid=$!

       wait $xpid
       if ps -p $ypid > /dev/null
       then
          wait $ypid
       fi

       stopstat
   done
done
