#!/bin/bash
exs=/home/cc/parsec_prebuilt

cat >run-stat.sh <<EOF
sfile=/sys/class/xstat/stat\$1
ofile=\$HOME/exp/stats/coolr1-1000000-\${2}-node\${1}-stat.log

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

function contains() {
  local list="$1"
  local item="$2"
  if [[ $list =~ (^|[[:space:]])"$item"($|[[:space:]]) ]] ; then
    # yes, list include item
    result=0
    #echo "the same"
  else
    result=1
  fi
  return $result
}

apps="blackscholes canneal ferret freqmine bodytrack bt.C.x cg.C.x ft.B.x sp.D.x mg.B.x ua.C.x lu.C.x dc.B.x"
parsec_apps="blackscholes canneal ferret freqmine bodytrack"
nbp_apps="bt.C.x cg.C.x ft.B.x sp.D.x mg.B.x ua.C.x lu.C.x dc.B.x"
declare -A optmap
optmap[blackscholes]="24 /home/cc/parsec_prebuilt/in_10M.txt black_out"
optmap[canneal]="24 80000 2000 /home/cc/parsec_prebuilt/2500000.nets 300000"
optmap[ferret]="/home/cc/parsec_prebuilt/corel/ lsh /home/cc/parsec_prebuilt/queries/ 3000 40 24 ferrer_out"
optmap[freqmine]="/home/cc/parsec_prebuilt/webdocs_250k.dat 7200"
optmap[bodytrack]="/home/cc/parsec_prebuilt/sequenceB_261/ 4 261 16000 220 0 24"
optmap[bt.C.x]="</dev/null"
optmap[cg.C.x]="</dev/null"
optmap[ft.B.x]="</dev/null"
optmap[sp.D.x]="</dev/null"
optmap[mg.B.x]="</dev/null"
optmap[ua.C.x]="</dev/null"
optmap[lu.C.x]="</dev/null"
optmap[dc.B.x]="</dev/null"

for x in $apps
do
   for y in $apps
   do
#       if [[ "$x" = "$y" ]];
#       then
#           echo "equal"
#           continue
#       fi
       echo $x-$y
       if contains "$parsec_apps" "$x" ;
       then 
       #    echo 'Yes'
           if contains "$parsec_apps" "$y" ;
           then
        #       echo 'Yes'
               if [[ "$x" != "$y" ]];
               then
                   echo 'Continue' 
                   continue
               fi
           fi
        elif contains "$nbp_apps" "$x" ;
        then
           if contains "$nbp_apps" "$y";
           then
               if [[ "$x" != "$y" ]];
               then
                   continue 
               fi
           fi  
        fi        
       echo "sleep"
       sleep 300s
       
       ts=$(date +%F-%H-%M)
       name=${x}-${y}
       logfx=/home/cc/exp/perf/${x}-${y}-${x}-${ts}.out
       logfy=/home/cc/exp/perf/${x}-${y}-${y}-${ts}.out
       errfx=/home/cc/exp/err/${x}-${y}-${x}-${ts}.err
       errfy=/home/cc/exp/err/${x}-${y}-${y}-${ts}.err
      
       startstat ${x}-${y}
       
       echo ${x}-on-node0
       echo ${y}-on-node1  
       numactl --cpunodebind=0 --membind=0 $exs/$x ${optmap[$x]} 1>$logfx 2>$errfx &
       xpid=$!

       numactl --cpunodebind=1 --membind=1 $exs/$y ${optmap[$y]} 1>$logfy 2>$errfy &
       ypid=$!

       sleep 1560s
       if ps -p $xpid > /dev/null
       then
          kill -9 $xpid
       else 
          echo ${x}-${ts}
       fi
       if ps -p $ypid > /dev/null
       then
          kill -9 $ypid
       else
          echo ${y}-${ts}
       fi

       stopstat
   done
done
