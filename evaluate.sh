#!/bin/bash
nspk=`LC_ALL=es_ES.iso cut -d' ' -f 1 $1 | sort -u | wc -l`
a=`LC_ALL=es_ES.iso cut -d' ' -f 6,7 $1 | grep "1 0" | wc -l`
b=`LC_ALL=es_ES.iso cut -d' ' -f 6,7 $1 | grep "0 1" | wc -l`
c=`LC_ALL=es_ES.iso cut -d' ' -f 6 $1 | grep "0" | wc -l`
d=`LC_ALL=es_ES.iso cut -d' ' -f 6 $1 | grep "1" | wc -l`
r=`bc -l <<< "scale=2;(($a+$b)*100)/($c + $d)"`
ri=`bc -l <<< "scale=2;($a*100)/$d"`
rp=`bc -l <<< "scale=2;($b*100)/$c"`
echo "Conversations:" $nspk
echo "No speaker change (0):" $c
echo "Speaker change (1):" $d
echo "1 -> 0: $a ($ri %)"
echo "0 -> 1: $b ($rp %)"
echo "Error rate: $r %"
