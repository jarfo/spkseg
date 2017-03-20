#!/bin/bash
nspk=`LC_ALL=es_ES.iso cut -d' ' -f 1 $1 | sort -u | wc -l`
tp=`LC_ALL=es_ES.iso cut -d' ' -f 6,7 $1 | grep "1 1" | wc -l`
fp=`LC_ALL=es_ES.iso cut -d' ' -f 6,7 $1 | grep "0 1" | wc -l`
tn=`LC_ALL=es_ES.iso cut -d' ' -f 6,7 $1 | grep "0 0" | wc -l`
fn=`LC_ALL=es_ES.iso cut -d' ' -f 6,7 $1 | grep "1 0" | wc -l`
accuracy=`bc -l <<< "scale=2;(($tp+$tn)*100)/($tp + $fp + $tn + $fn)"`
fnr=`bc -l <<< "scale=2;($fn*100)/($fn + $tp)"`
fpr=`bc -l <<< "scale=2;($fp*100)/($fp + $tn)"`
ppv=`bc -l <<< "scale=2;($tp*100)/($tp + $fp)"`
tpr=`bc -l <<< "scale=2;($tp*100)/($tp + $fn)"`
echo "Conversations:" $nspk
echo "No speaker change (0):" $(($fp + $tn))
echo "Speaker change    (1):" $(($tp + $fn))
echo "Precision (ppv): $ppv %"
echo "Recall    (tpr): $tpr %"
echo "Miss rate (fnr): $fnr %"
echo "Fall-out  (fpr): $fpr %"
echo "Accuracy       : $accuracy %"
