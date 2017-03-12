#!/bin/bash

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

spk_id=`cat $1 | cut -d' ' -f 1 | sort -u | shuf --random-source=<(get_seeded_random 42) | head -40`
echo "Selected test speakers:" $spk_id
test_id=`echo $spk_id | sed 's/ /\\\|/g'` 
grep    $test_id $1 > test.ctm
grep -v $test_id $1 > train.ctm
cp test.ctm valid.ctm

