#!/bin/bash

. ./path.sh
order=3
name=tg
. ./utils/parse_options.sh
export LC_ALL="C.UTF-8"

#rm -rf data/local/lm
mkdir -p data/local/lm
cut -d " " -f 1 local/dict/moe.merge.lex | uniq > data/local/lm/wlist
for x in aishell ner pts_tr tat1; do
    cut -d\  -f 2- data/$x/text > data/local/lm/$x.txt
done
cat data/pts_cv/text | \
    cut -d\  -f 2- > data/local/lm/valid.txt

for x in aishell ner pts_tr tat1; do
    ngram-count \
        -text data/local/lm/$x.txt \
        -vocab data/local/lm/wlist \
        -map-unk "[SPN]" \
        -unk -kndiscount -order $order -lm data/local/lm/$x.$name.gz

    ngram -unk -map-unk "[SPN]" \
        -order $order \
        -debug 2 -lm data/local/lm/$x.$name.gz \
        -ppl data/local/lm/valid.txt > data/local/lm/$x.$name.ppl
done

msg=$(compute-best-mix data/local/lm/{aishell,ner,pts_tr,tat1}.$name.ppl 2>&1 | grep "best lambda" | sed 's:[()]: :g' | cut -d ' ' -f 6-)
echo "$msg"
IFS=' ' read -r -a array <<< "$msg"
ngram -vocab data/local/lm/wlist -order $order -unk -map-unk "[SPN]" \
    -lm data/local/lm/aishell.$name.gz \
    -lambda ${array[0]} \
    -mix-lm data/local/lm/ner.$name.gz \
    -mix-lm2 data/local/lm/pts_tr.$name.gz \
    -mix-lambda2 ${array[2]} \
    -mix-lm3 data/local/lm/tat1.$name.gz \
    -mix-lambda3 ${array[3]} \
    -renorm \
    -write-lm data/local/lm/merge.$name.gz

ngram -unk -map-unk "[SPN]" \
    -prune-lowprobs \
    -order $order \
    -lm data/local/lm/merge.$name.gz \
    -ppl data/local/lm/valid.txt
