#!/bin/bash

source=local/dict/moe.merge.lex
dict_dir=data/local/dict
. ./utils/parse_options.sh

export LC_ALL="C"

[ -d $dict_dir ] && rm -rf $dict_dir
mkdir -p $dict_dir

cat << EOF > $dict_dir/lexicon.txt
[SIL] SIL
[SPN] SPN
… SP
。 SP
， SP
、 SP
； SP
： SP
？ SP
！ SP
「 SP
」 SP
『 SP
』 SP
─ SP
— SP
（ SP
） SP
〔 SP
〕 SP
《 SP
》 SP
〈 SP
〉 SP
． SP
EOF
cat $source | sort | uniq >> $dict_dir/lexicon.txt

cut -f 2- -d\  $source |\
   tr ' ' '\n' | sed 's:SP::g' | sort | uniq |\
   grep '[[:print:]]' > $dict_dir/nonsilence_phones.txt

cat << EOF > $dict_dir/silence_phones.txt
SIL
SP
SPN
EOF

echo SIL > $dict_dir/optional_silence.txt

( for phn in $(tr -d '12345678' < $dict_dir/nonsilence_phones.txt  | uniq); do
  grep "^$phn" $dict_dir/nonsilence_phones.txt | tr '\n' ' ';
  echo ;
done ) > $dict_dir/extra_questions.txt

exit 0;
