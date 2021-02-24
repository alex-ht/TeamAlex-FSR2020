#!/bin/bash
set -euxo pipefail

stage=0
nj=24

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

if [ $stage -le 0 ]; then
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 3 data/pts data/pts_tr data/pts_cv
  local/train_lm.sh --order 2 --name bg
  local/train_lm.sh --order 3 --name tg
  local/prepare_dict.sh --source local/dict/moe.merge.lex --dict-dir data/local/dict
  utils/prepare_lang.sh --position-dependent-phones false data/local/dict "[SPN]" data/local/lang data/lang
  utils/format_lm_sri.sh --srilm-opts "-order 2 -subset -unk -map-unk [SPN]" \
    data/lang data/local/lm/merge.bg.gz data/local/dict/lexicon.txt data/lang_test/merge_bg
  utils/format_lm_sri.sh --srilm-opts "-subset -unk -map-unk [SPN]" \
    data/lang data/local/lm/merge.tg.gz data/local/dict/lexicon.txt data/lang_test/merge_tg
fi

if [ $stage -le 1 ]; then
  for x in aishell  ner  pilot_test  pts_cv  pts_tr  tat1; do
    steps/make_mfcc_pitch.sh --nj $nj --cmd "$train_cmd" data/${x}
    steps/compute_cmvn_stats.sh data/${x}
    utils/fix_data_dir.sh --cmd "$train_cmd" data/${x}
  done
  utils/data/combine_data.sh data/gmm_train data/{aishell,ner,pts_tr,tat1}
fi

train_set=gmm_train
lang=data/lang
lang_test=data/lang_test
if [ $stage -le 2 ]; then
  utils/subset_data_dir.sh data/gmm_train 10000 data/gmm_train_mono
  steps/train_mono.sh --nj $nj --cmd "$train_cmd" data/gmm_train_mono $lang exp/mono
  steps/align_si.sh --nj $nj --cmd "$train_cmd" data/${train_set} $lang exp/mono exp/mono_ali
fi

if [ $stage -le 3 ]; then
  steps/train_deltas.sh --cmd "$train_cmd" \
    2500 30000 data/${train_set} $lang exp/mono_ali exp/tri1
fi

if [ $stage -le 4 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/${train_set} $lang exp/tri1 exp/tri1_ali

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    4000 50000 data/${train_set} $lang exp/tri1_ali exp/tri2
fi

if [ $stage -le 5 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/${train_set} $lang exp/tri2 exp/tri2_ali

  steps/train_sat.sh --cmd "$train_cmd" \
    5000 100000 data/${train_set} $lang exp/tri2_ali exp/tri3
fi

if [ $stage -le 6 ]; then
  $mkgraph_cmd exp/tri3/graph_merge/log/mkgraph.log \
    utils/mkgraph.sh $lang_test/merge_bg exp/tri3 exp/tri3/graph_merge_bgpr
  $mkgraph_cmd exp/tri3/graph_minnan/log/mkgraph.log \
    utils/mkgraph.sh $lang_test/merge_tg exp/tri3 exp/tri3/graph_merge_tgpr

  steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" \
    exp/tri3/graph_merge_bgpr data/pilot_test exp/tri3/decode_pilot_test_merge_bgpr
  steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" \
    exp/tri3/graph_merge_tgpr data/pilot_test exp/tri3/decode_pilot_test_merge_tgpr
  steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" \
    exp/tri3/graph_merge_bgpr data/pts_cv exp/tri3/decode_pts_cv_merge_bgpr
  steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" \
    exp/tri3/graph_merge_tgpr data/pts_cv exp/tri3/decode_pts_cv_merge_tgpr
fi

if [ $stage -le 7 ]; then
  steps/align_fmllr.sh --nj 3000 --cmd "$train_cmd" \
    data/${train_set} $lang exp/tri3 exp/tri3_${train_set}_ali
  steps/cleanup/clean_and_segment_data.sh --nj 3000 --cmd "$mkgraph_cmd" \
    data/${train_set} data/lang exp/tri3_${train_set}_ali exp/tri3_cleanup data/${train_set}_cleaned
fi

train_set=${train_set}_cleaned
if [ $stage -le 8 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/${train_set} $lang exp/tri3 exp/tri3_${train_set}_ali

  steps/train_sat_basis.sh --cmd "$train_cmd" \
    6000 240000 data/${train_set} $lang exp/tri3_${train_set}_ali exp/tri4
fi

if [ $stage -le 9 ]; then
  $mkgraph_cmd exp/tri4/graph_merge/log/mkgraph.log \
    utils/mkgraph.sh $lang_test/merge_bg exp/tri4 exp/tri4/graph_merge_bgpr
  $mkgraph_cmd exp/tri4/graph_minnan/log/mkgraph.log \
    utils/mkgraph.sh $lang_test/merge_tg exp/tri4 exp/tri4/graph_merge_tgpr

  steps/decode_basis_fmllr.sh --nj $nj --cmd "$decode_cmd" --scoring-opts "--max-lmwt 22 --min-lmwt 12" \
    exp/tri4/graph_merge_bgpr data/pilot_test exp/tri4/decode_pilot_test_merge_bgpr
  steps/decode_basis_fmllr.sh --nj $nj --cmd "$decode_cmd" --scoring-opts "--max-lmwt 22 --min-lmwt 12" \
    exp/tri4/graph_merge_tgpr data/pilot_test exp/tri4/decode_pilot_test_merge_tgpr
  steps/decode_basis_fmllr.sh --nj $nj --cmd "$decode_cmd" --scoring-opts "--max-lmwt 22 --min-lmwt 12" \
    exp/tri4/graph_merge_bgpr data/pts_cv exp/tri4/decode_pts_cv_merge_bgpr
  steps/decode_basis_fmllr.sh --nj $nj --cmd "$decode_cmd" --scoring-opts "--max-lmwt 22 --min-lmwt 12" \
    exp/tri4/graph_merge_tgpr data/pts_cv exp/tri4/decode_pts_cv_merge_tgpr
fi

if [ $stage -le 10 ]; then
  steps/get_prons.sh --cmd "$train_cmd" \
    data/${train_set} data/lang exp/tri4
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict/ \
    exp/tri4/pron_counts.txt \
    data/local/dict/reestimated
  utils/prepare_lang.sh --position-dependent-phones false \
    data/local/dict/reestimated "[SPN]" data/local/dict/reestimated data/lang_reestimated
  utils/format_lm_sri.sh --srilm-opts "-order 2 -subset -unk -map-unk [SPN]" \
    data/lang_reestimated data/local/lm/merge.bg.gz data/local/dict/lexicon.txt data/lang_reestimated_test/merge_bg
  utils/format_lm_sri.sh --srilm-opts "-subset -unk -map-unk [SPN]" \
    data/lang_reestimated data/local/lm/merge.tg.gz data/local/dict/lexicon.txt data/lang_reestimated_test/merge_tg
fi
lang=data/lang_reestimated
lang_test=data/lang_reestimated_test
if [ $stage -le 11 ]; then
  $mkgraph_cmd exp/tri4/graph_merge/log/mkgraph.log \
    utils/mkgraph.sh $lang_test/merge_bg exp/tri4 exp/tri4/graph_merge_bgpr_re
  $mkgraph_cmd exp/tri4/graph_minnan/log/mkgraph.log \
    utils/mkgraph.sh $lang_test/merge_tg exp/tri4 exp/tri4/graph_merge_tgpr_re

  steps/decode_basis_fmllr.sh --nj $nj --cmd "$decode_cmd" --scoring-opts "--max-lmwt 22 --min-lmwt 12" \
    exp/tri4/graph_merge_bgpr_re data/pilot_test exp/tri4/decode_pilot_test_merge_bgpr_re
  steps/decode_basis_fmllr.sh --nj $nj --cmd "$decode_cmd" --scoring-opts "--max-lmwt 22 --min-lmwt 12" \
    exp/tri4/graph_merge_tgpr_re data/pilot_test exp/tri4/decode_pilot_test_merge_tgpr_re
  steps/decode_basis_fmllr.sh --nj $nj --cmd "$decode_cmd" --scoring-opts "--max-lmwt 22 --min-lmwt 12" \
    exp/tri4/graph_merge_bgpr_re data/pts_cv exp/tri4/decode_pts_cv_merge_bgpr_re
  steps/decode_basis_fmllr.sh --nj $nj --cmd "$decode_cmd" --scoring-opts "--max-lmwt 22 --min-lmwt 12" \
    exp/tri4/graph_merge_tgpr_re data/pts_cv exp/tri4/decode_pts_cv_merge_tgpr_re
fi

if [ $stage -le 12 ]; then
  steps/align_basis_fmllr_lats.sh --cmd "$train_cmd" --nj $nj \
    --generate-ali-from-lats true \
    data/$train_set $lang exp/tri4 exp/tri4_${train_set}_ali
  rm exp/tri4_${train_set}_ali/fsts.*.gz # save space

  # Reverb
  utils/data/get_reco2dur.sh data/gmm_train_cleaned
  steps/data/augment_data_dir.py \
    --random-seed 555 \
    --modify-spk-id true \
    --utt-prefix REVERB_NOISE --fg-snrs 20:10:5:0 \
    --bg-snrs 20:15:10 --num-bg-noise 1:2:3 \
    --fg-interval 1 --fg-noise-dir data/musan_noise \
    --bg-noise-dir data/musan_music \
    data/$train_set data/${train_set}_noise

  utils/validate_data_dir.sh --no-feats --non-print data/${train_set}_noise || exit 1
  steps/data/augment_data_dir.py \
    --random-seed 666 \
    --modify-spk-id true \
    --utt-prefix REVERB_SPEECH  --fg-snrs 20:15:10 \
    --bg-snrs 20:15:10 --num-bg-noise 1:2:3 \
    --fg-interval 1 --fg-noise-dir data/musan_speech \
    --bg-noise-dir data/musan_music \
    data/$train_set data/${train_set}_speech

  utils/validate_data_dir.sh --no-feats --non-print data/${train_set}_speech || exit 1

  steps/copy_ali_dir.sh --prefixes REVERB_NOISE data/${train_set}_noise exp/tri4_${train_set}_ali exp/tri4_${train_set}_noise_ali/
  steps/copy_ali_dir.sh --prefixes REVERB_SPEECH data/${train_set}_speech exp/tri4_${train_set}_ali exp/tri4_${train_set}_speech_ali/
  steps/copy_lat_dir.sh --prefixes REVERB_NOISE data/${train_set}_noise exp/tri4_${train_set}_ali exp/tri4_${train_set}_noise_lats/
  steps/copy_lat_dir.sh --prefixes REVERB_SPEECH data/${train_set}_speech exp/tri4_${train_set}_ali exp/tri4_${train_set}_speech_lats/
fi

if [ $stage -le 13 ]; then
  for x in gmm_train_cleaned_speech gmm_train_cleaned_noise; do
    utils/copy_data_dir.sh data/$x data/${x}_hires
    steps/make_mfcc.sh --nj 24 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${x}_hires || exit 1;
    steps/compute_cmvn_stats.sh data/${x}_hires || exit 1;
    utils/fix_data_dir.sh data/${x}_hires || exit 1;
  done
fi
