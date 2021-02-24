#!/usr/bin/env bash

set -e

# configs for 'chain'
affix=1a
nnet3_affix=
stage=0
train_stage=-10
get_egs_stage=-10
dir=exp/chain/cnn  # Note: _sp will get added to this
decode_iter=

# training options
num_epochs=4
initial_effective_lrate=0.00025
final_effective_lrate=0.000025
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=2
num_jobs_final=6
minibatch_size=128,64
frames_per_eg=150,110,90
remove_egs=false
common_egs_dir=
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

dir=${dir}${affix:+_$affix}_sp
gmm=tri4
train_set=gmm_train_cleaned
ali_dir=exp/${gmm}_${train_set}_ali
treedir=exp/chain/${gmm}_tree
lang=data/lang_chain


# if we are using the speed-perturbed data we need to generate
# alignments for it.
local/nnet3/run_ivector_common.sh --stage $stage --train-set ${train_set} --gmm ${gmm} --nnet3-affix "$nnet3_affix"

if [ $stage -le 7 ]; then
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat $ali_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/${train_set}_sp \
    data/lang_reestimated exp/${gmm} exp/${gmm}_${train_set}_sp_lats
  rm exp/${gmm}_${train_set}_sp_lats/fsts.*.gz # save space
  utils/combine_data.sh data/${train_set}_sp_musan data/${train_set}_sp data/${train_set}_speech data/${train_set}_noise
  steps/combine_lat_dirs.sh --cmd "$train_cmd" --nj $nj --combine_ali false data/${train_set}_sp_musan exp/${gmm}_${train_set}_sp_musan_lats exp/${gmm}_${train_set}_sp_lats exp/${gmm}_${train_set}_noise_lats exp/${gmm}_${train_set}_speech_lats
  utils/data/combine_data.sh data/${train_set}_sp_musan_hires data/${train_set}_sp_hires data/${train_set}_noise_hires data/${train_set}_speech_hires
  ivectordir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_musan
  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  temp_data_root=${ivectordir}
  #utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
  #  data/${train_set}_sp_musan_hires ${temp_data_root}/${train_set}_sp_musan_hires_max2
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 24 \
    ${temp_data_root}/${train_set}_sp_musan_hires_max2 \
    exp/nnet3${nnet3_affix}/extractor $ivectordir
fi

if [ $stage -le 8 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang_reestimated $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 9 ]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$train_cmd" 5000 data/${train_set} $lang $ali_dir $treedir
fi

if [ $stage -le 10 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  cnn_opts="l2-regularize=0.03"
  ivector_layer_opts="l2-regularize=0.03"
  ivector_affine_opts="l2-regularize=0.03"
  tdnn_opts="l2-regularize=0.03 dropout-proportion=0.0 dropout-per-dim-continuous=true"
  tdnnf_first_opts="l2-regularize=0.03 dropout-proportion=0.0 bypass-scale=0.0"
  tdnnf_opts="l2-regularize=0.03 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.03 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.03"
  output_opts="l2-regularize=0.015"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=64 name=input

  # this takes the MFCCs and generates filterbank coefficients.  The MFCCs
  # are more compressible so we prefer to dump the MFCCs to disk rather
  # than filterbanks.
  idct-layer name=idct input=input dim=64 cepstral-lifter=22 affine-transform-file=$dir/configs/idct.mat
  linear-component name=ivector-linear $ivector_affine_opts dim=256 input=ReplaceIndex(ivector, t, 0)
  batchnorm-component name=ivector-batchnorm target-rms=0.025
  batchnorm-component name=idct-batchnorm input=idct
  spec-augment-layer name=idct-spec-augment freq-max-proportion=0.5 time-zeroed-proportion=0.2 time-mask-max-frames=20
  combine-feature-maps-layer name=combine_inputs input=Append(idct-spec-augment, ivector-batchnorm) num-filters1=1 num-filters2=4 height=64
  conv-relu-batchnorm-layer name=cnn1 $cnn_opts height-in=64 height-out=64 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64 learning-rate-factor=0.333 max-change=0.25
  conv-relu-batchnorm-layer name=cnn2 $cnn_opts height-in=64 height-out=64 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  conv-relu-batchnorm-layer name=cnn3 $cnn_opts height-in=64 height-out=32 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  conv-relu-batchnorm-layer name=cnn4 $cnn_opts height-in=32 height-out=32 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  conv-relu-batchnorm-layer name=cnn5 $cnn_opts height-in=32 height-out=16 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
  conv-relu-batchnorm-layer name=cnn6 $cnn_opts height-in=16 height-out=8 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
  # the first TDNN-F layer has no bypass (since dims don't match), and a larger bottleneck so the
  # information bottleneck doesn't become a problem.  (we use time-stride=0 so no splicing, to
  # limit the num-parameters).
  tdnnf-layer name=tdnnf7 $tdnnf_first_opts dim=1536 bottleneck-dim=192 time-stride=0
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  linear-component name=prefinal-l dim=256 $linear_opts
  ## adding the layers for chain branch
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts small-dim=256 big-dim=1536
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts
  # adding the layers for xent branch
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts small-dim=256 big-dim=1536
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi
train_set=${train_set}_sp_musan
if [ $stage -le 11 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir exp/nnet3$nnet3_affix/ivectors_${train_set} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false" \
    --egs.chunk-width $frames_per_eg \
    --trainer.optimization.num-jobs-step 2 \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 3000000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set}_hires \
    --tree-dir $treedir \
    --lat-dir exp/${gmm}_${train_set}_lats \
    --use-gpu wait \
    --dir $dir  || exit 1;
fi

if [ $stage -le 12 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_reestimated_test/merge_bg $dir $dir/graph_bgpr
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_reestimated_test/merge_tg $dir $dir/graph_tgpr
fi

graph_dir=$dir/graph_bgpr
if [ $stage -le 13 ]; then
  for test_set in pts_cv pilot_test; do
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj 4 --cmd "$decode_cmd" \
      --online-ivector-dir exp/nnet3${nnet3_affix:+$nnet3_affix}/ivectors_$test_set \
      $graph_dir data/${test_set}_hires $dir/decode_bgpr_${test_set} || exit 1;
  done
  wait;
fi

graph_dir=$dir/graph_tgpr
if [ $stage -le 14 ]; then
  for test_set in pts_cv pilot_test; do
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj 4 --cmd "$decode_cmd" \
      --online-ivector-dir exp/nnet3${nnet3_affix:+$nnet3_affix}/ivectors_$test_set \
      $graph_dir data/${test_set}_hires $dir/decode_tgpr_${test_set} || exit 1;
  done
  wait;
fi

dir=${dir}_ftv
mkdir -p $dir
train_set=${train_set}_ftv
if [ $stage -le 15 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir exp/nnet3$nnet3_affix/ivectors_${train_set} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false" \
    --egs.chunk-width $frames_per_eg \
    --trainer.input-model exp/chain/cnn_1a_sp/final.mdl \
    --trainer.optimization.num-jobs-step 2 \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 3000000 \
    --trainer.num-epochs 1 \
    --trainer.optimization.num-jobs-initial $num_jobs_final \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $final_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set}_hires \
    --tree-dir $treedir \
    --lat-dir exp/${gmm}_${train_set}_lats \
    --use-gpu wait \
    --dir $dir  || exit 1;
fi
exit 0;
