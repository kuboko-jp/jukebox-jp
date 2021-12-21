#!/bin/bash
mpiexec -n 8 python jukebox/train.py \
--jp_lyrics \
--hps=vqvae,prior_1b_jp,all_fp16,cpu_ema \
--name=pretrained_vqvae_prior_1b_jp_11708_alignedLyrics_vocab145_longTokens_fineTune30epochAligned \
--sample_length=786432 \
--bs=1 \
--aug_shift \
--aug_blend \
--audio_files_dir=/workspace/dataset/wav_dataset_006/wav_data \
--labels=True \
--train \
--test \
--prior \
--levels=3 \
--level=2 \
--weight_decay=0.01 \
--save_iters=1000 \
--epochs=10000 \
--curr_epoch=31 \

