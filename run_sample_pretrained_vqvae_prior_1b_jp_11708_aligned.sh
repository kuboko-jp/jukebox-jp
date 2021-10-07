#!/bin/bash
mpiexec -n 1 python jukebox/sample.py \
--model=1b_lyrics_jp_11708 \
--name=pretrained_vqvae_prior_1b_jp_11708_alignedLyrics_vocab145_epoch23 \
--levels=3 \
--sample_length_in_seconds=60 \
--total_sample_length_in_seconds=240 \
--sr=44100 \
--n_samples=16 \
--hop_fraction=0.5,0.5,0.125 \
--input_new_lyric \
--base_dir='/workspace/dataset/wav_dataset_006/' \
--jp=True

