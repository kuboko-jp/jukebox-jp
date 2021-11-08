#!/bin/bash
mpiexec -n 1 python jukebox/sample.py \
--model=1b_lyrics_finetune_roma_11708 \
--name=1b_lyrics_finetune_roma_11708_epoch010_enjplyrics \
--levels=3 \
--sample_length_in_seconds=90 \
--total_sample_length_in_seconds=180 \
--sr=44100 \
--n_samples=16 \
--hop_fraction=0.5,0.5,0.125 \
--input_new_lyric \
--base_dir='/workspace/dataset/wav_dataset_006/' \