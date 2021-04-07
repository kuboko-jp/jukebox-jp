#! /bin/bash
python jukebox/train.py \
    --hps=vqvae,prior_1b_lyrics,all_fp16,cpu_ema \
    --name=finetuned \
    --sample_length=786432 \
    --bs=1 \
    --aug_shift \
    --aug_blend \
    --audio_files_dir=../wav_dataset \
    --labels=True \
    --train \
    --test \
    --prior \
    --levels=3 \
    --level=2 \
    --weight_decay=0.01 \
    --save_iters=1000