**Status:** Archive (code is provided as-is, no updates expected)<br>
**ステータス** アーカイブ (コードはそのまま提供されています。更新は期待されていません)


# Jukebox
Code for "Jukebox: A Generative Model for Music"

[Paper](https://arxiv.org/abs/2005.00341) 
[Blog](https://openai.com/blog/jukebox) 
[Explorer](http://jukebox.openai.com/) 
[Colab](https://colab.research.google.com/github/openai/jukebox/blob/master/jukebox/Interacting_with_Jukebox.ipynb) 

# Install
Install the conda package manager from https://docs.conda.io/en/latest/miniconda.html <br> 
conda パッケージマネージャを https://docs.conda.io/en/latest/miniconda.html からインストールします。   

    
``` 
# Required: Sampling
conda create --name jukebox python=3.7.5
conda activate jukebox
conda install mpi4py=3.0.3 # if this fails, try: pip install mpi4py==3.0.3
conda install pytorch=1.4 torchvision=0.5 cudatoolkit=10.0 -c pytorch
git clone https://github.com/openai/jukebox.git
cd jukebox
pip install -r requirements.txt
pip install -e .

# Required: Training
conda install av=7.0.01 -c conda-forge 
pip install ./tensorboardX
 
# Optional: Apex for faster training with fused_adam
conda install pytorch=1.1 torchvision=0.3 cudatoolkit=10.0 -c pytorch
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
```

# Sampling
## Sampling from scratch
To sample normally, run the following command. Model can be `5b`, `5b_lyrics`, `1b_lyrics`<br>
通常のサンプルを作成するには、以下のコマンドを実行します。モデルは `5b`, `5b_lyrics`, `1b_lyrics` です。
``` 
python jukebox/sample.py --model=5b_lyrics --name=sample_5b --levels=3 --sample_length_in_seconds=20 \
--total_sample_length_in_seconds=180 --sr=44100 --n_samples=6 --hop_fraction=0.5,0.5,0.125
```
``` 
python jukebox/sample.py --model=1b_lyrics --name=sample_1b --levels=3 --sample_length_in_seconds=20 \
--total_sample_length_in_seconds=180 --sr=44100 --n_samples=16 --hop_fraction=0.5,0.5,0.125
```
The above generates the first `sample_length_in_seconds` seconds of audio from a song of total length `total_sample_length_in_seconds`.
To use multiple GPU's, launch the above scripts as `mpiexec -n {ngpus} python jukebox/sample.py ...` so they use `{ngpus}`<br>
上記は、全長 `total_sample_length_in_seconds` の曲から最初の `sample_length_in_seconds` 秒の音声を生成します。
複数のGPUを利用するには、`mpiexec -n {ngpus} python jukebox/sample.py ...`として上記のスクリプトを起動すると、`{ngpus}`を利用するようになります。

The samples decoded from each level are stored in `{name}/level_{level}`. 
You can also view the samples as an html with the aligned lyrics under `{name}/level_{level}/index.html`. 
Run `python -m http.server` and open the html through the server to see the lyrics animate as the song plays.  
A summary of all sampling data including zs, x, labels and sampling_kwargs is stored in `{name}/level_{level}/data.pth.tar`.<br>
各レベルからデコードされたサンプルは `{name}/level_{level}` に格納されます。
また、`{name}/level_{level}/index.html`の下に歌詞を並べてhtmlとして表示することもできます。
python -m http.server`を実行して、サーバからhtmlを開くと、曲の再生に合わせて歌詞がアニメーションで表示されます。 
zs, x, labels, sampling_kwargsを含むすべてのサンプリングデータの要約は、`{name}/level_{level}/data.pth.tar`に格納されています。

The hps are for a V100 GPU with 16 GB GPU memory. The `1b_lyrics`, `5b`, and `5b_lyrics` top-level priors take up 
3.8 GB, 10.3 GB, and 11.5 GB, respectively. The peak memory usage to store transformer key, value cache is about 400 MB 
for `1b_lyrics` and 1 GB for `5b_lyrics` per sample. If you are having trouble with CUDA OOM issues, try `1b_lyrics` or 
decrease `max_batch_size` in sample.py, and `--n_samples` in the script call.<br>
hpsは、16GBのGPUメモリを搭載したV100 GPUの場合。`1b_lyrics`、`5b`、`5b_lyrics`のトップレベルプリオはそれぞれ3.8GB、10.3GB、11.5GBを使用しています。トランスフォーマーのキー、値キャッシュを格納するためのピークメモリ使用量は、1サンプルあたり`1b_lyrics`で約400MB、`5b_lyrics`で約1GBです。CUDA OOMの問題で悩んでいる場合は、`1b_lyrics`を試すか、sample.pyで`max_batch_size`を減らし、スクリプトコールで`--n_samples`を減らしてみてください。

On a V100, it takes about 3 hrs to fully sample 20 seconds of music. Since this is a long time, it is recommended to use `n_samples > 1` so you can generate as many samples as possible in parallel. The 1B lyrics and upsamplers can process 16 samples at a time, while 5B can fit only up to 3. Since the vast majority of time is spent on upsampling, we recommend using a multiple of 3 less than 16 like `--n_samples 15` for `5b_lyrics`. This will make the top-level generate samples in groups of three while upsampling is done in one pass.<br>
V100では、20秒の音楽を完全にサンプリングするのに約3時間かかります。これは長いので、できるだけ多くのサンプルを並行して生成できるように、`n_samples > 1`を使うことをお勧めします。1Bの歌詞とアップサンプラーは一度に16個のサンプルを処理できますが、5Bでは3個までしか処理できません。 大部分の時間がアップサンプリングに費やされるので、`5b_lyrics`には`--n_samples 15`のように16よりも3の倍数を使うことをお勧めします。これにより、トップレベルは3つのグループに分けてサンプルを生成し、アップサンプリングは1回のパスで行われます。

To continue sampling from already generated codes for a longer duration, you can run<br>
すでに生成されたコードからより長い期間サンプリングを続けるには、次のように実行します。
```
python jukebox/sample.py --model=5b_lyrics --name=sample_5b --levels=3 --mode=continue \
--codes_file=sample_5b/level_0/data.pth.tar --sample_length_in_seconds=40 --total_sample_length_in_seconds=180 \
--sr=44100 --n_samples=6 --hop_fraction=0.5,0.5,0.125
```
Here, we take the 20 seconds samples saved from the first sampling run at `sample_5b/level_0/data.pth.tar` and continue by adding 20 more seconds.<br>
ここでは、`sample_5b/level_0/data.pth.tar`にある最初のサンプリング実行から保存された20秒のサンプルを取り、さらに20秒のサンプルを追加して続けます。

You could also continue directly from the level 2 saved outputs, just pass `--codes_file=sample_5b/level_2/data.pth.tar`.
 Note this will upsample the full 40 seconds song at the end.<br>
 レベル2の保存された出力から直接続けることもできます。
 これにより、最後に40秒の曲がアップサンプルされることに注意してください。

If you stopped sampling at only the first level and want to upsample the saved codes, you can run<br>
最初のレベルだけでサンプリングを停止し、保存されたコードをアップサンプリングしたい場合は、次のように実行します。
```
python jukebox/sample.py --model=5b_lyrics --name=sample_5b --levels=3 --mode=upsample \
--codes_file=sample_5b/level_2/data.pth.tar --sample_length_in_seconds=20 --total_sample_length_in_seconds=180 \
--sr=44100 --n_samples=6 --hop_fraction=0.5,0.5,0.125
```
Here, we take the 20 seconds samples saved from the first sampling run at `sample_5b/level_2/data.pth.tar` and upsample the lower two levels.<br>
ここでは、`sample_5b/level_2/data.pth.tar`の最初のサンプリング実行から保存された20秒のサンプルを取り、下位の2つのレベルをアップサンプリングします。

## Prompt with your own music
If you want to prompt the model with your own creative piece or any other music, first save them as wave files and run<br>
あなた自身の創造的な作品やその他の音楽でモデルを表示したい場合は、まずそれらをwaveファイルとして保存してから実行する

```
python jukebox/sample.py --model=5b_lyrics --name=sample_5b_prompted --levels=3 --mode=primed \
--audio_file=path/to/recording.wav,awesome-mix.wav,fav-song.wav,etc.wav --prompt_length_in_seconds=12 \
--sample_length_in_seconds=20 --total_sample_length_in_seconds=180 --sr=44100 --n_samples=6 --hop_fraction=0.5,0.5,0.125
```
This will load the four files, tile them to fill up to `n_samples` batch size, and prime the model with the first `prompt_length_in_seconds` seconds.<br>
これにより、4つのファイルがロードされ、`n_samples` のバッチサイズを満たすようにタイル化され、最初の `prompt_length_in_seconds` 秒でモデルがプライムされます。

# Training
## VQVAE
To train a small vqvae, run
```
mpiexec -n {ngpus} python jukebox/train.py --hps=small_vqvae --name=small_vqvae --sample_length=262144 --bs=4 \
--audio_files_dir={audio_files_dir} --labels=False --train --aug_shift --aug_blend
```
Here, `{audio_files_dir}` is the directory in which you can put the audio files for your dataset, and `{ngpus}` is number of GPU's you want to use to train. 
The above trains a two-level VQ-VAE with `downs_t = (5,3)`, and `strides_t = (2, 2)` meaning we downsample the audio by `2**5 = 32` to get the first level of codes, and `2**8 = 256` to get the second level codes.  
Checkpoints are stored in the `logs` folder. You can monitor the training by running Tensorboard<br>
ここで，`{audio_files_dir}` はデータセットの音声ファイルを格納するディレクトリ，`{ngpus}` は学習に使用するGPUの数である．
上の例では，`downs_t = (5,3)`，`strides_t = (2, 2)` の2レベルのVQ-VAEを学習しているが，これは，第1レベルのコードを得るために`2**5 = 32`，第2レベルのコードを得るために`2**8 = 256`でオーディオをダウンサンプルすることを意味している． 
チェックポイントは `logs` フォルダに保存されます。
Tensorboard によって訓練の様子をモニター出来ます。
```
tensorboard --logdir logs
```
    
## Prior
### Train prior or upsamplers
Once the VQ-VAE is trained, we can restore it from its saved checkpoint and train priors on the learnt codes. 
To train the top-level prior, we can run<br>
VQ-VAEが訓練されると、保存されたチェックポイントから復元し、学習されたコードの優先順位を訓練することができます。
'the top-level prior'を学習するには、以下のように実行します。

```
mpiexec -n {ngpus} python jukebox/train.py --hps=small_vqvae,small_prior,all_fp16,cpu_ema --name=small_prior \
--sample_length=2097152 --bs=4 --audio_files_dir={audio_files_dir} --labels=False --train --test --aug_shift --aug_blend \
--restore_vqvae=logs/small_vqvae/checkpoint_latest.pth.tar --prior --levels=2 --level=1 --weight_decay=0.01 --save_iters=1000
```

To train the upsampler, we can run<br>
アップサンプラを訓練するには、次のようにします。
```
mpiexec -n {ngpus} python jukebox/train.py --hps=small_vqvae,small_upsampler,all_fp16,cpu_ema --name=small_upsampler \
--sample_length=262144 --bs=4 --audio_files_dir={audio_files_dir} --labels=False --train --test --aug_shift --aug_blend \
--restore_vqvae=logs/small_vqvae/checkpoint_latest.pth.tar --prior --levels=2 --level=0 --weight_decay=0.01 --save_iters=1000
```
We pass `sample_length = n_ctx * downsample_of_level` so that after downsampling the tokens match the n_ctx of the prior hps. 
Here, `n_ctx = 8192` and `downsamples = (32, 256)`, giving `sample_lengths = (8192 * 32, 8192 * 256) = (65536, 2097152)` respectively for the bottom and top level.<br>
ここで，`sample_length = n_ctx * downsample_of_level` を渡すことで，ダウンサンプリング後のトークンが前の hps の n_ctx と一致するようにします．
ここで、`n_ctx = 8192`, `downsamples = (32, 256)` とすると、ボトムレベルとトップレベルについて、それぞれ `sample_lengths = (8192 * 32, 8192 * 256) = (65536, 2097152)` となります。

### Learning rate annealing
To get the best sample quality anneal the learning rate to 0 near the end of training. To do so, continue training from the latest 
checkpoint and run with<br>
最高のサンプル品質を得るためには、トレーニングの終了近くで学習率を0にアニーリングします。そのためには、最新の チェックポイントから実行を続ける。
```
--restore_prior="path/to/checkpoint" --lr_use_linear_decay --lr_start_linear_decay={already_trained_steps} --lr_decay={decay_steps_as_needed}
```

### Reuse pre-trained VQ-VAE and train top-level prior on new dataset from scratch.(事前に学習したVQ-VAEを再利用し、新しいデータセットでトップレベルの優先順位をゼロから学習します。)
#### Train without labels
Our pre-trained VQ-VAE can produce compressed codes for a wide variety of genres of music, and the pre-trained upsamplers 
can upsample them back to audio that sound very similar to the original audio.
To re-use these for a new dataset of your choice, you can retrain just the top-level<br>
事前に訓練されたVQ-VAEは、様々なジャンルの音楽に対応した圧縮コードを作成することができ、訓練済みのアップサンプラーは、元のオーディオに非常に近いサウンドのオーディオにアップサンプリングして戻すことができます。
これらを任意の新しいデータセットに再利用するために、トップレベルだけを再学習することができます。

To train top-level on a new dataset, run
```
mpiexec -n {ngpus} python jukebox/train.py --hps=vqvae,small_prior,all_fp16,cpu_ema --name=pretrained_vqvae_small_prior \
--sample_length=1048576 --bs=4 --aug_shift --aug_blend --audio_files_dir={audio_files_dir} \
--labels=False --train --test --prior --levels=3 --level=2 --weight_decay=0.01 --save_iters=1000
```
Training the `small_prior` with a batch size of 2, 4, and 8 requires 6.7 GB, 9.3 GB, and 15.8 GB of GPU memory, respectively. A few days to a week of training typically yields reasonable samples when the dataset is homogeneous (e.g. all piano pieces, songs of the same style, etc).<br>
バッチサイズが2、4、8の場合、`small_prior`の学習にはそれぞれ6.7GB、9.3GB、15.8GBのGPUメモリが必要です。データセットが均質な場合（例えば、すべてのピアノ曲、同じスタイルの曲など）、数日から1週間のトレーニングで、通常は妥当なサンプルを得ることができます。

Near the end of training, follow [this](#learning-rate-annealing) to anneal the learning rate to 0<br>
トレーニングの終わり近くでは、[これ](#learning-rate-annealing)に従って学習率を0にするためのアニーリングを行う


#### Sample from new model
You can then run sample.py with the top-level of our models replaced by your new model. To do so,
- Add an entry `my_model=("vqvae", "upsampler_level_0", "upsampler_level_1", "small_prior")` in `MODELS` in `make_models.py`. 
- Update the `small_prior` dictionary in `hparams.py` to include `restore_prior='path/to/checkpoint'`. If you
you changed any hps directly in the command line script (eg:`heads`), make sure to update them in the dictionary too so 
that `make_models` restores our checkpoint correctly.
- Run sample.py as outlined in the sampling section, but now with `--model=my_model` <br>

そして、モデルのトップレベルを新しいモデルに置き換えてsample.pyを実行することができます。
これを行うには、以下のようにします。
- my_model=("vqvae", "upsampler_level_0", "upsampler_level_1", "small_prior")`を `make_models.py` の `MODELS` に追加する。
- hparams.py` の `small_prior` 辞書を `restore_prior='path/to/checkpoint'` を含むように更新する。
コマンドラインスクリプトで直接 hps を変更した場合 (例: `heads`) は、以下のように辞書を更新してください。で `make_models` がチェックポイントを正しく復元していることを確認してください。
- サンプリングセクションで説明したようにsample.pyを実行しますが、`--model=my_model`を指定します。

For example, let's say we trained `small_vqvae`, `small_prior`, and `small_upsampler` under `/path/to/jukebox/logs`. In `make_models.py`, we are going to declare a tuple of the new models as `my_model`.<br>
例えば、`/path/to/jukebox/logs` の下で `small_vqvae`, `small_prior`, `small_upsampler` を学習させたとします。make_models.pyでは、新しいモデルのタプルを `my_model` として宣言します。
```
MODELS = {
    '5b': ("vqvae", "upsampler_level_0", "upsampler_level_1", "prior_5b"),
    '5b_lyrics': ("vqvae", "upsampler_level_0", "upsampler_level_1", "prior_5b_lyrics"),
    '1b_lyrics': ("vqvae", "upsampler_level_0", "upsampler_level_1", "prior_1b_lyrics"),
    'my_model': ("my_small_vqvae", "my_small_upsampler", "my_small_prior"),
}
```

Next, in `hparams.py`, we add them to the registry with the corresponding `restore_`paths and any other command line options used during training. Another important note is that for top-level priors with lyric conditioning, we have to locate a self-attention layer that shows alignment between the lyric and music tokens. Look for layers where `prior.prior.transformer._attn_mods[layer].attn_func` is either 6 or 7. If your model is starting to sing along lyrics, it means some layer, head pair has learned alignment. Congrats!<br>
次に、`hparams.py`の中で、`restore_`paths'やトレーニング中に使用されたその他のコマンドラインオプションに対応する`hparams.py`をレジストリに追加します。
もう一つの重要な注意点は、歌詞条件付けを伴うトップレベルのプリオールについては、歌詞とミュージックトークンの間のアライメントを示す自己注意層を見つけなければならないということです。
`prior.previous.transformer._attn_mods[layer].attn_func`が6か7のレイヤを探してください。あなたのモデルが歌詞に沿って歌い始めたら、それはあるレイヤー、ヘッドペアがアライメントを学習したことを意味します。
おめでとうございます。
```
my_small_vqvae = Hyperparams(
    restore_vqvae='/path/to/jukebox/logs/small_vqvae/checkpoint_some_step.pth.tar',
)
my_small_vqvae.update(small_vqvae)
HPARAMS_REGISTRY["my_small_vqvae"] = my_small_vqvae

my_small_prior = Hyperparams(
    restore_prior='/path/to/jukebox/logs/small_prior/checkpoint_latest.pth.tar',
    level=1,
    labels=False,
    # TODO For the two lines below, if `--labels` was used and the model is
    # trained with lyrics, find and enter the layer, head pair that has learned
    # alignment.
    alignment_layer=47,
    alignment_head=0,
)
my_small_prior.update(small_prior)
HPARAMS_REGISTRY["my_small_prior"] = my_small_prior

my_small_upsampler = Hyperparams(
    restore_prior='/path/to/jukebox/logs/small_upsampler/checkpoint_latest.pth.tar',
    level=0,
    labels=False,
)
my_small_upsampler.update(small_upsampler)
HPARAMS_REGISTRY["my_small_upsampler"] = my_small_upsampler
```

#### Train with labels 
To train with you own metadata for your audio files, implement `get_metadata` in `data/files_dataset.py` to return the 
`artist`, `genre` and `lyrics` for a given audio file. For now, you can pass `''` for lyrics to not use any lyrics.<br>
オーディオファイルのメタデータを使って学習するには、`data/files_dataset.py` で `get_metadata` を実装して 
与えられたオーディオファイルに対して `artist`, `genre`, `lyrics` を指定します。今のところ、歌詞に `''` を渡すことで歌詞を使わないようにすることができます。

For training with labels, we'll use `small_labelled_prior` in `hparams.py`, and we set `labels=True,labels_v3=True`. <br>
ラベルを用いた学習では、`hparams.py` で `small_labelled_prior` を用い、`labels=True,labels_v3=True` を設定します。<br>
We use 2 kinds of labels information:
- Artist/Genre: 
  - For each file, we return an artist_id and a list of genre_ids. The reason we have a list and not a single genre_id 
  is that in v2, we split genres like `blues_rock` into a bag of words `[blues, rock]`, and we pass atmost 
  `max_bow_genre_size` of those, in `v3` we consider it as a single word and just set `max_bow_genre_size=1`.
  - 各ファイルについて、私たちは artist_id と genre_id のリストを返します。単一のジャンルIDではなくリストを返すのは、v2では `blues_rock` のようなジャンルを `[blues, rock]` のような単語の袋に分割していたためです。 これらのうち `max_bow_genre_size` は、`v3` では一つの単語とみなして `max_bow_genre_size=1` とします。
  - Update the `v3_artist_ids` and `v3_genre_ids` to use ids from your new dataset. 
  - v3_artist_ids` と `v3_genre_ids` を更新して、新しいデータセットの ID を使うようにします。
  - In `small_labelled_prior`, set the hps `y_bins = (number_of_genres, number_of_artists)` and `max_bow_genre_size=1`. 
  - small_labelled_prior` で、`y_bins = (number_of_genres, number_of_artists)` と `max_bow_genre_size=1` を設定します。
- Timing: 
  - For each chunk of audio, we return the `total_length` of the song, the `offset` the current audio chunk is at and 
  the `sample_length` of the audio chunk. We have three timing embeddings: total_length, our current position, and our 
  current position as a fraction of the total length, and we divide the range of these values into `t_bins` discrete bins. 
  - 各オーディオチャンクに対して、曲の `total_length`、現在のオーディオチャンクの `offset`、および はオーディオチャンクの `sample_length` です。3つのタイミングエンベッディングがあります: total_length、現在の位置、そして これらの値の範囲を `t_bins` の離散的なビンに分割します。
  - In `small_labelled_prior`, set the hps `min_duration` and `max_duration` to be the shortest/longest duration of audio 
  files you want for your dataset, and `t_bins` for how many bins you want to discretize timing information into. Note 
  `min_duration * sr` needs to be at least `sample_length` to have an audio chunk in it.
  - `small_labelled_prior`では、`min_duration` と `max_duration` にオーディオの最短/最長の長さを設定します。ファイル、タイミング情報を離散化したいビン数を `t_bins` で指定します。
注意事項 `min_duration * sr` は少なくとも `sample_length` でなければなりません。

After these modifications, to train a top-level with labels, run<br>
これらの修正の後、ラベルを使ってトップレベルを学習するには、次のように実行します。
```
mpiexec -n {ngpus} python jukebox/train.py --hps=vqvae,small_labelled_prior,all_fp16,cpu_ema --name=pretrained_vqvae_small_prior_labels \
--sample_length=1048576 --bs=4 --aug_shift --aug_blend --audio_files_dir={audio_files_dir} \
--labels=True --train --test --prior --levels=3 --level=2 --weight_decay=0.01 --save_iters=1000
```

For sampling, follow same instructions as [above](#sample-from-new-model) but use `small_labelled_prior` instead of `small_prior`.<br>
サンプリングについては、[上記](#sample-from-new-model)と同じ指示に従いますが、`small_prior` の代わりに `small_labelled_prior` を使用します。 

#### Train with lyrics
To train in addition with lyrics, update `get_metadata` in `data/files_dataset.py` to return `lyrics` too.
For training with lyrics, we'll use `small_single_enc_dec_prior` in `hparams.py`. <br>
歌詞を加えて学習するには、`data/files_dataset.py` の `get_metadata` を更新して `lyrics` も返すようにします。
歌詞を使った学習には、`hparams.py` の `small_single_enc_dec_prior` を利用します。
- Lyrics: 
  - For each file, we linearly align the lyric characters to the audio, find the position in lyric that corresponds to 
  the midpoint of our audio chunk, and pass a window of `n_tokens` lyric characters centred around that. 
  - In `small_single_enc_dec_prior`, set the hps `use_tokens=True` and `n_tokens` to be the number of lyric characters 
  to use for an audio chunk. Set it according to the `sample_length` you're training on so that its large enough that 
  the lyrics for an audio chunk are almost always found inside a window of that size.
  - If you use a non-English vocabulary, update `text_processor.py` with your new vocab and set
  `n_vocab = number of characters in vocabulary` accordingly in `small_single_enc_dec_prior`. In v2, we had a `n_vocab=80` 
  and in v3 we missed `+` and so `n_vocab=79` of characters. 
- 歌詞 
  - 各ファイルについて、歌詞の文字を音声に合わせて直線的に配置し、歌詞の中で をオーディオチャンクの中点に設定し、そこを中心とした `n_tokens` のリリック文字のウィンドウを渡す。
  - `small_single_enc_dec_prior` で、`use_tokens=True` と `n_tokens` にオーディオチャンクに使用するリリック文字の数を設定する。これを `sample_length` に合わせて設定することで、音声チャンクの歌詞がほとんどの場合、そのサイズのウィンドウ内に収まるようになります。
  - 英語以外の語彙を使う場合は、`text_processor.py` を新しい語彙で更新し、`small_single_enc_dec_prior` で `n_vocab = 語彙の文字数` を適宜設定してください。v2では `n_vocab=80` としていました。v3では `+` を見落としていたため、`n_vocab=79` の文字が出てきてしまいました。

After these modifications, to train a top-level with labels and lyrics, run<br>
これらの修正の後、ラベルと歌詞でトップレベルをトレーニングするには、以下のように実行します。
```
mpiexec -n {ngpus} python jukebox/train.py --hps=vqvae,small_single_enc_dec_prior,all_fp16,cpu_ema --name=pretrained_vqvae_small_single_enc_dec_prior_labels \
--sample_length=786432 --bs=4 --aug_shift --aug_blend --audio_files_dir={audio_files_dir} \
--labels=True --train --test --prior --levels=3 --level=2 --weight_decay=0.01 --save_iters=1000
```
To simplify hps choices, here we used a `single_enc_dec` model like the `1b_lyrics` model that combines both encoder and 
decoder of the transformer into a single model. We do so by merging the lyric vocab and vq-vae vocab into a single 
larger vocab, and flattening the lyric tokens and the vq-vae codes into a single sequence of length `n_ctx + n_tokens`. 
This uses `attn_order=12` which includes `prime_attention` layers with keys/values from lyrics and queries from audio. 
If you instead want to use a model with the usual encoder-decoder style transformer, use `small_sep_enc_dec_prior`.<br>
hpsの選択を簡単にするために、ここでは `1b_lyrics` モデルのような `single_enc_dec` モデルを使用しています。これは、歌詞ボキャブラとvq-vaeボキャブラを単一の より大きなボキャブを使用し、歌詞トークンとvq-vaeコードを平坦化して、長さ `n_ctx + n_tokens` の単一のシーケンスにします。
これは `attn_order=12` を利用しており、`prime_attention` レイヤには歌詞のキー/値と音声のクエリが含まれています。
代わりに通常のエンコーダー・デコーダー形式の変換器を用いたい場合は、`small_sep_enc_dec_prior` を用います。

For sampling, follow same instructions as [above](#sample-from-new-model) but use `small_single_enc_dec_prior` instead of 
`small_prior`. To also get the alignment between lyrics and samples in the saved html, you'll need to set `alignment_layer` 
and `alignment_head` in `small_single_enc_dec_prior`. To find which layer/head is best to use, run a forward pass on a training example,
save the attention weight tensors for all prime_attention layers, and pick the (layer, head) which has the best linear alignment 
pattern between the lyrics keys and music queries. <br>
サンプリングの方法は、[上記](#sample-from-new-model)と同じですが、`small_prior` の代わりに `small_single_enc_dec_prior` を使います。
また、保存したhtmlの歌詞とサンプルの位置合わせを行うには、`small_single_enc_dec_prior`で `alignment_layer` と `alignment_head` を設定する必要があります。
どのレイヤー/ヘッドを使うのが最適かを知るためには、学習例に対してフォワードパスを実行してみてください。すべての prime_attention レイヤーの注目度テンソルを保存し、最も直線的なアライメントを持つ (layer, head) を選択します。歌詞のキーと音楽のクエリの間のパターン。

### Fine-tune pre-trained top-level prior to new style(s)
Previously, we showed how to train a small top-level prior from scratch. Assuming you have a GPU with at least 15 GB of memory and support for fp16, you could fine-tune from our pre-trained 1B top-level prior. Here are the steps:<br>
前回までに、小さなトップレベルの先行技術をスクラッチからトレーニングする方法を紹介しました。少なくとも15GB以上のメモリとFP16をサポートするGPUを持っていると仮定すると、事前に訓練された1Bトップレベルの先行技術を使って微調整することができます。以下に手順を示します。

- Support `--labels=True` by implementing `get_metadata` in `jukebox/data/files_dataset.py` for your dataset.
- Add new entries in `jukebox/data/ids`. We recommend replacing existing mappings (e.g. rename `"unknown"`, etc with styles of your choice). This uses the pre-trained style vectors as initialization and could potentially save some compute.
- `jukebox/data/files_dataset.py` で `get_metadata` を実装することで `--labels=True` をサポートします。
- jukebox/data/idに新しいエントリを追加します。既存のマッピングを置き換えることを推奨します (例: `"unknown"` などの名前を任意のスタイルに変更する)。これは、事前に学習されたスタイルベクトルを初期化に使用し、計算量を節約できる可能性があります。

After these modifications, run <br>
これらの修正を行った後、次のように実行します。
```
mpiexec -n {ngpus} python jukebox/train.py --hps=vqvae,prior_1b_lyrics,all_fp16,cpu_ema --name=finetuned \
--sample_length=1048576 --bs=1 --aug_shift --aug_blend --audio_files_dir={audio_files_dir} \
--labels=True --train --test --prior --levels=3 --level=2 --weight_decay=0.01 --save_iters=1000
```
To get the best sample quality, it is recommended to anneal the learning rate in the end. Training the 5B top-level requires GPipe which is not supported in this release.<br>
最高のサンプル品質を得るためには、最終的に学習率をアニーリングすることをお勧めします。5Bのトップレベルの学習にはGPipeが必要ですが、このリリースではサポートされていません。

# Citation(引用)

Please cite using the following bibtex entry:<br>
以下のbibtexエントリーを参考にして引用してください：

```
@article{dhariwal2020jukebox,
  title={Jukebox: A Generative Model for Music},
  author={Dhariwal, Prafulla and Jun, Heewoo and Payne, Christine and Kim, Jong Wook and Radford, Alec and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2005.00341},
  year={2020}
}
```

# License 
[Noncommercial Use License](./LICENSE) 

It covers both released code and weights. <br>
リリースされたコードとウェイトの両方をカバーしています。

