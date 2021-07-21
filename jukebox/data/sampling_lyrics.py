import pandas as pd
from pprint import pprint
import os
import json

def input_meta(offset:int, total_length:int) -> list:

    path_wav = "wav_dataset_005"
    sampling_file_name = "1b_lyrics_v3_artist.csv"
    lyric_file_name = "free_lyric_01.json"
    df_sampling_list = pd.read_csv(f"/workspace/dataset/{path_wav}/sampling_list/{sampling_file_name}", header=0)
    lyric_path = os.path.join('/', 'workspace', 'dataset', path_wav, 'input_jukebox', lyric_file_name)

    # 歌詞を読み込み
    with open(lyric_path, mode='r', encoding='utf-8') as f:
        lyric = json.load(f)
    lyrics_use = lyric['lyric_roma']

    metas = []
    for idx in df_sampling_list.index:
        dic_sample = dict(artist=df_sampling_list.loc[idx, 'artist_alphabet_name'],
                            genre="j-pop",
                            lyrics=lyrics_use,
                            total_length=total_length,
                            offset=offset,
                            )
        metas.append(dic_sample)

    #pprint(metas)
    return metas


if __name__ == '__main__':
    input_meta(offset=0, total_length=90)