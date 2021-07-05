import pandas as pd
from pprint import pprint
import os
import json

def input_meta(offset:int, total_length:int) -> list:

    path_wav = "wav_dataset_004"
    sampling_no = 0
    df_sampling_list = pd.read_csv(f"/workspace/dataset/{path_wav}/sampling_list/title_list_{str(sampling_no)}.csv", header=0)
    lyric_path = os.path.join('/', 'workspace', 'dataset', path_wav, 'input_jukebox', 'free_lyric_01.json')

    # 歌詞を読み込み
    with open(lyric_path, mode='r', encoding='utf-8') as f:
        lyric = json.load(f)
    lyric_roma = lyric['lyric_hira']

    metas = []
    for idx in df_sampling_list.index:
        dic_sample = dict(artist=df_sampling_list.loc[idx, 'artist_alphabet_name'],
                            genre="j-pop",
                            lyrics=lyric_roma,
                            total_length=total_length,
                            offset=offset,
                            )
        metas.append(dic_sample)

    #pprint(metas)
    return metas


if __name__ == '__main__':
    input_meta(offset=0, total_length=90)