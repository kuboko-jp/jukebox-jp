import pandas as pd
from pprint import pprint
import os
import json
from pprint import pprint

def input_meta(offset:int, total_length:int, input_new_lyric=True, 
               base_dir='/workspace/dataset/wav_dataset_005/', jp=False) -> list:
    """
    サンプリング時に入力するメタ情報を指定
    
    Parameters
    ----------
        offset : int 
        total_length : int
        input_new_lyric : bool
            True -> Training時に入力していない、新しい歌詞を入力する。
            False -> メタ情報で指定するArtistの曲における学習済みの歌詞を入力する。
        base_dir : str
            Base Directory.
    
    Returns
    -------
    metas : list
        生成する曲のメタ情報をディクショナリ形式で格納したリスト。
    """
    
    sampling_path = os.path.join(base_dir, "sampling_list/title_list_0.csv")
    df_sampling_list = pd.read_csv(sampling_path, header=0)

    metas = []
    for idx in df_sampling_list.index:
        artist_alphabet_name = df_sampling_list.loc[idx, 'artist_alphabet_name']
        genre = df_sampling_list.loc[idx, 'genre']

        if input_new_lyric:
            lyric_path = os.path.join(base_dir, "input_jukebox/back_number_short.json")
        else:
            lyric_file_name = df_sampling_list.loc[idx, 'lyric_file_name']
            lyric_path = os.path.join(base_dir, "lyric_data", f"{lyric_file_name}.json")
        with open(lyric_path, mode='r', encoding='utf-8') as f:
            lyric = json.load(f)
        lyric_lang = 'lyric_hira' if jp==True else 'lyric_roma'
        input_lyric = lyric[lyric_lang]

        dic_sample = dict(artist=artist_alphabet_name, genre=genre,
                          lyrics=input_lyric, total_length=total_length, offset=offset,)
        metas.append(dic_sample)
    return metas

if __name__ == '__main__':
    input_metas = input_meta(offset=0, total_length=240, input_new_lyric=True, 
               base_dir='/workspace/dataset/wav_dataset_006/', jp=True)
    pprint(input_metas)