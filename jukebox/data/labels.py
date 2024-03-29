import torch as t
import numpy as np
from jukebox.data.artist_genre_processor import ArtistGenreProcessor
from jukebox.data.text_processor import TextProcessor

# Linear window heurisic to get a window of lyric_tokens
def get_relevant_lyric_tokens(full_tokens, n_tokens, total_length, offset, duration):
    if len(full_tokens) < n_tokens:
        tokens = [0] * (n_tokens - len(full_tokens)) + full_tokens
        indices = [-1] * (n_tokens - len(full_tokens)) + list(range(0, len(full_tokens)))
    else:
        assert 0 <= offset < total_length
        midpoint = int(len(full_tokens) * (offset + duration / 2.0) / total_length)
        midpoint = min(max(midpoint, n_tokens // 2), len(full_tokens) - n_tokens // 2)
        tokens = full_tokens[midpoint - n_tokens // 2:midpoint + n_tokens // 2]
        indices = list(range(midpoint - n_tokens // 2, midpoint + n_tokens // 2))
    assert len(tokens) == n_tokens, f"Expected length {n_tokens}, got {len(tokens)}"
    assert len(indices) == n_tokens, f"Expected length {n_tokens}, got {len(indices)}"
    assert tokens == [full_tokens[index] if index != -1 else 0 for index in indices]
    return tokens, indices

class EmptyLabeller():
    def get_label(self, artist=None, genre=None, lyrics=None, total_length=None, offset=None):
        y = np.array([], dtype=np.int64)
        info = dict(artist="n/a", genre="n/a", lyrics=[], full_tokens=[])
        return dict(y=y, info=info)

    def get_batch_labels(self, metas, device='cpu'):
        ys, infos = [], []
        for meta in metas:
            label = self.get_label()
            y, info = label['y'], label['info']
            ys.append(y)
            infos.append(info)

        ys = t.stack([t.from_numpy(y) for y in ys], dim=0).to(device).long()
        assert ys.shape[0] == len(metas)
        assert len(infos) == len(metas)
        return dict(y=ys, info=infos)

class Labeller():
    def __init__(self, max_genre_words, n_tokens, sample_length, v3=False, v3_ftune=False, jp=False, jpfull=False):
        print(f"!!!JP_lyrics : {jp} !!!")
        self.ag_processor = ArtistGenreProcessor(v3, v3_ftune, jp)
        self.text_processor = TextProcessor(v3, jp, jpfull)
        self.n_tokens = n_tokens
        self.max_genre_words = max_genre_words
        self.sample_length = sample_length
        self.label_shape = (4 + self.max_genre_words + self.n_tokens, )
        self.jp = jp

    def get_label(self, artist, genre, lyrics, total_length, offset):
        artist_id = self.ag_processor.get_artist_id(artist)
        genre_ids = self.ag_processor.get_genre_ids(genre)

        lyrics = self.text_processor.clean(lyrics)
        full_tokens = self.text_processor.tokenise(lyrics)
        tokens, _ = get_relevant_lyric_tokens(full_tokens, self.n_tokens, total_length, offset, self.sample_length)

        assert len(genre_ids) <= self.max_genre_words
        genre_ids = genre_ids + [-1] * (self.max_genre_words - len(genre_ids))
        y = np.array([total_length, offset, self.sample_length, artist_id, *genre_ids, *tokens], dtype=np.int64)
        assert y.shape == self.label_shape, f"Expected {self.label_shape}, got {y.shape}"
        info = dict(artist=artist, genre=genre, lyrics=lyrics, full_tokens=full_tokens)
        return dict(y=y, info=info)

    def get_aligned_lyrics(self, lyrics_df, total_length, offset, sr, duration):
        """
        lyrics_dfより、offsetに対応する歌詞を取得する

        Parameters
        ----------
        lyrics_df : df
            Aligned lyrics を格納したDF
        total_length : int
            1曲の長さ
        offset : float
            song_chunkの曲の開始時刻の経過時間
        sr : int
            Sampling rate
            total_length, offsetに乗じる(かける)ことで単位が秒になる
        duration : int
            AudioChunkの長さ

        Returns
        -------
        full_lyrics : str
            歌詞全体
        chunk_lyrics : str
            (offset)〜(offset+total_length)  までの歌詞チャンク
        """
        assert (offset + duration) <= total_length
        mid_sec = float((offset + (duration / 2)) / sr)
        mid_idx = np.argmin(np.abs(lyrics_df.mid - mid_sec))

        front_lyrics_df = lyrics_df[:mid_idx]
        back_lyrics_df = lyrics_df[mid_idx:]

        lyrics_lang = 'hira' if self.jp==True else 'roma'
        full_lyrics_list = [str(word) for word in lyrics_df[lyrics_lang]]
        front_lyrics_list = [str(word) for word in front_lyrics_df[lyrics_lang]]
        back_lyrics_list = [str(word) for word in back_lyrics_df[lyrics_lang]]
        
        full_lyrics = ' '.join(full_lyrics_list)
        front_lyrics = ' '.join(front_lyrics_list)
        back_lyrics = ' '.join(back_lyrics_list)
        back_lyrics = ' ' + back_lyrics  # Insert to connect front_lyrics and back_lyrics with a space.

        return full_lyrics, front_lyrics, back_lyrics

    def get_label_train(self, artist, genre, lyrics, total_length, offset, sr):
        """get_label for train phase"""
        full_lyrics, front_lyrics, back_lyrics = self.get_aligned_lyrics(lyrics_df=lyrics, total_length=total_length,\
                                                            offset=offset, sr=sr, duration=self.sample_length)

        artist_id = self.ag_processor.get_artist_id(artist)
        genre_ids = self.ag_processor.get_genre_ids(genre)

        full_lyrics = self.text_processor.clean(full_lyrics)
        front_lyrics = self.text_processor.clean(front_lyrics)
        back_lyrics = self.text_processor.clean(back_lyrics)

        full_tokens = self.text_processor.tokenise(full_lyrics)
        front_tokens = self.text_processor.tokenise(front_lyrics)
        back_tokens = self.text_processor.tokenise(back_lyrics)

        n_tokens_sep = self.n_tokens // 2
        # Fill n_tokens as much as possible with lyrics corresponding to AudioChunk.
        if len(full_tokens) < self.n_tokens:
            tokens = [0]*(self.n_tokens-len(full_tokens)) + full_tokens
        elif len(front_tokens) < n_tokens_sep:
            tokens = front_tokens + back_tokens[:(self.n_tokens-len(front_tokens))]
        elif len(back_tokens) < n_tokens_sep:
            tokens = front_tokens[-(self.n_tokens-len(back_tokens)):] + back_tokens
        else:
            tokens = front_tokens[-n_tokens_sep:] + back_tokens[:n_tokens_sep]

        assert len(tokens) == self.n_tokens, f"{len(tokens)} != {self.n_tokens}"
        assert len(genre_ids) <= self.max_genre_words
        genre_ids = genre_ids + [-1] * (self.max_genre_words - len(genre_ids))
        y = np.array([total_length, offset, self.sample_length, artist_id, *genre_ids, *tokens], dtype=np.int64)
        assert y.shape == self.label_shape, f"Expected {self.label_shape}, got {y.shape}"
        info = dict(artist=artist, genre=genre, lyrics=full_lyrics, full_tokens=full_tokens)
        return dict(y=y, info=info)

    def get_label_sample(self, artist, genre, lyrics, total_length, offset):
        artist_id = self.ag_processor.get_artist_id(artist)
        genre_ids = self.ag_processor.get_genre_ids(genre)

        lyrics = self.text_processor.clean(lyrics)
        full_tokens = self.text_processor.tokenise(lyrics)
        tokens, _ = get_relevant_lyric_tokens(full_tokens, self.n_tokens, total_length, offset, self.sample_length)

        assert len(genre_ids) <= self.max_genre_words
        genre_ids = genre_ids + [-1] * (self.max_genre_words - len(genre_ids))
        y = np.array([total_length, offset, self.sample_length, artist_id, *genre_ids, *tokens], dtype=np.int64)
        assert y.shape == self.label_shape, f"Expected {self.label_shape}, got {y.shape}"
        info = dict(artist=artist, genre=genre, lyrics=lyrics, full_tokens=full_tokens)
        return dict(y=y, info=info)

    def get_y_from_ids(self, artist_id, genre_ids, lyric_tokens, total_length, offset):
        assert len(genre_ids) <= self.max_genre_words
        genre_ids = genre_ids + [-1] * (self.max_genre_words - len(genre_ids))
        if self.n_tokens > 0:
            assert len(lyric_tokens) == self.n_tokens
        else:
            lyric_tokens = []
        y = np.array([total_length, offset, self.sample_length, artist_id, *genre_ids, *lyric_tokens], dtype=np.int64)
        assert y.shape == self.label_shape, f"Expected {self.label_shape}, got {y.shape}"
        return y

    def get_batch_labels(self, metas, device='cpu'):
        ys, infos = [], []
        for meta in metas:
            label = self.get_label_sample(**meta)
            y, info = label['y'], label['info']
            ys.append(y)
            infos.append(info)

        ys = t.stack([t.from_numpy(y) for y in ys], dim=0).to(device).long()
        assert ys.shape[0] == len(metas)
        assert len(infos) == len(metas)
        return dict(y=ys, info=infos)

    def set_y_lyric_tokens(self, ys, labels):
        info = labels['info']
        assert ys.shape[0] == len(info)
        if self.n_tokens > 0:
            # total_length, offset, duration):
            tokens_list = []
            indices_list = []  # whats the index of each current character in original array
            for i in range(ys.shape[0]):
                full_tokens = info[i]['full_tokens']
                total_length, offset, duration = ys[i, 0], ys[i, 1], ys[i, 2]
                tokens, indices = get_relevant_lyric_tokens(full_tokens, self.n_tokens, total_length, offset, duration)
                tokens_list.append(tokens)
                indices_list.append(indices)
            ys[:, -self.n_tokens:] = t.tensor(tokens_list, dtype=t.long, device='cuda')
            return indices_list
        else:
            return None

    def describe_label(self, y):
        assert y.shape == self.label_shape, f"Expected {self.label_shape}, got {y.shape}"
        y = np.array(y).tolist()
        total_length, offset, length, artist_id, *genre_ids = y[:4 + self.max_genre_words]
        tokens = y[4 + self.max_genre_words:]
        artist = self.ag_processor.get_artist(artist_id)
        genre = self.ag_processor.get_genre(genre_ids)
        lyrics = self.text_processor.textise(tokens)
        return dict(artist=artist, genre=genre, lyrics=lyrics)


if __name__ == '__main__':
    labeller = Labeller(5, 512, 8192*8*4*4, v3=False, jp=False)
    label = labeller.get_label("Alan Jackson", "Country Rock", "old town road", 4*60*44100, 0)
    print(label, labeller.describe_label(label['y']))

    labeller = Labeller(1, 384, 6144*8*4*4, v3=True, jp=False)
    label = labeller.get_label("Alan Jackson", "Country Rock", "old town road", 4*60*44100, 0)
    print(label, labeller.describe_label(label['y']))

    labeller = Labeller(1, 384, 6144*8*4*4, v3=True, jp=True)
    lyrics = 'だれもがみなへいこうせん そんなあわいじゅんしんも'
    label = labeller.get_label("Alan Jackson", "Country Rock", lyrics=lyrics, total_length=4*60*44100, offset=0)
    print(label, labeller.describe_label(label['y']))





