import re
from unidecode import unidecode

class TextProcessor():
    def __init__(self, v3=False, jp=False, jpfull=False):
        if v3:
            if jpfull:
                print("Use en, jp, sy tokens.")
                jp_vocab = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉゃゅょっ"
                vocab = f'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:;!?-\'\"()[] \t\n{jp_vocab}'
                not_vocab = re.compile(f'[^A-Za-z0-9.,:;!?\-\'\"()\[\] \t\n{jp_vocab}]+')
            elif jp:
                print("Use jp tokens.")
                jp_vocab = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉゃゅょっ"
                vocab = f'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 \n{jp_vocab}'
                not_vocab = re.compile(f'[^{vocab}]+')
            else:
                vocab = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:;!?-\'\"()[] \t\n'
                not_vocab = re.compile('[^A-Za-z0-9.,:;!?\-\'\"()\[\] \t\n]+')
        else:
            vocab = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:;!?-+\'\"()[] \t\n'
            not_vocab = re.compile('[^A-Za-z0-9.,:;!?\-+\'\"()\[\] \t\n]+')
        print(f"【n_vocab】 : {len(vocab) + 1}")
        self.vocab = {vocab[index]: index + 1 for index in range(len(vocab))}
        self.vocab['<unk>'] = 0
        self.n_vocab = len(vocab) + 1
        self.tokens = {v: k for k, v in self.vocab.items()}
        self.tokens[0] = ''  # <unk> became ''
        self.not_vocab = not_vocab

    def clean(self, text):
        #text = unidecode(text)  # Convert to ascii
        text = text.replace('\\', '\n')
        text = self.not_vocab.sub('', text)  # Remove non vocab
        return text

    def tokenise(self, text):
        return [self.vocab[char] for char in text]

    def textise(self, tokens):
        return ''.join([self.tokens[token] for token in tokens])

    def characterise(self, tokens):
        return [self.tokens[token] for token in tokens]
