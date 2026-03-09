import collections
import torch


def build_vocab(texts):

    all_words = []

    for text_pair in texts:

        if isinstance(text_pair, tuple):

            for text in text_pair:
                all_words.extend(text.lower().replace('?', '').split())

        elif isinstance(text_pair, dict):

            all_words.extend(text_pair['prompt'].lower().replace('?', '').split())
            all_words.extend(text_pair['chosen'].lower().split())
            all_words.extend(text_pair['rejected'].lower().split())

    word_counts = collections.Counter(all_words)

    vocab = {'<pad>':0,'<unk>':1}

    for word,_ in word_counts.most_common():

        vocab[word] = len(vocab)

    return vocab



def tokenize_text(text,vocab):

    tokens = []

    for word in text.lower().replace('?','').split():

        tokens.append(vocab.get(word,vocab['<unk>']))

    return torch.tensor(tokens,dtype=torch.long).unsqueeze(0)
