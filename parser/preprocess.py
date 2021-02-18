# encoding: utf8
"""Convert .conllu into word-level parsing data"""

import os
import re
import tarfile
from collections import Counter

UNK = "UNK"
BOS = "BOS"
EOS = "EOS"

# Courtesy Kipperwasser & Goldberg
numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");

def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()
# Thank you!


def parse_conllu_blob(blob):
    blob = blob.decode('utf8').strip()

    # sentences are separated by one empty line:
    sent_blobs = blob.split("\n\n")
    return (parse_conllu_sent(sent_blob)
            for sent_blob in sent_blobs)


def parse_conllu_sent(sent_blob):
    metadata = {}
    toks = []
    for line in sent_blob.splitlines():
        if line.startswith("#"):
            line = line[1:]
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip()
            metadata[key] = val
        else:
            toks.append(parse_conllu_line(line))

    return metadata, toks


def parse_conllu_line(line):
    keys = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats',
            'head', 'deprel', 'deps', 'misc']

    return {key: val for key, val in zip(keys, line.split("\t"))}


paths = {
    'ja': {
        'train': 'ud-treebanks-conll2017/UD_Japanese/ja-ud-train.conllu',
        'valid': 'ud-treebanks-conll2017/UD_Japanese/ja-ud-dev.conllu',
        'test': [
            'ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/ja.conllu',
            'ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/ja_pud.conllu',
        ]
    },

    'zh': {
        'train': 'ud-treebanks-conll2017/UD_Chinese/zh-ud-train.conllu',
        'valid': 'ud-treebanks-conll2017/UD_Chinese/zh-ud-dev.conllu',
        'test': [
            'ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/zh.conllu',
        ]
    },

    'vi': {
        'train': 'ud-treebanks-conll2017/UD_Vietnamese/vi-ud-train.conllu',
        'valid': 'ud-treebanks-conll2017/UD_Vietnamese/vi-ud-dev.conllu',
        'test': [
            'ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/vi.conllu',
        ]
    },

    'en': {
        'train': 'ud-treebanks-conll2017/UD_English/en-ud-train.conllu',
        'valid': 'ud-treebanks-conll2017/UD_English/en-ud-dev.conllu',
        'test': [
            'ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/en.conllu',
        ]
    },

    'ro': {
        'train': 'ud-treebanks-conll2017/UD_Romanian/ro-ud-train.conllu',
        'valid': 'ud-treebanks-conll2017/UD_Romanian/ro-ud-dev.conllu',
        'test': [
            'ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/ro.conllu',
        ]
    }
}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lang', metavar='lang', choices=list(paths.keys()) + ["xx"])
    parser.add_argument('--root')
    parser.add_argument('--out', default="./data")
    opts = parser.parse_args()

    xx = False
    if opts.lang == "xx":
        opts.lang = "en"
        xx = True

    def extract(data, sort=False):
        out = []

        for metadata, toks in parse_conllu_blob(data):
            tok_forms = [tok['form'] for tok in toks]
            tok_upos = [tok['upos'] for tok in toks]
            tok_xpos = [tok['xpos'] for tok in toks]

            try:
                tree = [-1] + [int(tok['head']) for tok in toks]
            except:
                print("Skipping sentence", metadata['sent_id'])
                continue

            assert len(tok_forms) + 1 == len(tree)
            out.append((
                tok_forms,
                tok_upos,
                tok_xpos,
                tree,
                metadata['sent_id']))

        if sort:
            out.sort(key=lambda x: len(x[0]))

        return out

    # read train and dev
    fn = os.path.join(opts.root, 'ud-treebanks-conll2017.tgz')
    with tarfile.open(fn) as tarf:
        train = tarf.extractfile(paths[opts.lang]['train']).read()
        valid = tarf.extractfile(paths[opts.lang]['valid']).read()

    # read test
    fn = os.path.join(opts.root, 'ud-test-v2.0-conll2017.tgz')
    with tarfile.open(fn) as tarf:
        test = {test_fn: tarf.extractfile(test_fn).read()
                for test_fn in paths[opts.lang]['test']}

    if xx:
        opts.lang = "xx"
        train = train[:(15000 + train[15000:].index(b"\n\n"))]
        valid = train
        test = {key: train for key in test}

    out_tpl = os.path.join(opts.out, f"{opts.lang}-{{}}.txt")


    train_data = extract(train)
    valid_data = extract(valid)

    pos_counter = Counter(w
                          for _, sent, _, _, _ in train_data
                          for w in sent)

    train_counter = Counter(normalize(w)
                            for sent, _, _, _, _ in train_data
                            for w in sent)

    pos_vocab = [UNK, BOS, EOS] + [c for c, _ in pos_counter.most_common()]
    pos_bacov = {c: k for k, c in enumerate(pos_vocab)}

    vocab = [c for c, count in train_counter.most_common()]
    actual_vocab_set = set(vocab)

    vocab = [UNK, BOS, EOS] + vocab
    bacov = {c: k for k, c in enumerate(vocab)}

    def _vectorize(w):
        w = normalize(w)
        if w not in actual_vocab_set:
            return bacov[UNK]
        else:
            return bacov[w]

    def _vectorize_pos(p):
        return pos_bacov.get(p, pos_bacov[UNK])

    with open(out_tpl.format("vocab"), "w") as f:
        for c in vocab:
            print(c, train_counter.get(c, -1), sep="\t", file=f)

    with open(out_tpl.format("posvocab"), "w") as f:
        for c in pos_vocab:
            print(c, pos_counter.get(c, -1), sep="\t", file=f)

    def write_out(f, data):
        for sent, upos, _, tree, sent_id in data:
            toks = " ".join(str(_vectorize(w))
                            for w in sent)
            pos = " ".join(str(_vectorize_pos(p))
                           for p in upos)
            tree = " ".join(str(i) for i in tree)
            sent_ = " ".join(sent)

            assert "\t" not in toks
            assert "\t" not in pos
            assert "\t" not in tree
            assert "\t" not in sent_
            print("\t".join([sent_, toks, pos, tree]), file=f)

    with open(out_tpl.format("train"), "w") as f:
        write_out(f, train_data)

    with open(out_tpl.format("valid"), "w") as f:
        write_out(f, valid_data)

    for test_fn, test_blob in test.items():
        test_data = extract(test_blob)
        identifier = os.path.splitext(os.path.basename(test_fn))[0]

        with open(out_tpl.format("test-{}".format(identifier)), "w") as f:
            write_out(f, test_data)
