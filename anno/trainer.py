import random

import spacy
from spacy.gold import GoldParse
from spacy.language import Language
from spacy.util import minibatch, compounding
from tqdm.auto import tqdm
from typing import List

from anno.ner import MyNER


def create_ner(nlp):
    return MyNER(nlp.vocab)


def train_model(labels, examples, epochs=10, verbose=False):
    nlp = spacy.blank('ru')
    ner = create_ner(nlp)
    nlp.add_pipe(ner, last=True)
    for l in labels:
        print("Found label:", l)
        ner.add_label(l)

    optimizer = nlp.begin_training()

    if verbose:
        print("Training data:")
        for t in examples:
            # print(t['text'])
            for ls, le, lt in t['labels']:
                print('{} : "{}"'.format(lt, t['text'][ls: le]))

    sizes = compounding(1, 16, 1.001)
    for e in tqdm(range(epochs), desc="Training Epoch"):
        random.shuffle(examples)
        for batch in minibatch([e for e in examples], size=sizes):
            # print([t['labels'] for t in batch])
            docs = [nlp.tokenizer(t['text']) for t in batch]
            goldparses = [GoldParse(d, entities=t['labels']) for d, t in zip(docs, batch)]
            losses = {}
            nlp.update(docs, goldparses, drop=0.5, losses=losses, sgd=optimizer)

    return nlp


def get_predictions(nlp: Language, docs: List[dict]):
    from collections import Counter
    ner = nlp.get_pipe('ner')
    parses = list(nlp.pipe([t['text'] for t in docs]))
    beams = [ner.beam_parse([x], beam_width=16)[0] for x in tqdm(parses, desc="Predicting labels...")]

    results = []
    # print(type(docs), type(parses), type(beams))
    # print(len(docs), len(parses), len(beams))
    items = zip(docs, parses, beams)
    for document, parse, beam in items:
        text = document['text']
        # if parse.ents:
        #     print("Entities:", text, parse.ents)
        # else:
        #     print("No entities found:", text, parse.ents)
        entities = ner.moves.get_beam_annot(beam)
        words = Counter()
        start_end = {}
        for (estart, eend, etype), v in sorted(entities.items(), key=lambda x: (x[1], x[0])):
            etype_str = parse.vocab.strings[etype]
            if (estart, eend) in start_end:
                print("Removing completely overlapping entry:", (estart, eend, etype_str))
                continue
            words[estart, eend, etype_str] = v
            start_end[estart, eend] = True

        words_items = sorted(words.items(), key=lambda x: (-x[1], x[0]))
        labels = []
        predicts = []
        unsure = 0.001
        # print(repr(text))
        max_per_type = Counter()
        for (estart, eend, etype), escore in words_items:
            cstart = parse[estart].idx
            if eend == len(parse):
                cend = len(text)
            else:
                cend = parse[eend].idx
                # cend = parse[eend-1].idx + len(parse[eend].text)
            # print(cstart, cend, estart, eend, f"'{parse[estart:eend]}', '{text[cstart:cend]}'", escore)
            # assert parse[estart:eend].text.strip() == text[cstart:cend].strip()
            unsure += 0.5 - abs(escore - 0.5)
            if escore > 0.01:  # 0.4 <= escore:
                max_per_type[etype] += 1
                if max_per_type[etype] < 100:
                    labels.append((cstart, cend, etype))
                predicts.append((cstart, cend, parse[estart:eend].text, etype, escore))

        results.append({
            'document': document,
            'labels': labels,
            'unsure': unsure / len(text),
            'predicts': predicts,
        })

    return results


def main():
    import json
    from pathlib import Path

    examples = [json.loads(l) for l in Path('data/examples.jsonl').read_text().strip().split('\n')]
    train_set = [e for e in examples if e['annotation_approver']]
    nlp = train_model(['at', 'dur'], train_set, epochs=20, verbose=False)

    import pandas

    test_set = [e['text'] for e in examples if not e['annotation_approver']]
    results = get_predictions(nlp, test_set)
    results = sorted(results, key=lambda x: x['unsure'], reverse=True)
    for r in results:
        print(r['unsure'], r['text'][:140] + '...')

    r = results[0]
    print("Annotations for", r['text'], ':')
    df = pandas.DataFrame(r['predicts'], columns=['start', 'stop', 'text', 'label', 'score'])
    print(df)


if __name__ == '__main__':
    main()
