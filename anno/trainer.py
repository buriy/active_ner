import spacy
from spacy.gold import GoldParse
from spacy.util import minibatch
from tqdm.auto import tqdm

from anno.ner import MyNER


def create_ner(nlp):
    return MyNER(nlp.vocab)


def train_model(labels, examples, epochs=10, verbose=False):
    nlp = spacy.blank('ru')
    ner = create_ner(nlp)
    nlp.add_pipe(ner, last=True)
    for l in labels:
        print("Label:", l)
        ner.add_label(l)

    optimizer = nlp.begin_training()

    if verbose:
        print("Training data:")
        for t in examples:
            # print(t['text'])
            for ls, le, lt in t['labels']:
                print('{} : "{}"'.format(lt, t['text'][ls: le]))

    for e in tqdm(range(epochs)):
        for batch in minibatch([e for e in examples], size=1):
            # print([t['labels'] for t in batch])
            docs = [nlp.tokenizer(t['text']) for t in batch]
            goldparses = [GoldParse(d, entities=t['labels']) for d, t in zip(docs, batch)]
            losses = {}
            nlp.update(docs, goldparses, drop=0.5, losses=losses, sgd=optimizer)

    return nlp


def get_predictions(nlp, docs):
    from collections import Counter
    ner = nlp.get_pipe('ner')
    parsed_docs = [nlp.make_doc(t) for t in docs]
    beams = [ner.beam_parse([x], beam_width=16)[0] for x in tqdm(parsed_docs)]

    results = []
    for text, doc, beam in zip(docs, parsed_docs, beams):
        parsed_ents = doc.ents
        if doc.ents:
            print(text, doc.ents)
        entities = ner.moves.get_beam_annot(beam)
        words = Counter()
        for e, v in entities.items():
            estart, eend, etype = e
            etype = doc.vocab.strings[etype]

            words[estart, eend, etype] = v

        words_items = sorted(words.items(), key=lambda x: (-x[1], x[0]))
        labels = []
        predicts = []
        unsure = 0.001
        # print(repr(text))
        max_per_type = Counter()
        for (estart, eend, etype), escore in words_items:
            cstart = doc[estart].idx
            if eend == len(doc):
                cend = len(text)
            else:
                cend = doc[eend].idx
                # cend = doc[eend-1].idx + len(doc[eend].text)
            # print(cstart, cend, estart, eend, f"'{doc[estart:eend]}', '{text[cstart:cend]}'", escore)
            # assert doc[estart:eend].text.strip() == text[cstart:cend].strip()
            unsure += 0.5 - abs(escore - 0.5)
            if escore > 0.01:  # 0.4 <= escore:
                max_per_type[etype] += 1
                if max_per_type[etype] < 100:
                    labels.append((cstart, cend, etype))
                predicts.append((cstart, cend, doc[estart:eend].text, etype, escore))

        results.append({
            'text': text,
            'labels': labels,
            'unsure': unsure / len(text),
            'predicts': predicts,
        })

    return results


if __name__ == '__main__':
    import json
    from pathlib import Path

    examples = [json.loads(l) for l in Path('data/examples.jsonl').read_text().strip().split('\n')]
    train_set = [e for e in examples if e['annotation_approver']]
    nlp = train_model(['at', 'dur'], train_set, verbose=True)

    import pandas

    test_set = [e['text'] for e in examples if not e['annotation_approver']]
    results = get_predictions(nlp, test_set)
    results = sorted(results, key=lambda x: x['unsure'], reverse=True)
    for r in results:
        print(r['unsure'], r['text'][:100])

    r = results[0]
    df = pandas.DataFrame(r['predicts'], columns=['start', 'stop', 'text', 'label', 'score'])
    print(df)
