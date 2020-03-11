import spacy
from spacy.gold import GoldParse
from spacy.util import minibatch


def train_model(labels, examples, epochs=10):
    nlp = spacy.blank('ru')
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)
    for l in labels:
        ner.add_label(l)

    optimizer = nlp.begin_training()

    print("Training data:")
    for t in examples:
        # print(t['text'])
        for ls, le, lt in t['labels']:
            print(lt, ':', t['text'][ls: le])

    for e in range(epochs):
        for batch in minibatch([e for e in examples], size=4):
            # print([t['labels'] for t in batch])
            docs = [nlp(t['text']) for t in batch]
            goldparses = [GoldParse(d, labels=t['labels']) for d, t in zip(docs, batch)]
            losses = {}
            nlp.update(docs, goldparses, drop=0.2, losses=losses, sgd=optimizer)

    return nlp


def get_predictions(nlp, docs):
    from collections import Counter
    ner = nlp.get_pipe('ner')
    parsed_docs = [nlp.make_doc(t) for t in docs]
    beams = ner.beam_parse(parsed_docs, beam_width=16)

    results = []
    for text, doc, beam in zip(docs, parsed_docs, beams):
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
            max_per_type[etype] += 1
            if max_per_type[etype] < 5:
                labels.append((cstart, cend, etype))
            predicts.append((cstart, cend, doc[estart:eend].text, etype, escore))
            if 0.01 < escore < 0.99:
                unsure += abs(escore - 0.5)

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

    examples = json.loads(Path('data/example.json').read_text())
    train_set = [e for e in examples if e['labels']]
    nlp = train_model(['codex', 'person', 'term'], train_set)

    import pandas

    test_set = [e['text'] for e in examples if not e['labels']]
    results = get_predictions(nlp, test_set)
    results = sorted(results, key=lambda x: x['unsure'], reverse=True)
    for r in results:
        print(r['unsure'], r['text'][:100])

    r = results[0]
    df = pandas.DataFrame(r['predicts'], columns=['start', 'stop', 'text', 'label', 'score'])
    print(df[df.label != 'codex'])
