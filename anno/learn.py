import sys

from anno.anno import Client, get_project_by_id, init_django
from anno.trainer import train_model, get_predictions


class Learner:
    def __init__(self, project):
        self.client = Client(project)

    def run(self, texts, max_add=None):
        labels = self.client.get_labels()
        docs = self.client.get_docs()
        print("Docs to train:", len(docs))
        nlp = train_model(labels, docs)
        results = get_predictions(nlp, texts)
        # print([r['predicts'] for r in results])
        results = sorted(results, key=lambda x: x['unsure'], reverse=True)
        self.client.del_unapproved(max_add)
        self.client.add_docs(results, max_add=max_add)


def get_records():
    records = load_lenta('data/lenta-ru-news.csv.gz')
    # texts = [t['text'] for t in test_set]
    test_set = []
    for i, r in enumerate(records):
        test_set.append(r.title + '\n' + r.text)
        if i >= 200:
            break
    return test_set


if __name__ == '__main__':
    init_django()
    learner = Learner(get_project_by_id(id=2))

    from corus import load_lenta

    texts = get_records()
    learner.run(texts, max_add=int(sys.argv[1]))
