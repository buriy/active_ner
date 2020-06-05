import argparse
import datetime
import time

from tqdm.auto import tqdm

from anno.anno import Client, get_project_by_id, init_django
from anno.trainer import train_model, get_predictions


def train(client):
    labels = client.get_labels()
    docs = client.get_approved_docs()
    print("Docs to train:", len(docs))
    model = train_model(labels, docs)
    return model


def main(args):
    init_django()
    project = get_project_by_id(id=args.project_id)
    while True:
        client = Client(project)
        model = train(client)

        docs = client.get_unapproved_docs(with_anno=False, limit=args.max_update)
        results = get_predictions(model, docs)
        results = sorted(results, key=lambda x: x['unsure'], reverse=True)

        now = datetime.datetime.utcnow()

        for r in tqdm(results, desc="Updating DB"):
            doc = r['document']
            priority = int(1000 - r['unsure'] * 1000)  # 0 is the most urgent, 1 is the least urgent
            updated = now - datetime.timedelta(seconds=priority)
            # print("Predicted labels:", r['labels'])
            status = client.update_doc(id=doc['id'], labels=r['labels'], priority=priority, updated=updated)
            if not status:
                print("Document", doc['id'], "was already marked as annotated. Skipping.")

        if not args.watch:
            break
        approved_count = client.get_unapproved_doc_count()
        while True:
            time.sleep(1)
            new_count = client.get_unapproved_doc_count()
            if new_count != approved_count:
                print("Unapproved documents count changed, now we have:", new_count)
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("project_id", action="store")
    parser.add_argument("-m", "--max_update", action="store", type=int, default=100000)
    parser.add_argument("-w", "--watch", action="store_true",
                        help="Watchdog: reapply the auto-labeling when adding/approving a document.")
    parsed = parser.parse_args()
    main(parsed)
