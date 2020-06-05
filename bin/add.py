import argparse
import datetime
import json
import sys

from tqdm.auto import tqdm

from anno.anno import Client, get_project_by_id, init_django


def main(args):
    init_django()

    client = Client(get_project_by_id(id=args.project_id))

    docs = [json.loads(text) for text in sys.stdin.readlines()]
    if args.nodups:
        existing_docs = set(client.get_doc_texts())
    else:
        existing_docs = []

    now = datetime.datetime.utcnow()

    added = 0
    for doc in tqdm(docs):
        if doc['text'] in existing_docs:
            # skip the document, it was already added
            continue
        client.add_doc(doc['text'], doc.get('meta', {}), doc['labels'], priority=1000, updated=now)
        added += 1
    print("Added:", added)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("project_id")
    parser.add_argument("--nodups", help="Don't allow duplicates. Can be slow on a large database.")
    parsed = parser.parse_args()
    main(parsed)
