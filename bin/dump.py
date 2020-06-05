import argparse
import json

from anno.anno import Client, get_project_by_id, init_django


def main(args):
    init_django()

    project = get_project_by_id(id=args.project_id)
    client = Client(project)
    docs = client.get_approved_docs()

    for doc in docs:
        print(json.dumps(doc, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("project_id")
    parsed = parser.parse_args()
    main(parsed)
