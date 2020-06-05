import argparse
import json

from corus import load_lenta


def main(args):
    records = load_lenta('data/lenta-ru-news.csv.gz')
    for i, r in enumerate(records):
        print(json.dumps({'text': r.title + '\n' + r.text}, ensure_ascii=False))
        if i >= args.doc_count:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("doc_count", type=int, help="Max items to dump.")
    parsed = parser.parse_args()
    main(parsed)
