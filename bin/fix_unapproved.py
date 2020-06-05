import argparse

from anno.anno import Client, get_project_by_id, init_django


def main(args):
    init_django()

    client = Client(get_project_by_id(id=args.project_id))
    client.fix_unapproved()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("project_id")
    parsed = parser.parse_args()
    main(parsed)
