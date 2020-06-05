import argparse

from anno.anno import Client, get_project_by_id, init_django


def main(args):
    init_django()
    client = Client(get_project_by_id(id=args.project_id))
    deleted = client.del_unapproved(args.delete_count)
    print("Deleted:", deleted)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("project_id", type=int)
    parser.add_argument("delete_count", type=int, help="Max items to delete.")
    parsed = parser.parse_args()
    main(parsed)
