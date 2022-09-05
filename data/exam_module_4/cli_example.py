import argparse


def parse_cli_arguments():
    parser = argparse.ArgumentParser(
        u"Python PyiNotifier events watcher and logger"
    )
    parser.add_argument(u"--name1", required=True, type=str, help="help string 1")
    parser.add_argument(u"--name2", required=False, type=str, default='default_Val',
                        help="help string 2")
    parser.add_argument(u"--usr", required=True, type=str, help="Help String 1")
    parser.add_argument(u"--pwd", required=True, type=str, help="Help String 1")
    parser.add_argument(u"--env", required=True, type=str, help="Help String 1")
    return parser.parse_args()


def monitor():
    # pyinotify:
    parsed_cli_arguments = parse_cli_arguments()
    first_argument = parsed_cli_arguments.monitor_dir
    second_argument = parsed_cli_arguments.logs_dir

    try:
        ...
    except KeyboardInterrupt:
        ...
    finally:
        ...


if __name__ == "__main__":
    monitor()