#!/usr/bin/python
# coding=utf-8


def manual_iter():
    with open('system_config.yaml') as f:
        while True:
            line = next(f)
            if line is None:
                break
            print(line)


if __name__ == '__main__':
    manual_iter()
