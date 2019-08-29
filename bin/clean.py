#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import context
import os

if __name__ == '__main__':
    print("Remove saved weights...")
    files = os.listdir("saves/weights")
    for file in files:
        os.remove('saves/weights/' + file)
    print("Done")

    print("Remove saved losses...")
    files = os.listdir("saves/losses")
    for file in files:
        os.remove('saves/losses/' + file)
    print("Done")

    print("Remove saved architectures...")
    files = os.listdir("saves/architectures")
    for file in files:
        os.remove('saves/architectures/' + file)
    print("Done")
