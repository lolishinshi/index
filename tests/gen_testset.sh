#!/usr/bin/bash

python3 tests/gen_testset.py -s 0.5 -m 2 -n 2 -c 50 tests/dataset/images tests/dataset/search

python3 tests/gen_testset.py -s 1.0 -m 5 -n 5 -c 50 tests/dataset/images tests/dataset/search

python3 tests/gen_testset.py -s 0.5 -m 5 -n 5 -c 50 tests/dataset/images tests/dataset/search

python3 tests/gen_testset.py -s 2.0 -m 5 -n 5 -c 50 tests/dataset/images tests/dataset/search