#!/bin/bash

for e in EU_Commission 10DowningStreet POTUS ecb IMFNews UN
do
    echo $e
    python3 preprocess.py --entity $e
done