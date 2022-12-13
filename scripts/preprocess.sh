#!/bin/bash

for e in coe #EUCouncil Europarl_EN NATO OECD scotgov EU_Commission 10DowningStreet ecb IMFNews UN
do
    echo $e
    python3 preprocess.py --entity $e
done