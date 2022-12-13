#!/bin/bash

for e in coe scotgov NATO Europarl_EN EUCouncil OECD EU_Commission 10DowningStreet ecb IMFNews UN
do
    echo $e
    python3 split.py --entity $e
done