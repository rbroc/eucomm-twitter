#!/bin/bash

python3 predict_engagement_xgb.py --out-metric sum_count --cv 1
python3 predict_engagement_xgb.py --out-metric sum_count --cv 0
