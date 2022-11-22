#!/bin/bash

python3 predict_engagement_xgb.py --out-metric like_count
python3 predict_engagement_xgb.py --out-metric retweet_count
python3 predict_engagement_xgb.py --out-metric quote_count
python3 predict_engagement_xgb.py --out-metric reply_count
