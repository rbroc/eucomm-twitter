#!/bin/bash

for model in distilbert-base-uncased sentence-transformers/all-MiniLM-L6-v2 distilbert-base-uncased-finetuned-sst-2-english #Twitter/twhin-bert-base cardiffnlp/twitter-roberta-base
do
    for tp in 1.01 1.20 1.40 1.60 1.80 1.99
    do
        for om in like_count retweet_count quote_count reply_count sum_count
        do 
            for lr in 1e-2 1e-3 1e-5 1r-6
            do 
                for batch in 16 64
                do
                    for warmup in 100
                    do
                        for fl in 0 1
                        do  
                            mid="$(cut -d'/' -f2 <<<"$model")"
                            echo $mid
                            python3 predict_engagement_transformer.py --model-id $mid --checkpoint $model --epochs 100 --train-examples-per-device $batch --eval-examples-per-device $batch --logging-steps 10 --learning-rate $lr --weight-decay 0.001 --warmup-steps $warmup --freeze-layers $fl --metric $om --tweedie-p $tp
                            rm -rf ./models/transformers/*
                            rm -rf ./wandb/*
                        done
                    done
                done
            done
        done
    done
done

