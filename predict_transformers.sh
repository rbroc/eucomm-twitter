#!/bin/bash

for model in distilbert-base-uncased sentence-transformers/all-MiniLM-L6-v2 distilbert-base-uncased-finetuned-sst-2-english Twitter/twhin-bert-base cardiffnlp/twitter-roberta-base
do
    for tp in poisson nll
    do
        for om in like_count retweet_count quote_count reply_count
        do 
            for lr in 0.01 0.001 0.0001 0.00001 0.000001
            do 
                for batch in 8 16 64
                do
                    for warmup in 10 100
                    do
                        for fl in 1
                        do  
                            mid="$(cut -d'/' -f2 <<<"$model")"
                            echo $mid
                            python3 predict_engagement_transformer.py --model-id $mid --checkpoint $model --epochs 100 --train-examples-per-device $batch --eval-examples-per-device $batch --logging-steps 1000 --learning-rate $lr --weight-decay 0.001 --warmup-steps $warmup --freeze-layers $fl --metric $om --tweedie-p $tp
                            rm -rf ./models/transformers/*
                            rm -rf ./wandb/*
                        done
                    done
                done
            done
        done
    done
done

for model in distilbert-base-uncased sentence-transformers/all-MiniLM-L6-v2 distilbert-base-uncased-finetuned-sst-2-english
do
    for tp in poisson nll
    do
        for om in like_count retweet_count quote_count reply_count
        do 
            for lr in 0.01 0.001 0.0001 0.00001 0.000001
            do 
                for batch in 8 16
                do
                    for warmup in 10 100
                    do
                        for fl in 0
                        do  
                            mid="$(cut -d'/' -f2 <<<"$model")"
                            echo $mid
                            python3 predict_engagement_transformer.py --model-id $mid --checkpoint $model --epochs 100 --train-examples-per-device $batch --eval-examples-per-device $batch --logging-steps 1000 --learning-rate $lr --weight-decay 0.001 --warmup-steps $warmup --freeze-layers $fl --metric $om --tweedie-p $tp
                            rm -rf models/transformers/*
                            rm -rf ./wandb/*
                        done
                    done
                done
            done
        done
    done
done
rm -rf models/transformers/*