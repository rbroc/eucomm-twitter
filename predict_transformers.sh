#!/bin/bash

for model in distilbert-base-uncased-finetuned-sst-2-english distilbert-base-uncased sentence-transformers/all-MiniLM-L6-v2 Twitter/twhin-bert-base cardiffnlp/twitter-roberta-base
do
    for tp in 1.0 1.2 1.4 1.6 1.8 2.0
    do
        for om in like_count retweet_count quote_count reply_count
        do 
            for lr in 0.01 0.001 0.0001 0.00001 0.000001
            do 
                for batch in 16
                do
                    for warmup in 100
                    do
                        for fl in 0
                        do  
                            mid="$(cut -d'/' -f2 <<<"$model")"
                            echo $mid
                            python3 predict_engagement_transformer.py --model-id $mid --checkpoint $model --epochs 100 --train-examples-per-device $batch --eval-examples-per-device $batch --logging-steps 1000 --learning-rate $lr --weight-decay 0.001 --warmup-steps $warmup --freeze-layers $fl --metric $om --tweedie-p $tp
                        done
                    done
                done
            done
        done
    done
done

for model in distilbert-base-uncased-finetuned-sst-2-english distilbert-base-uncased sentence-transformers/all-MiniLM-L6-v2
do
    for tp in 1.0 1.2 1.4 1.6 1.8 2.0
    do
        for om in like_count retweet_count quote_count reply_count
        do 
            for lr in 0.01 0.001 0.0001 0.00001 0.000001
            do 
                for batch in 16
                do
                    for warmup in 100
                    do
                        for fl in 1
                        do  
                            mid="$(cut -d'/' -f2 <<<"$model")"
                            echo $mid
                            python3 predict_engagement_transformer.py --model-id $mid --checkpoint $model --epochs 100 --train-examples-per-device $batch --eval-examples-per-device $batch --logging-steps 1000 --learning-rate $lr --weight-decay 0.001 --warmup-steps $warmup --freeze-layers $fl --metric $om --tweedie-p $tp
                        done
                    done
                done
            done
        done
    done
done
