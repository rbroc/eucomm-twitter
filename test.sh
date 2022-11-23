#!/bin/bash

for model in distilbert-base-uncased
do
    for tp in 1.5
    do
        for om in like_count
        do 
            for lr in 0.01
            do 
                for batch in 16
                do
                    for warmup in 100
                    do
                        for fl in 1
                        do  
                            mid="$(cut -d'/' -f2 <<<"$model")"
                            echo $mid
                            python3 predict_engagement_transformer.py --model-id $mid --checkpoint $model --epochs 10 --train-examples-per-device $batch --eval-examples-per-device $batch --logging-steps 1000 --learning-rate $lr --weight-decay 0.001 --warmup-steps $warmup --freeze-layers $fl --metric $om --tweedie-p $tp
                        done
                    done
                done
            done
        done
    done
done
