#!/bin/bash

for model in distilbert-base-uncased-finetuned-sst-2-english
do
    for om in sum_count
    do 
        for lr in 1e-2 1e-3 1e-5 1e-6
        do 
            for batch in 16 64
            do
                for warmup in 100
                do
                    for fl in 0 1
                    do  
                        mid="$(cut -d'/' -f2 <<<"$model")"
                        echo $mid
                        python3 fit_transformers.py --model-id $mid --checkpoint $model --epochs 100 --train-examples-per-device $batch --eval-examples-per-device $batch --logging-steps 100 --learning-rate $lr --weight-decay 0.001 --warmup-steps $warmup --freeze-layers $fl --metric $om
                        rm -rf ./models/transformers/*
                        rm -rf ./wandb/*
                    done
                done
            done
        done
    done
done



# distilbert-base-uncased
# sentence-transformers/all-MiniLM-L6-v2 