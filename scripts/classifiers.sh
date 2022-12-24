#!/bin/bash
for run in 0 1 2 3 4
do
    for model in distilbert-base-uncased
    do
        for om in EUCouncil Europarl_EN OECD scotgov 10DowningStreet ecb IMFNews UN
        do 
            for lr in 1e-5
            do 
                for batch in 64
                do
                    for warmup in 100
                    do
                        for fl in 1
                        do  
                            mid="$(cut -d'/' -f2 <<<"$model")"
                            echo $mid
                            python3 fit_transformer_classifier.py --model-id $mid --checkpoint $model --epochs 100 --train-examples-per-device $batch --eval-examples-per-device $batch --logging-steps 1000 --learning-rate $lr --weight-decay 0.001 --warmup-steps $warmup --freeze-layers $fl --entity $om --run $run
                            rm -rf ./wandb/*
                            rm -rf logs/classifier/$om*
                            rm -rf logs/classifier/$om
                        done
                    done
                done
            done
        done
    done
done



# distilbert-base-uncased-finetuned-sst-2-english