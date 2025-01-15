#/bin/bash

num_workers=20
epochs=50
alpha=.75
beta=.05
# num_workers=2
# epochs=20


for size in 0.1 0.25 0.5 0.75;
do
    # for size in 0.1;
    for e in 0 50 100 150 200 250 300 350 400 450 500 550;
    do
        python crest_train.py --num_workers $num_workers --epochs $epochs --resume_from_epoch $e --freddy_similarity 'codist' --train_frac $size --selection_method grad_freddy --alpha $alpha --beta $beta
        python crest_train.py --num_workers $num_workers --epochs $epochs --resume_from_epoch $e --freddy_similarity 'codist' --train_frac $size --selection_method freddy --alpha $alpha --beta $beta
        python crest_train.py --num_workers $num_workers --epochs $epochs --resume_from_epoch $e --train_frac $size --selection_method crest
        python crest_train.py --num_workers $num_workers --epochs $epochs --resume_from_epoch $e --train_frac $size --selection_method random
    done
    python crest_train.py --num_workers $num_workers --selection_method none --resume_from_epoch $e
done

