#/bin/bash

num_workers=20
epochs=30
# alpha=.95
beta=.15
e=0
size=.1
# for size in 0.1 0.25 0.5 0.75;
# do
#     python crest_train.py --num_workers $num_workers --epochs $epochs --resume_from_epoch $e --freddy_similarity 'codist' --train_frac $size --selection_method grad_freddy --alpha $alpha --beta $beta
#     python crest_train.py --num_workers $num_workers --epochs $epochs --resume_from_epoch $e --freddy_similarity 'codist' --train_frac $size --selection_method freddy --alpha $alpha --beta $beta
#     python crest_train.py --num_workers $num_workers --epochs $epochs --resume_from_epoch $e --train_frac $size --selection_method crest
#     python crest_train.py --num_workers $num_workers --epochs $epochs --resume_from_epoch $e --train_frac $size --selection_method random
# done
# python crest_train.py --num_workers $num_workers --selection_method none --resume_from_epoch $e


for alpha in 0.1 0.25 0.5 0.75 1 1.5 2;
do
    python crest_train.py --num_workers $num_workers --epochs $epochs --freddy_similarity 'codist' --train_frac $size --selection_method grad_freddy --alpha $alpha
    python crest_train.py --num_workers $num_workers --epochs $epochs --freddy_similarity 'codist' --train_frac $size --selection_method freddy --alpha $alpha
done