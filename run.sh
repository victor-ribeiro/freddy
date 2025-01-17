#/bin/bash

num_workers=12
epochs=40
beta=.1
alpha=1
size=.1
# for size in 0.1 0.25 0.5 0.75;
# do
#     python crest_train.py --num_workers $num_workers --epochs $epochs  --freddy_similarity 'codist' --train_frac $size --selection_method grad_freddy --alpha $alpha --beta $beta
#     python crest_train.py --num_workers $num_workers --epochs $epochs  --freddy_similarity 'codist' --train_frac $size --selection_method freddy --alpha $alpha --beta $beta
#     python crest_train.py --num_workers $num_workers --epochs $epochs  --train_frac $size --selection_method crest
#     python crest_train.py --num_workers $num_workers --epochs $epochs  --train_frac $size --selection_method random
# done

# CONTROLE
# python crest_train.py --num_workers $num_workers --selection_method none --epochs $epochs  &
# python crest_train.py --num_workers $num_workers --epochs $epochs --train_frac $size --selection_method crest   &
# python crest_train.py --num_workers $num_workers --epochs $epochs --train_frac $size --selection_method random  &


# BETA = 0
# feitos (alpha): 1, 1.25, 1.5

# BETA = 0.1
# feitos (alpha):


# for alpha in .25 .5 .75 1 2 3 5 10 20;
# do
python crest_train.py --num_workers $num_workers --epochs $epochs --freddy_similarity 'similarity' --train_frac $size --selection_method grad_freddy --alpha $alpha --beta $beta &
# python crest_train.py --num_workers $num_workers --epochs $epochs --freddy_similarity 'similarity' --train_frac $size --selection_method freddy --alpha $alpha  --beta $beta &
# done