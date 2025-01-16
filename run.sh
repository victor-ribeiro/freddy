#/bin/bash

num_workers=20
epochs=200
beta=0.1
e=0
size=.1
# for size in 0.1 0.25 0.5 0.75;
# do
#     python crest_train.py --num_workers $num_workers --epochs $epochs --resume_from_epoch $e --freddy_similarity 'codist' --train_frac $size --selection_method grad_freddy --alpha $alpha --beta $beta
#     python crest_train.py --num_workers $num_workers --epochs $epochs --resume_from_epoch $e --freddy_similarity 'codist' --train_frac $size --selection_method freddy --alpha $alpha --beta $beta
#     python crest_train.py --num_workers $num_workers --epochs $epochs --resume_from_epoch $e --train_frac $size --selection_method crest
#     python crest_train.py --num_workers $num_workers --epochs $epochs --resume_from_epoch $e --train_frac $size --selection_method random
# done

# CONTROLE
python crest_train.py --num_workers $num_workers --selection_method none  &
python crest_train.py --num_workers $num_workers --epochs $epochs --train_frac $size --selection_method crest  &
python crest_train.py --num_workers $num_workers --epochs $epochs --train_frac $size --selection_method random &


# BETA = 0
# feitos (alpha): 1, 1.25, 1.5

# BETA = 0.1
# feitos (alpha):

for alpha in  1 1.25 1.5;
do
    python crest_train.py --num_workers $num_workers --epochs $epochs --freddy_similarity 'similarity' --train_frac $size --selection_method grad_freddy --alpha $alpha --beta $beta &
    python crest_train.py --num_workers $num_workers --epochs $epochs --freddy_similarity 'similarity' --train_frac $size --selection_method freddy --alpha $alpha  --beta $beta &
done
