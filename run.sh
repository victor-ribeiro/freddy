#/bin/bash

# num_workers=10
# epochs=2000
num_workers=2
epochs=20


# for size in 0.1 0.25 0.5 0.75;
for size in 0.1;
do
    python crest_train.py --num_workers $num_workers --epochs $epochs --freddy_similarity 'similarity' --train_frac $size --selection_method freddy --alpha .25 --beta .75
    python crest_train.py --num_workers $num_workers --epochs $epochs --freddy_similarity 'similarity' --train_frac $size --selection_method grad_freddy --alpha .25 --beta .75
    # python crest_train.py --num_workers $num_workers --epochs $epochs --train_frac $size --selection_method crest
    # python crest_train.py --num_workers $num_workers --epochs $epochs --train_frac $size --selection_method random
done
