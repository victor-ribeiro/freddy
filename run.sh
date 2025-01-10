#/bin/bash

num_workers=10
epochs=2000
save_freq=200

for size in 0.1 0.25 0.5 0.75;
do
    python crest_train.py --num_workers $num_workers --epochs $epochs  --save_freq $save_freq --freddy_similarity 'codist' --train_frac $size --selection_method freddy
    python crest_train.py --num_workers $num_workers --epochs $epochs  --save_freq $save_freq --train_frac $size --selection_method crest
    python crest_train.py --num_workers $num_workers --epochs $epochs  --save_freq $save_freq --train_frac $size --selection_method random
done
