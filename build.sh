#!/bin/bash

home=$(pwd)

if [[ "$OSTYPE" == "darwin"* ]]; then
    conda env create -f environ/macos.yml
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    conda env create -f environ/linux.yml
else
    echo 'unknown OS'
fi


if [ $? == 0 ]; then
    if [ ! -d submodlib ]; then
        git clone https://github.com/decile-team/submodlib.git
    fi
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate freddy
    cd $home/submodlib
    pip install .
fi


# usage: crest_train.py [-h] [--arch {resnet20,resnet18,resnet50}] [--data_dir DATA_DIR] [--dataset {cifar10,cifar100,tinyimagenet}] [--num_workers NUM_WORKERS] [--epochs N]
#                       [--resume_from_epoch RESUME_FROM_EPOCH] [--batch_size BATCH_SIZE] [--lr LR] [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY] [--save-dir SAVE_DIR] [--save_freq SAVE_FREQ]
#                       [--gpu GPU [GPU ...]] [--selection_method {none,random,crest}] [--smtk SMTK] [--train_frac TRAIN_FRAC] [--lr_milestones LR_MILESTONES [LR_MILESTONES ...]] [--gamma GAMMA]
#                       [--seed SEED] [--runs RUNS] [--warm_start_epochs WARM_START_EPOCHS] [--subset_start_epoch SUBSET_START_EPOCH] [--cache_dataset [CACHE_DATASET]]
#                       [--clean_cache_selection [CLEAN_CACHE_SELECTION]] [--clean_cache_iteration [CLEAN_CACHE_ITERATION]] [--approx_moment [APPROX_MOMENT]] [--approx_with_coreset [APPROX_WITH_CORESET]]
#                       [--check_interval CHECK_INTERVAL] [--num_minibatch_coreset NUM_MINIBATCH_CORESET] [--batch_num_mul BATCH_NUM_MUL] [--interval_mul INTERVAL_MUL]
#                       [--check_thresh_factor CHECK_THRESH_FACTOR] [--shuffle [SHUFFLE]] [--random_subset_size RANDOM_SUBSET_SIZE] [--partition_start PARTITION_START] [--drop_learned [DROP_LEARNED]]
#                       [--watch_interval WATCH_INTERVAL] [--drop_interval DROP_INTERVAL] [--drop_thresh DROP_THRESH] [--min_train_size MIN_TRAIN_SIZE] [--use_wandb [USE_WANDB]]
