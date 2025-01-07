#!/bin/bash


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
    cd submodlib
    pip install .
fi
