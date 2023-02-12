#!/bin/bash

DIR="./gm_images"
if [[ -d "$DIR" ]]; then
    rm -r $DIR
fi
mkdir $DIR
echo "Create Directory ${DIR}"

if [ $# -eq 0 ]
    then
    echo "Executing glosten_milgrom_parameter_analysis.py"
    python glosten_milgrom_parameter_analysis.py -ap $DIR 

    echo "Executing inactive_trader_feature.py"
    python inactive_trader_feature.py -ap $DIR

    echo "Executing glosten_milgrom_p_eta.py"
    python glosten_milgrom_p_eta.py -ap $DIR --distribution 0.6 0.8 0.98
else
    echo "The number of files is: $#"
    for var in "$@"
    do
        echo "Executing $var ..."
        python $var -ap $DIR
    done
fi

echo "[[Finished!]]"