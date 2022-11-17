#!/bin/bash

# If used with a scheduling system like slurm enter job allocation here

##################### CHANGES HERE (start) ####################
CONFIG_DIR="/dir/of/your/config/"
CONFIG_NAME="name_of_your_config.yaml"
##################### CHANGES HERE (stop) ####################

CONFIG_PATH=$CONFIG_DIR$CONFIG_NAME

TIMESTAMP=`date +%Y-%m-%dT%H-%M-%S`
TEMP_CONFIG_PATH=$CONFIG_DIR$TIMESTAMP$CONFIG_NAME

cp $CONFIG_PATH $TEMP_CONFIG_PATH

##################### CHANGES HERE (start) ####################
for i in {0..31}
do
    python /path/to/flexpart_safe_start.py $TEMP_CONFIG_PATH $i &
done
##################### CHANGES HERE (stop) ####################

wait
rm $TEMP_CONFIG_PATH