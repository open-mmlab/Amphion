# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $exp_dir))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

######## Parse the Given Parameters from the Commond ###########
options=$(getopt -o c:n:s --long gpu:,reference_folder:,generated_folder:,dump_folder:,metrics: -- "$@")
eval set -- "$options"

while true; do
  case $1 in
    # Reference Audio Folder
    --reference_folder) shift; ref_dir=$1 ; shift ;;
    # Generated Audio Folder
    --generated_folder) shift; deg_dir=$1 ; shift ;;
    # Result Dumping Folder
    --dump_folder) shift; dump_dir=$1 ; shift ;;
    # Metrics to Compute
    --metrics) shift; metrics=$1 ; shift ;;

    --) shift ; break ;;
    *) echo "Invalid option: $1" exit 1 ;;
  esac
done

######## Calculate Objective Metrics ###########
CUDA_VISIBLE_DEVICES=$gpu python "$work_dir"/bins/calc_metrics.py \
    --ref_dir $ref_dir
    --deg_dir $deg_dir
    --dump_dir $dump_dir
    --metrics $metrics
    --fs 