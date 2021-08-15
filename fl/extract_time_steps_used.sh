base_dir=$1
echo $base_dir
experiment_dirs=( $base_dir/63* )

for experiment_dir in "${experiment_dirs[@]}"; do
    echo $experiment_dir
    grep "total time steps used" $experiment_dir/experiment.log | cut -d " " -f 6 > $experiment_dir/time_steps_used.txt
done
