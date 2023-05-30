set -e
modality=$1
run_idx=$2
subdataset=$3
n_trial=$4

cmd="python main.py --dataset=$subdataset --run_id=$run_idx --trials=$n_trial
--batch_size=128 --clip=0.8 --num_epochs=100
--distribute --modalities=$modality"
echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh