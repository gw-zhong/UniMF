set -e
modality=$1
run_idx=$2
n_trial=$3

cmd="python main.py --dataset=urfunny --run_id=$run_idx --trials=$n_trial
--batch_size=16 --clip=1.0 --num_epochs=20
--distribute --modalities=$modality"
echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh