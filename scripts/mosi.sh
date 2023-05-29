set -e
modality=$1
run_idx=$2

cmd="python main.py --dataset=mosi --run_id=$run_idx
--batch_size=128 --clip=0.8 --num_epochs=100
--distribute --modalities=$modality"
echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh