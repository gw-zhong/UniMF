set -e
modality=$1
run_idx=$2

cmd="python main.py --dataset=urfunny --run_id=$run_idx
--batch_size=16 --clip=1.0 --num_epochs=20
--distribute --modalities=$modality"
echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh