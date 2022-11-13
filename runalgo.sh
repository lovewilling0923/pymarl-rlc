#!/bin/bash

envs=(3m 8m 2s3z 3s5z 1c3s5z)

for e in "${envs[@]}"
do
   for i in {0..9}
   do
      python src/main.py --config=$1 --env-config=sc2 with env_args.map_name=$e seed=$i &
      echo "Running with $1 and $e for seed=$i"
      sleep 2s
   done
done

# demo sc2
python src/main.py --config=mixrts --env-config=sc2 with env_args.map_name=MMM2 gpu_id=0 t_max=2010000 epsilon_anneal_time=50000 seed=0
# demo stag_hunt
python3 src/main.py --config=mixrts --env-config=stag_hunt with env_args.map_name=pp-2 env_args.miscapture_punishment=-2 gpu_id=0
# demo foraging
python3 src/main.py --config=qnam --env-config=foraging with env_args.map_name=lbf-4 seed=0 gpu_id=0