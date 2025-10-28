
import os
import argparse
import json
from pathlib import Path
from train_classification import launch_train
from test_classification import launch_test

def main():
    parser = argparse.ArgumentParser(description="3D classification training script")
    parser.add_argument(
        "-c",
        "--config-file",
        default="/config/config_train_32g.json",
        help="config json file that stores hyper-parameters",
    )

    
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    
    args = parser.parse_args()
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in config_dict.items():
        setattr(args, k, v)
    
    ddp_bool = args.gpus > 1  # whether to use distributed data parallel

    if ddp_bool:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        world_size = 1
        device = 0

    if rank == 0:
        
        os.makedirs(f"{args.root_dir}/StrokeUADiag/{config_dict['experiment_name']}/{config_dict['sub_experiment_name']}/models/", exist_ok=True)
        os.makedirs(f"{args.root_dir}/StrokeUADiag/tensorboard/{config_dict['sub_experiment_name']}/", exist_ok=True)


    for step in args.pipeline:

        if step == "train_classification":
            print(f"Launching classification training: {config_dict['experiment_name']}/{config_dict['sub_experiment_name']} with {args.gpus} gpus")
            launch_train(args)
        
        if step == "test_classification" and rank==0:
            print(f"Launching classification testing: {config_dict['experiment_name']}/{config_dict['sub_experiment_name']} with {args.gpus} gpus")
            launch_test(args)

        
            
    


if __name__ == "__main__":
    main()