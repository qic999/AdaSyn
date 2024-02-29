import os
import subprocess
import sys
sys.path.append(os.getcwd())
from connectomics.config.defaults import get_cfg_defaults
import argparse
from tensorboardX import SummaryWriter
import time

def infer_syn(model_dir, model_id, config_base, config_file):
    command = "python scripts/main.py --config-base\
            {}\
            --config-file\
            {}\
            --inference\
            --checkpoint\
            {}checkpoint_{:05d}.pth.tar\
            --opts\
            SYSTEM.NUM_GPUS\
            1\
            SYSTEM.NUM_CPUS\
            3\
            INFERENCE.SAMPLES_PER_BATCH\
            4\
            INFERENCE.INPUT_SIZE\
            [128,128,128]\
            INFERENCE.OUTPUT_SIZE\
            [128,128,128]\
            INFERENCE.STRIDE\
            [64,64,64]\
            INFERENCE.PAD_SIZE\
            [64,64,64]\
            INFERENCE.AUG_NUM\
            None\
            INFERENCE.INPUT_PATH\
            /data/qic99/ISBI23_code/synapse_detection/data/training_set\
            INFERENCE.IMAGE_NAME\
            train_sample3_vol0/img.tif\
        ".format(config_base, config_file, model_dir, model_id)
    out = subprocess.run(command, shell=True)
    print(command, "\n |-------------| \n", out, "\n |-------------| \n")

def get_args():
    r"""Get args from command lines.
    """
    parser = argparse.ArgumentParser(description="Model Inference")
    parser.add_argument('--config-file', type=str, help='configuration file (yaml)')
    parser.add_argument('--config-base', type=str,
                        help='base configuration file (yaml)', default=None)

    args = parser.parse_args()
    return args

if __name__=="__main__":
    # print(sys.path)
    args = get_args()
    cfg = get_cfg_defaults()

    cfg.merge_from_file(args.config_base)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    # start_iter = cfg.SOLVER.START_SAVE
    start_iter = 160000
    end_iter = cfg.SOLVER.ITERATION_TOTAL
    step_iter = cfg.SOLVER.ITERATION_SAVE
    model_ids = range(start_iter, end_iter+step_iter, step_iter)
    # [267000, 241000, 233000, 121000, 137000, 145000] [129000, 166000, 145000, 136000, 99000, 72000, 65000, 47000]
    # model_ids = args.test_model_list # [210000, 187000, 354000] [209000, 249000, 145000, 121000] [92000, 97000, 98000] [267000, 241000, 233000, 121000, 137000, 145000]

    model_dir=cfg.DATASET.OUTPUT_PATH
    pre_dir=cfg.INFERENCE.OUTPUT_PATH

    start_time = time.time()
    for model_id in model_ids:
        pth_file =  "{}checkpoint_{:06d}.pth.tar".format(model_dir, model_id)
        # pre = '{}{:06d}_out_1_1024_1024_aug_None_pad_0.h5'.format(pre_dir, model_id)
        print(pth_file)
        # if os.path.exists(pre):
        #     continue
        score = infer_syn(model_dir, model_id, args.config_base, args.config_file)
        # break
    end_time = time.time()
    day = (end_time - start_time) // (24*60*60)
    hour = (end_time - start_time - day*(24*60*60)) // (60*60)
    minu = (end_time - start_time - day*(24*60*60) - hour*(60*60)) // 60
    total_time = print(f"{day}day {hour}hour {minu}min")
    print('total_time:', total_time)

    # model_id = 282000
    # args = get_args()
    # cfg = get_cfg_defaults()
    # cfg.merge_from_file(args.config_file)
    # cfg.freeze()

    # root_dir=cfg.DATASET.INFERENCE_PATH
    # model_dir=cfg.MODEL.SAVE_PATH
    # pre_dir=cfg.INFERENCE.OUTPUT_PATH
    # yaml_dir=cfg.DATASET.YMLY_PATH
    # score = cal_infer_epfl(root_dir, model_dir, model_id, pre_dir, yaml_dir, 1)


    