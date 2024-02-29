import os
import subprocess
import sys
sys.path.append(os.getcwd())
from connectomics.config.defaults import get_cfg_defaults
import argparse
from tensorboardX import SummaryWriter
import time

def infer_syn(model_dir, model_id, config_base, config_file, test_volume_name):
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
            INFERENCE.IMAGE_NAME\
            {}\
        ".format(config_base, config_file, model_dir, model_id, test_volume_name)
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

    model_dir=cfg.DATASET.OUTPUT_PATH
    pre_dir=cfg.INFERENCE.OUTPUT_PATH

    start_time = time.time()
    model_id=185000
    pth_file =  "{}checkpoint_{:06d}.pth.tar".format(model_dir, model_id)
    print(pth_file)
    test_volume_dict = {'test_sample1_vol0/img_zyx_5500-6100_6000-6600_1800-2400.h5':[5500,6000,1800],
                    'test_sample1_vol1/img_zyx_3400-4000_6962-7562_3865-4465.h5':[3400,6962,3865],
                    'test_sample1_vol2/img_zyx_3166-3766_7531-8131_2440-3040.h5':[3166,7531,2440],
                    'test_sample2_vol0/img_zyx_13070-13470_6940-7340_3370-3770.h5':[13070,6940,3370],
                    'test_sample2_vol1/img_zyx_7138-7538_5790-6190_6948-7348.h5':[7138,5790,6948],
                    'test_sample2_vol2/img_zyx_6800-7400_1800-2400_4200-4800.h5':[6800,1800,4200],
                    'test_sample3_vol0/img_zyx_945-1345_3247-3647_4643-5043.h5':[945,3247,4643],
                    'test_sample3_vol1/img_zyx_2304-2720_2976-3392_6304-6720.h5':[2304,2976,6304],
                    'test_sample3_vol2/img_zyx_2688-3104_5408-5824_2944-3360.h5':[2688,5408,2944],
                    }
    for test_volume_name in test_volume_dict.keys():
        # breakpoint()
        # if os.path.exists(os.path.join(pre_dir,'submission',test_volume_name.split('/')[0], test_volume_name.split('/')[0]+'_predict.h5')):
        #     continue
        infer_syn(model_dir, model_id, args.config_base, args.config_file, test_volume_name)


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


    