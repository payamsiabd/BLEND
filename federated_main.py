from collections import defaultdict

from utils.fed_utils import average_weights, count_parameters, show_results, save_acc_csv, aggregate_local_models
from Dassl.dassl.utils import setup_logger, set_random_seed
from Dassl.dassl.config import get_cfg_default
from Dassl.dassl.engine import build_trainer
import numpy as np
import argparse
import random
import torch
import time
import copy
import os
import os
import json
from pathlib import Path
import gc
import copy

# import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# print("🔎 Waiting for VS Code to attach on :5678 ...")
# debugpy.wait_for_client() 

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root
    
    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg, args):
    from yacs.config import CfgNode as CN


    cfg.TRAINER.BLEND = CN()
    cfg.TRAINER.BLEND.TEXT_CTX_INIT = ""  # initialization words
    cfg.TRAINER.BLEND.PREC = "amp"  # fp16, fp32, amp
    cfg.TRAINER.BLEND.ADAPTER_LAYERS= args.adapter_layers
    cfg.TRAINER.BLEND.ADAPTER_DIM = args.adapter_dim
    cfg.TRAINER.BLEND.IS_SHARED = args.is_shared
    cfg.TRAINER.BLEND.ADAPTER_SCALE = 0.1
    cfg.TRAINER.BLEND.TOKEN_SEL_IDX = args.token_idx
    cfg.TRAINER.BLEND.ADAPTER_LAYERS_TXT = args.adapter_layers_txt
    cfg.TRAINER.BLEND.GAMMA_IMPORTANCE = args.gamma_importance
    cfg.TRAINER.BLEND.LAMBDA = args.Lambda
    cfg.TRAINER.BLEND.PER_LORA_DIM = args.per_lora_dim


    cfg.DATASET.SUBSAMPLE_CLASSES = args.subsample  # all, base or new
    cfg.DATASET.USERS = args.num_users  # number of clients
    cfg.DATASET.DIR_ALPHA = args.dir_alpha
    cfg.DATASET.USER_PROMPT_LENGTHS = []
    
    cfg.DATASET.IID = args.iid  # is iid
    cfg.DATASET.PARTITION = args.partition

    cfg.DATASET.USEALL = args.useall  # use all data for training instead of few shot
    cfg.DATASET.NUM_SHOTS = args.num_shots

    cfg.DATASET.BETA = args.beta
    cfg.DATASET.REPEATRATE = 0.0  # repeat rate on each client

    cfg.OPTIM.ROUND = 1  # global round
    cfg.OPTIM.GAMMA = args.gamma  # gamma of single-step
    cfg.OPTIM.ROUND = args.global_rounds
    cfg.OPTIM.LR = args.lr
    cfg.OPTIM.MAX_EPOCH = args.local_epoch 

    cfg.MODEL.BACKBONE.PRETRAINED = True




def setup_cfg(args):
    cfg = get_cfg_default()

    extend_cfg(cfg, args)
    cfg.set_new_allowed(True)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. set batch size
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size
    
    # 3. From input arguments
    reset_cfg(cfg, args)

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    
    random.seed(cfg.SEED)
 
     
    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)
    
    # cfg.OUTPUT_DIR = f"output/{cfg.DATASET.NAME}/{args.trainer}/shot_{args.num_shots}/beta_{args.beta}/ep{cfg.OPTIM.MAX_EPOCH}_r{cfg.OPTIM.ROUND}/alpha{args.alpha}_ratio{args.ratio}/seed_{args.seed}"
    
     
    
    cfg.freeze()

    return cfg

def write_results(args, test_accuracy):
    extention = ""
    if args.personalized_test:
        extention = "personalized"
    else:
        extention = "global"
    results = {
         "accuracy": {str(u): float(v[0]['accuracy']) for u, v in zip(range(len(test_accuracy)), test_accuracy)},
         "Loss" :  {str(u): float(v[1]) for u, v in zip(range(len(test_accuracy)), test_accuracy)},
    }
    path = os.path.join(args.output_dir,"results", args.subsample, extention)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "federated_results.json"), "w") as f:
        json.dump(results, f, indent=2)

def main(args):
 
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    # args.para_dir = setup_logger(cfg)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # print_args(args, cfg)
    global_representation_state = [[] for i in range(args.num_users)]
    local_representation_state = [[] for i in range(args.num_users)]
 

    local_weights_per = [{} for i in range(cfg.DATASET.USERS)]
    global_test_acc_list = []
    global_test_error_list = []
    global_test_f1_list = []
    global_epoch_list = []
    global_time_list = []
    local_trainer = build_trainer(args, cfg)

    local_weights = [[] for i in range(args.num_users)]
 
    global_prompt = None
    # local_trainer.train(idx=0, global_epoch=0, is_fed=True)

    local_trainer.fed_before_train()
    count_parameters(local_trainer.model, "prompt_learner")
    count_parameters(local_trainer.model, "image_encoder")
    count_parameters(local_trainer.model, "text_encoder")

    datanumber_client = []
    if args.trainer == 'CLIP':
        global_weights = copy.deepcopy(local_trainer.model.state_dict())
    else:
        for net_i in range(cfg.DATASET.USERS):
            # local_trainer = build_trainer(cfg)
            datanumber_client.append(len(local_trainer.fed_train_loader_x_dict_new[net_i].dataset))
        global_weights = copy.deepcopy(local_trainer.model.state_dict())

    if args.trainer == "BLEND":
        # 1) find + save keys
        model_state = local_trainer.model.state_dict()
        global_keys = [
            k for k in model_state.keys()
            if "shared_adapter" in k or "text_adapter" in k or "visual_adapter" in k 
        ]
        local_keys=[k for k in model_state.keys() if ("proj_adapter" in k)]

        print(f"Global_keys :{global_keys}")
        print(f"local_keys :{local_keys}")
    # Training
    start_epoch = 0
    end_epoch = cfg.OPTIM.ROUND
    global_test_acc_dict = {}
    global_time_list = []
    local_test = args.personalized_test
    start = time.time()
    for epoch in range(start_epoch, end_epoch):

        if args.trainer == 'CLIP':
            print("------------Global test start without training -------------")
            results = []

            idxs_users = list(range(cfg.DATASET.USERS))
            print("idxs_users:", idxs_users)

            # for idx in idxs_users:
            local_trainer.model.load_state_dict(global_weights, strict=False)
            data_loader = local_trainer.test_loader
            results.append(local_trainer.test(split = args.subsample, data_loader = data_loader ))
            
            # result = local_trainer.test(idx=idx)
            # results.append(result)

            write_results(args, results)

            global_test_acc = []
            global_test_error = []
            global_test_f1 = []
            for k in range(len(results)):
                global_test_acc.append(results[k][0]['accuracy'])
                global_test_error.append(results[k][0]["error_rate"])
                global_test_f1.append(results[k][0]["macro_f1"])
            global_time_list.append(time.time() - start)
            global_test_acc_list.append(sum(global_test_acc) / len(global_test_acc))
            global_test_error_list.append(sum(global_test_error) / len(global_test_error))
            global_test_f1_list.append(sum(global_test_f1) / len(global_test_f1))
            global_epoch_list.append(epoch)
            print("Global test acc:", sum(global_test_acc) / len(global_test_acc))
            print("Global test error:", sum(global_test_error) / len(global_test_error))
            print("Global test macro_f1:", sum(global_test_f1) / len(global_test_f1))
            print("------------local test finish-------------")
            return

        if args.trainer == 'BLEND':
            # global prompt + local prompt
            if args.eval_only:
                print("Loading models")
                idxs_users = list(range(0, cfg.DATASET.USERS))
                results = []
                local_weights_per = torch.load(args.output_dir + "/blend_save.pt")
                
                if local_test:
                    for idx in idxs_users:
                        local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                        data_loader = local_trainer.fed_test_loader_dict_new[idx]
                        results.append(local_trainer.test(split = args.subsample, data_loader =data_loader ))
                else:
                    for idx in idxs_users:
                        local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                        data_loader = local_trainer.test_loader
                        results.append(local_trainer.test(split = args.subsample, data_loader = data_loader ))

                write_results(args, results)

                global_test_acc = []
                global_test_error = []
                global_test_f1 = []
                for k in range(len(results)):
                    global_test_acc.append(results[k][0]['accuracy'])
                    global_test_error.append(results[k][0]["error_rate"])
                    global_test_f1.append(results[k][0]["macro_f1"])
                global_time_list.append(time.time() - start)
                global_test_acc_list.append(sum(global_test_acc) / len(global_test_acc))
                global_test_error_list.append(sum(global_test_error) / len(global_test_error))
                global_test_f1_list.append(sum(global_test_f1) / len(global_test_f1))
                global_epoch_list.append(epoch)
                print("Global test acc:", sum(global_test_acc) / len(global_test_acc))
                print("Global test error:", sum(global_test_error) / len(global_test_error))
                print("Global test macro_f1:", sum(global_test_f1) / len(global_test_f1))
                print("------------local test finish-------------")
                return
            else:
                if epoch == 0:
                    idxs_users = list(range(0, cfg.DATASET.USERS))
                else:
                    m = max(int(args.frac * args.num_users), 1)
                    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
                print("idxs_users", idxs_users)
                print("------------local train start epoch:", epoch, "-------------")
                
                for idx in idxs_users:
                    if epoch == 0:
                        local_trainer.model.load_state_dict(global_weights, strict=False)
                    else:
                        local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                    # for op_id in range(2):
                    #     local_trainer.set_training(op_id=op_id)
                    local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                    local_weight = local_trainer.model.state_dict()
                    global_representation_state[idx] = {k: local_weight[k].cpu() for k in global_keys}
                    local_representation_state[idx] = {k: local_weight[k].cpu() for k in local_keys}

                print("------------local train finish epoch:", epoch, "-------------")
                all_users = list(range(0, cfg.DATASET.USERS))
                global_weights = aggregate_local_models(global_representation_state, datanumber_client)
                for idx in all_users:
                    combined_weights = {**global_weights, **local_representation_state[idx]}
                    local_weights_per[idx] = combined_weights

                print("------------local test start-------------")
                results = []
                
                if epoch + 1== end_epoch:
                    torch.save(local_weights_per, args.output_dir + "/blend_save.pt")
                    for idx in all_users:
                        local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                        data_loader = local_trainer.fed_test_loader_dict_new[idx]
                        results.append(local_trainer.test(split = args.subsample, data_loader =data_loader ))
                    
                    write_results(args, results)

                    # global_test_acc = show_results(cfg, results, epoch)
                    global_test_acc = []
                    global_test_error = []
                    global_test_f1 = []
                    for k in range(len(results)):
                        global_test_acc.append(results[k][0]['accuracy'])
                        global_test_error.append(results[k][0]["error_rate"])
                        global_test_f1.append(results[k][0]["macro_f1"])
                    global_time_list.append(time.time() - start)
                    global_test_acc_list.append(sum(global_test_acc) / len(global_test_acc))
                    global_test_error_list.append(sum(global_test_error) / len(global_test_error))
                    global_test_f1_list.append(sum(global_test_f1) / len(global_test_f1))
                    global_epoch_list.append(epoch)
                    print("Global test acc:", sum(global_test_acc) / len(global_test_acc))
                    print("Global test error:", sum(global_test_error) / len(global_test_error))
                    print("Global test macro_f1:", sum(global_test_f1) / len(global_test_f1))
                    print("------------local test finish-------------")
                    
        
        

    for idx in idxs_users:
        local_trainer.fed_after_train()
    for key, global_test_acc_list in global_test_acc_dict.items():
        print(key, "global_test_acc_list:", global_test_acc_list)
        print(key, "maximum test acc:", max(global_test_acc_list))
        print(key, "mean of acc:", np.mean(global_test_acc_list[-5:]))
        print(key, "std of acc:", np.std(global_test_acc_list[-5:]))
    # save_acc_csv(local_trainer.args.para_dir, global_test_acc_dict, cfg)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=str, default="BLEND", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="ViT-B/16", help="name of CNN backbone")
    parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution')

    parser.add_argument('--num_users', type=int, default=1, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help='the fraction of clients: C')
    parser.add_argument('--gamma', type=float, default=1, help='gamma of single_step')
    parser.add_argument('--train_batch_size', type=int, default=16, help="number of trainer batch size")
    parser.add_argument('--test_batch_size', type=int, default=128, help="number of test batch size")
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")

    parser.add_argument('--ctx_init', default=False, help="is using the ctx init, set True for CLIP")

   
    # parameters of datasets
    # caltech101, oxford_flowers, oxford_pets, food101 and dtd
    parser.add_argument('--num_shots', type=int, default=16, help="number of shots in few shot setting")
    parser.add_argument('--useall', default=False, help="is useall, True for all training samples, False for few shot learning")
    parser.add_argument('--iid', default=False, help="is iid, control the iid of caltech101, oxford_flowers, oxford_pets, food101 and dtd")

    # cifar10, cifar100
    parser.add_argument('--partition', type=str, default='noniid-labeldir',
                        help='the data partitioning strategy of cifar10 and cifar100,'
                            ' select from "noniid-labeluni, noniid-labeldir,noniid-labeldir100"')

    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="", help="path to config file for dataset setup")

 
   
 
    # parameters of path
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument("--root", type=str, default="CoOp-main/DATA", help="path to dataset")
    parser.add_argument("--output_dir", type=str, default="", help="output directory")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", default =False, action="store_true", help="evaluation only")
    parser.add_argument("--personalized_test", default =False, action="store_true", help="evaluation only")
    parser.add_argument('--subsample', type=str, default='base', help="all,base,new") 
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")

    parser.add_argument(
        "--adapter_layers",
        type=int,
        nargs="+",          # allows multiple values (space-separated)
        default=[1,2,3,4,5,6,7,8,9,10,11,12],
        help="List of transformer layers where adapters are inserted"
    )
    parser.add_argument(
        "--adapter_layers_txt",
        type=int,
        nargs="+",          # allows multiple values (space-separated)
        default=[1,2,3,4,5,6,7,8,9,10,11,12],
        help="List of transformer layers where adapters are inserted"
    )
    parser.add_argument("--adapter_dim", default = 128, type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--per_lora_dim", default = 128, type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--is_shared", default =True, action="store_true", help="evaluation only")
    parser.add_argument("--token_idx", default = 0, type=int, help="load model weights at this epoch for evaluation")

    parser.add_argument("--dir_alpha", type=float, help="Dirichlet alpha", default = 0.01)
    parser.add_argument("--gamma_importance", type=float, help=" ", default = 0.90)
    parser.add_argument("--Lambda", type=float, help=" ", default = 0.50)
    
    parser.add_argument("--global_rounds", default = 50, type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--local_epoch", default = 1, type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--lr", default = 0.0015, type=float, help="load model weights at this epoch for evaluation")

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")
    
    args = parser.parse_args()
    
    for eval_only in [False, True]:
        args.eval_only = eval_only
        args.subsample = "new" if eval_only else "base"
        main(args)
