from collections import defaultdict
import os
import torch
import time
try :
    from tqdm import tqdm
except ImportError:
    pass
from code_dataset import create_dataset
from code_model import create_model
from code_config.parser import parse
try:
    from code_record.visualizer import Visualizer
except ImportError:
    pass
from code_util.data.read_save import save_test_image
from code_util.util import get_file_name,generate_paths_from_list

def test(status_config = None, common_config = None):

    config,common_config = parse("test",status_config = status_config, common_config=common_config) 
    config["record"]["validation"] = False
    
    # dataset
    dataset, _ = create_dataset(config)  # create a dataset given dataset_mode and other configurations

    # model
    model = create_model(config)      # create a model given opt.model and other options
    model.setup(config)               # regular setup: load and print networks; create schedulers

    # create a website
    if config.get("docker",{}).get("use_docker") != True:
        dataset = tqdm(dataset, desc="Testing")
        visualizer = Visualizer(config)    

    # test with eval mode. This only affects layers like batchnorm and dropout.
    model.eval("test")
    epoch_iter = 0

    use_html = config["record"].get("html",{}).get("use_html",False) 
    use_tensorboard = config["record"].get("tensorboard",{}).get("use_tensorboard",False) 

    # # Save all test results locally
    save_list = config["result"].get("save_list",["fake_B","fake_A"])
    
    from torch.utils.flop_counter import FlopCounterMode
    from contextlib import nullcontext

    B_cfg = int(config["dataset"]["dataloader"]["batch_size"])
    use_ft16 = bool(config["model"].get("use_ft16"))
    time_cfg = config["record"].get("time",{})
    warmup_iters = int(time_cfg.get("warmup_iters", 3))
    measure_iters = int(time_cfg.get("measure_iters", 10))
    measure_flops_cfg = bool(time_cfg.get("measure_flops",True))

    autocast_ctx = (
        torch.amp.autocast('cuda', dtype=torch.float16)
        if use_ft16 else nullcontext()
    )

    for i, data in enumerate(dataset):
        epoch_iter += 1
        model.set_input(data)

        measure_flops = ((epoch_iter == 1) and measure_flops_cfg == True)  # 只统计一次

        if measure_flops:
            # ==========================================================
            # Warm-up (not timed)
            # ==========================================================
            for _ in range(warmup_iters):
                with torch.no_grad(), autocast_ctx:
                    model.test()

            torch.cuda.synchronize()

            # ==========================================================
            # Timed runs
            # ==========================================================
            times = []

            for _ in range(measure_iters):
                torch.cuda.synchronize()
                start_time = time.perf_counter()

                with torch.no_grad(), autocast_ctx:
                    model.test()

                torch.cuda.synchronize()
                end_time = time.perf_counter()

                times.append(end_time - start_time)

            avg_batch_latency = sum(times) / len(times)
            per_sample_latency = avg_batch_latency / max(B_cfg, 1)

            # ==========================================================
            # FLOPs (only once)
            # ==========================================================
            with torch.no_grad(), autocast_ctx, FlopCounterMode(display=True) as fc:
                model.test()

            total_flops = fc.get_total_flops()
            per_sample_flops = float(total_flops) / max(B_cfg, 1)
            per_sample_gflops = per_sample_flops / 1e9

            # ==========================================================
            # Record
            # ==========================================================
            visualizer.record_log(
                {
                    "GFLOPs_per_sample": per_sample_gflops,
                    "FLOPs_per_sample": per_sample_flops,
                    "Latency_batch_ms": avg_batch_latency * 1000.0,
                    "Latency_per_sample_ms": per_sample_latency * 1000.0,
                    "benchmark_warmup_iters": warmup_iters,
                    "benchmark_measure_iters": measure_iters,
                    "batch_size": B_cfg,
                },
                phase="test"
            )
        else:
            with torch.no_grad(), autocast_ctx:
                model.test()

        # Display results to HTML if needed
        if use_html:
            if epoch_iter % config["record"]["html"]["display_per_iter"] == 0:
                # print('processing (%04d)-th image... %s' % (i, img_paths))
                visualizer.display_on_html(model.get_current_visuals(), data["A"]["params"]["path"], phase = "test")
        if use_tensorboard:
            if epoch_iter % config["record"]["tensorboard"]["display_per_iter"] == 0:
                visualizer.display_on_tensorboard(model.get_current_visuals(), step=epoch_iter, phase="test")
        if config["record"].get("CAM",{}).get("use_CAM",False):
            if epoch_iter % config["record"]["CAM"]["display_CAM_per_iter"] == 0:
                # visualizer.draw_CAM(model,config,img_paths = img_paths)
                pass
        A_params = data["A"]["params"]
        save_test_image(model.get_current_results(), A_params, config, save_list)

    # if the dataset is not patch_wise, it must be a 2D dataset
    if config.get("reconstruction",{}).get("conduct_reconstruction",False) == True:
        data_format = config["dataset"]["data_format"]
        result_dir = config["work_dir"]
        # reconstruct the whole volume from 2D images
        from code_util.dataset.reconstruct import recontruct_3D_from_2D_4folder
        twoD_indicator = config["model"]["dim"]
        twoD_dir = os.path.join(result_dir,twoD_indicator,"synthesis") # result 2D images
        pattern = config["reconstruction"]["pattern_2D"]
        
        threeD_dir = os.path.join(result_dir,"3D/synthesis")
        # ref_pos = generate_paths_from_list(config["dataset"]["dataset_position"],postfix="3D/"+config["phase"])
        if config["dataset"].get("dir_A",None) is not None:
            ref_folder  = [os.path.join(config["dataset"]["dataroot"],config["dataset"]["dir_A"]).replace(twoD_indicator,"3D")]
        else:
            ref_folder = generate_paths_from_list(config["dataset"]["dataset_position"],postfix="3D/"+config["phase"]+"A") # ref_folder指定了模态 是不灵活的 但是方便 因此暂时这样
        recontruct_3D_from_2D_4folder(twoD_dir, threeD_dir, pattern, data_format, ref_folder)
    
    if config.get("segmentation",{}).get("conduct_segmentation",False) == True:
        from code_util.data.totalseg import totalseg_segmentation_batch
        seg_input_folder_base = os.path.join(result_dir,"3D/synthesis")
        task = config["segmentation"].get("task","total")
        seg_output_folder_base = os.path.join(result_dir,"3D/segmentation",task)
        seg_list = ["fake_B","fake_A"]
        for seg_item in seg_list:
            seg_input_folder = os.path.join(seg_input_folder_base,seg_item)
            if not os.path.exists(seg_input_folder):
                print("Segmentation input folder does not exist: {}, skip.".format(seg_input_folder))
                continue
            seg_output_folder = os.path.join(seg_output_folder_base,seg_item)
            print("Totalseg segmentation starts, input folder: {}, output folder: {}".format(seg_input_folder,seg_output_folder))
            ml = config["segmentation"].get("ml",True)
            gpu = config["model"]["gpu_ids"][0]
            device = "gpu:{}".format(gpu) if gpu >=0 else "cpu"
            modality = config["segmentation"]["modality"]
            if "ct" in modality:
                pass
            elif "mr" in modality:
                task = task + "_mr"
            else:
                raise ValueError("modality should contain 'ct' or 'mr'")
            totalseg_segmentation_batch(seg_input_folder, seg_output_folder, modality, ml = ml, task = task, device = device)
            print("Totalseg segmentation ends.")
  
    return common_config
    
if __name__ == '__main__':
    test()