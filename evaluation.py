import os
from code_config.parser import parse
from code_util.util import generate_paths_from_list

def evaluation(status_config = None, common_config = None):
    
    # opt >>>> config
    config,common_config = parse("evaluation",status_config = status_config, common_config=common_config) 
    data_format = config["dataset"]["data_format"]
    result_dir = config["work_dir"]
    config["phase"] = "test"
    
    if config["reconstruction"]["conduct_reconstruction"] == True:
        from code_util.dataset.reconstruct import recontruct_3D_from_2D_4folder
        twoD_indicator = config["model"]["dim"]
        input_dir = os.path.join(result_dir,twoD_indicator,"synthesis") # result 2D images
        output_dir = os.path.join(result_dir,"3D/synthesis")
        pattern = config["reconstruction"]["pattern"]
        phase = config["phase"]
        if config["dataset"].get("dir_A",None) is not None:
            ref_folder  = [os.path.join(config["dataset"]["dataroot"],config["dataset"]["dir_A"])]
        else:
            ref_folder = generate_paths_from_list(config["dataset"]["dataset_position"],postfix="3D/"+config["phase"]+"A") # ref_folder指定了模态 是不灵活的 但是方便 因此暂时这样
        recontruct_3D_from_2D_4folder(input_dir, output_dir, pattern, data_format, ref_folder)
    
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
            gpu_ids = config["model"].get("gpu_ids",None)
            if gpu_ids is None:
                gpu = config["gpu"]
            else:
                gpu = gpu_ids[0]
            modality = config["segmentation"]["modality"]
            if "ct" in modality:
                pass
            elif "mr" in modality:
                task = task + "_mr"
            else:
                raise ValueError("modality should contain 'ct' or 'mr'")
            device = "gpu:{}".format(gpu) if gpu >=0 else "cpu"
            totalseg_segmentation_batch(seg_input_folder, seg_output_folder, modality,  ml = ml, task = task, device = device)
            print("Totalseg segmentation ends.")

    # calculate metrics
    if config["metrics"]["image_similarity"].get("calculate_metrics",False) == True:
        from code_util.metrics.calculate import calculate_folder,calculate_folder_SynthRAD2023
        cal_list = ["fake_B","fake_A"]
        ref_folder = generate_paths_from_list(config["dataset"]["dataset_position"],postfix="3D") # reference images
        for cal_item in cal_list:
            result_folder = os.path.join(result_dir,"3D/synthesis",cal_item) # generated images
            if not os.path.exists(result_folder):
                print("Result folder does not exist: {}, skip.".format(result_folder))
                continue
            phase = config["phase"]
            target_modality = cal_item.split("_")[-1]
            source_modality = "A" if target_modality == "B" else "B"
            source_folder = generate_paths_from_list(ref_folder,postfix=phase+source_modality)
            target_folder = generate_paths_from_list(ref_folder,postfix=phase+target_modality)
            mask_folfer = generate_paths_from_list(ref_folder,postfix="mask/"+phase)
            dynamic_range = config["metrics"]["image_similarity"].get("dynamic_range",None)
            metric_names = config["metrics"]["image_similarity"].get("metric_names", None)
            pattern = config["metrics"]["pattern"]
            # calculate_folder(result_folder, source_folder,target_folder, pattern, mask_folder = binary_mask_folfer, metric_names=metric_names, device_id=device_id)
            class_range = config["metrics"]["image_similarity"].get("class_range",None)
            calculate_folder_SynthRAD2023(result_folder, source_folder, target_folder, pattern, metric_names, mask_folder = mask_folfer, class_range=class_range, dynamic_range = dynamic_range, output_folder = result_dir, cal_item = cal_item)
    
    if config["metrics"].get("segmentation",{}).get("calculate_metrics",False) == True:
        from code_util.metrics.calculate import calculate_folder_segmentation
        seg_list = ["fake_B","fake_A"]
        phase = config["phase"]
        task = config["metrics"]["segmentation"].get("task","total")
        result_folder_base = os.path.join(result_dir,"3D/segmentation", task)
        ref_folder = generate_paths_from_list(config["dataset"]["dataset_position"],postfix="3D")
        target_folder_base = generate_paths_from_list(ref_folder,postfix="segmentation/"+task)
        mask_folfer = generate_paths_from_list(ref_folder,postfix="mask/"+phase)
        for seg_item in seg_list:
            result_folder = os.path.join(result_folder_base,seg_item)
            if not os.path.exists(result_folder):
                print("Result folder does not exist: {}, skip.".format(result_folder))
                continue
            target_modality = seg_item.split("_")[-1]
            target_folder = generate_paths_from_list(target_folder_base,postfix=phase+target_modality)
            metric_names = config["metrics"]["segmentation"].get("metric_names",["DICE"])
            pattern = config["metrics"]["pattern"]
            calculate_folder_segmentation(result_folder, target_folder, pattern, metric_names, mask_folder= mask_folfer, output_folder = result_dir, seg_item=seg_item)

    return common_config

if __name__ == '__main__':
    evaluation()