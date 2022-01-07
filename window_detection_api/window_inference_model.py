import os
import copy
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader

# define project dependency
from window_detection_api import _init_paths

# project dependence
from common_pytorch.dataset.all_dataset import *
from common_pytorch.config_pytorch import update_config_from_file, get_config_files, \
    update_config_from_params
from common_pytorch.common_loss.balanced_parallel import DataParallelModel
from common_pytorch.net_modules import inferNet

from blocks.resnet_pose import get_default_network_config
from loss.heatmap import get_default_loss_config, get_merge_func

from core.loader import infer_facade_Dataset

from common_pytorch.blocks.resnet_pose import get_default_network_config, get_pose_net


class WindowsInferenceModel:

    def __init__(self, cfg, model):
        s_config = get_config_files(cfg)
        self.config = copy.deepcopy(s_config)
        self.config.network = get_default_network_config()
        self.config.loss = get_default_loss_config()

        self.config = update_config_from_file(self.config, cfg, check_necessity=True)
        self.config = update_config_from_params(self.config, '.', '..')

        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.pytorch.gpus  # a safer method
        devices = [int(i) for i in self.config.pytorch.gpus.split(',')]

        self.merge_hm_flip_func, self.merge_tag_flip_func = get_merge_func(self.config.loss)

        self.batch_size = len(devices) * self.config.dataiter.batch_images_per_ctx

        self.net = get_pose_net(self.config.network, self.config.loss.ae_feat_dim,
                                num_corners if not self.config.loss.useCenterNet else num_corners + 1)

        self.net = DataParallelModel(self.net).cuda()
        ckpt = torch.load(model)  # or other path/to/model
        self.net.load_state_dict(ckpt['network'])

    def infer(self, images_path):
        self.output_path = os.path.join(images_path, "infer_result")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        infer_imdbs = glob.glob(images_path + '/*.jpg')
        infer_imdbs += glob.glob(images_path + '/*.png')
        infer_imdbs.sort()
        dataset_infer = infer_facade_Dataset(infer_imdbs, self.config.train.patch_width, self.config.train.patch_height,
                                             self.config.aug)

        infer_data_loader = DataLoader(dataset=dataset_infer, batch_size=self.batch_size)

        windows_list_with_score = inferNet(infer_data_loader, self.net, self.merge_hm_flip_func,
                                           self.merge_tag_flip_func, flip_pairs,
                                           self.config.train.patch_width, self.config.train.patch_height,
                                           self.config.loss, self.config.test,
                                           self.output_path)

        return self.windows_list_to_json(
            windows_list_with_score, images_path)  # self.windows_list_post_processing(windows_list_with_score)

    def windows_list_post_processing(self, windows_list_with_score):
        ap_pred = []
        for image_index in range(len(windows_list_with_score)):

            # aggreate pred into list
            winPred = windows_list_with_score[image_index]
            for i in range(len(winPred)):
                score = winPred[i]['score']  # confident
                # if score < 0.85:
                #     continue
                temp = np.array(winPred[i]['position'])[:, :2].copy().tolist()

                ap_pred.append(temp)

        return ap_pred

    def windows_list_to_json(self, windows_list_with_score, images_path):
        image_windows_corners_dict_list = []
        infer_imdbs = glob.glob(images_path + '/*.jpg')
        infer_imdbs += glob.glob(images_path + '/*.png')
        for image_index in range(len(windows_list_with_score)):

            image_name = os.path.basename(infer_imdbs[image_index]).split(".")[0]
            windows_corners_list = []

            winPred = windows_list_with_score[image_index]
            for i in range(len(winPred)):
                score = winPred[i]['score']  # confident
                if score < 0.7:
                    continue
                corners = np.array(winPred[i]['position'])[:, :2].copy().tolist()
                uv_dict_list = []
                for corner in corners:
                    uv_dict = {"u": corner[0], "v": corner[1]}
                    uv_dict_list.append(uv_dict)

                windows_corners_list.append({"corners": uv_dict_list})

            windows_json_dict = {"image_name": image_name, "windows": windows_corners_list}
            image_windows_corners_dict_list.append(windows_json_dict)

        images_with_windows_corners_dict = {"images_with_windows": image_windows_corners_dict_list}
        return images_with_windows_corners_dict
