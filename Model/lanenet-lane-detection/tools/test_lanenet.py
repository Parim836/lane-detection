#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
test LaneNet model on single image
"""
import argparse
import os.path as ops
import time
import os
import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

BASE_DIR = Path(__file__).resolve().parent.parent

CFG = parse_config_utils.lanenet_cfg        # load config file
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')
# save_mask = f'D:\Research\Project\lanenet-lane-detection\data\output\save_mask'                # save file
# save_binary = f'D:\Research\Project\lanenet-lane-detection\data\output\save_binary'            # save file
# save_instance = f'D:\Research\Project\lanenet-lane-detection\data\output\save_instance'        # save file

# os.makedirs(save_mask, exist_ok=True)
# os.makedirs(save_binary, exist_ok=True)
# os.makedirs(save_instance, exist_ok=True)


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--road_name", type=str, required=True)
    parser.add_argument("--with_lane_fit", type=args_str2bool, default=True)

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_lanenet(image_path, weights_path, road_name, with_lane_fit=True):
    """
    Run LaneNet on single image or all images in a folder
    """
    binary_dir = BASE_DIR / "output" / road_name
    os.makedirs(binary_dir, exist_ok=True)

    # --- เตรียม list ของไฟล์ ---
    if os.path.isdir(image_path):
        image_list = glob.glob(os.path.join(image_path, "*.jpg"))
        image_list += glob.glob(os.path.join(image_path, "*.png"))
        image_list += glob.glob(os.path.join(image_path, "*.jpeg"))
        show_result = False  # folder → ไม่โชว์ภาพ
    else:
        assert ops.exists(image_path), '{:s} not exist'.format(image_path)
        image_list = [image_path]
        show_result = True   # file เดียว → โชว์ภาพ

    # --- TensorFlow graph ---
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    # Session config
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    # restore weights
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        for single_image_path in image_list:
            LOG.info(f'Start reading image {single_image_path} and preprocessing')
            image = cv2.imread(single_image_path, cv2.IMREAD_COLOR)

            if image is None:
                LOG.warning(f"Skipping {single_image_path}, cannot read as image")
                continue

            image_vis = image
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0

            # inference
            t_start = time.time()
            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )
            t_cost = time.time() - t_start
            LOG.info(f'Single image inference cost time: {t_cost:.5f}s')

            # postprocess
            postprocess_result = postprocessor.postprocess(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis,
                with_lane_fit=with_lane_fit,
                data_source='tusimple'
            )
            mask_image = postprocess_result['mask_image']
            if with_lane_fit:
                lane_params = postprocess_result['fit_params']
                LOG.info(f'Model have fitted {len(lane_params)} lanes')
                for i, params in enumerate(lane_params):
                    LOG.info(f'Fitted 2-order lane {i+1} curve param: {params}')

            # normalize embedding
            for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
                instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
            embedding_image = np.array(instance_seg_image[0], np.uint8)

            base_name = os.path.splitext(os.path.basename(single_image_path))[0]

            # cv2.imwrite(os.path.join(save_mask, f'{base_name}_mask.jpg'), mask_image)
            # cv2.imwrite(os.path.join(save_binary, f'{base_name}_binary.jpg'), binary_seg_image[0] * 255)
            # cv2.imwrite(os.path.join(save_instance, f'{base_name}_instance.jpg'), embedding_image)

            cv2.imwrite(str(binary_dir / f"{base_name}.jpg"), binary_seg_image[0] * 255)  #Edit

            LOG.info(f"Saved results for {single_image_path}")

            # --- Show images ---
            if show_result:
                fig, axs = plt.subplots(2, 2, figsize=(12, 6))

                axs[0, 0].imshow(image_vis[:, :, (2, 1, 0)])
                axs[0, 0].set_title('Source Image')
                axs[0, 0].axis('off')

                axs[0, 1].imshow(mask_image[:, :, (2, 1, 0)])
                axs[0, 1].set_title('Mask Image')
                axs[0, 1].axis('off')

                axs[1, 0].imshow(embedding_image[:, :, (2, 1, 0)])
                axs[1, 0].set_title('Instance Embedding')
                axs[1, 0].axis('off')

                axs[1, 1].imshow(binary_seg_image[0] * 255, cmap='gray')
                axs[1, 1].set_title('Binary Segmentation')
                axs[1, 1].axis('off')

                plt.tight_layout()
                plt.show()

    sess.close()



if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    test_lanenet(args.image_path, args.weights_path, args.road_name, with_lane_fit=args.with_lane_fit)
