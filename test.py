import argparse
import os.path
import pprint
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import U_Net
from dataset import LSMI, val_resize
from utils import set_seed, Evaluator, save_log, DEVICE, apply_wb, get_MAE, EPS
from post_processing import RobustPhysicalConstrainedPostProcessing

# ------------------This is the key parameters of our strategy----------

USING_POST_PROCESSING = True
if USING_POST_PROCESSING:
    robust_post_processing = RobustPhysicalConstrainedPostProcessing()
# ----------------------------------------------------------------------
MODEL_PATH = f"pretrained_models/ours_sony"
TEST_DATA_PATH = 'LSMI_dataset/sony_512'


def get_args_test():
    parser = argparse.ArgumentParser(description='Testing Processing.')
    parser.add_argument('--data_dir', type=str, default=TEST_DATA_PATH),
    parser.add_argument('--model_path', type=str, default=MODEL_PATH),
    parser.add_argument('--random-seed', type=int, metavar='SEED', dest='seed', default=666,
                        help='Random seed number for reproduction')
    return parser.parse_args()


def get_data_loaders(data_dir):
    data_val = LSMI(root=data_dir,
                    split='test',
                    illum_augmentation=None,
                    transform=val_resize())
    test_loader = DataLoader(data_val, batch_size=1, shuffle=False,
                             num_workers=15, drop_last=False, pin_memory=True)

    logging.info(f'The Test dataset is:{data_dir} and length is: {len(data_val)}')

    return test_loader


def test_net(net, data_dir):
    test_loader = get_data_loaders(data_dir=data_dir)
    logging.info(f"Test samples are: {len(test_loader)}\n")

    all_evaluator = Evaluator()

    net.eval()

    all_evaluator.reset_errors()
    single_light_MAE_illum = []
    dual_light_MAE_illum = []
    triple_light_MAE_illum = []
    dual_and_triple_light_MAE_illum = []

    for batch in test_loader:
        inputs, gd_uv, img_name = batch['input_uvl'], batch['gt_uv'], batch['img_file']
        inputs, gd_uv = inputs.to(DEVICE), gd_uv.to(DEVICE)

        with torch.no_grad():

            pred_uv = net(inputs)

            input_rgb = batch['input_rgb'].to(DEVICE)

            pred_rgb = apply_wb(input_rgb, pred_uv.detach(), pred_type='uv')

            pred_illu = input_rgb / (pred_rgb + EPS)

            pred_illu[:, 1, :, :] = 1.
            pred_illu_numpy = pred_illu.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()

            if USING_POST_PROCESSING:
                new_array = robust_post_processing.process_single_image(pred_illu_numpy)
                pred_illu = torch.from_numpy(new_array).permute(2, 0, 1).unsqueeze(0).cuda()

            gt_illu = batch['gt_illu'].to(DEVICE)
            ones = torch.ones_like(gt_illu[:, :1, :, :])
            gt_illu = torch.cat([gt_illu[:, :1, :, :], ones, gt_illu[:, 1:, :, :]], dim=1)

            mae_illu = np.round(get_MAE(pred_illu, gt_illu).item(), 2)

            all_evaluator.add_error(mae_illu)
            logging.info(f"Image {img_name[0]}: {mae_illu}")
            num_of_lights = img_name[0].split('.')[0].split('_')[-1]
            if len(num_of_lights) == 1:
                single_light_MAE_illum.append(mae_illu)
            elif len(num_of_lights) == 2:
                dual_light_MAE_illum.append(mae_illu)
                dual_and_triple_light_MAE_illum.append(mae_illu)
            elif len(num_of_lights) == 3:
                triple_light_MAE_illum.append(mae_illu)
                dual_and_triple_light_MAE_illum.append(mae_illu)

    logging.info(f"All mean:\n\n{pprint.pformat(all_evaluator.compute_metrics())}\n"
                 f"Mean/Median/Max\n\n"
                 f"Single:\n {np.nanmean(single_light_MAE_illum):.2f}"
                 f" {np.median(single_light_MAE_illum):.2f}"
                 f" {np.max(single_light_MAE_illum):.2f}\n"
                 f"Dual:\n {np.nanmean(dual_light_MAE_illum):.2f}"
                 f" {np.median(dual_light_MAE_illum):.2f}"
                 f" {np.max(dual_light_MAE_illum):.2f}\n"
                 f"Triple:\n {np.nanmean(triple_light_MAE_illum):.2f}"
                 f" {np.median(triple_light_MAE_illum):.2f}"
                 f" {np.max(triple_light_MAE_illum):.2f}\n"
                 f"Dual and Triple:\n {np.nanmean(dual_and_triple_light_MAE_illum):.2f}"
                 f" {np.median(np.median(dual_and_triple_light_MAE_illum)):.2f}"
                 f" {np.max(dual_and_triple_light_MAE_illum):.2f}\n")

    logging.info(f"---------The end---------")


if __name__ == '__main__':

    log_path = 'results'
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info(f"\nTesting of robust pixel-wise AWB\n")
    log_dir = os.path.join(f"{log_path}", f"{MODEL_PATH.split('/')[-1]}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    args = get_args_test()
    set_seed(args.seed)
    save_log(log_dir)

    net = U_Net()
    net.to(DEVICE)

    if args.model_path:
        logging.info(f"Loading checkpoint from {args.model_path}")
        pretrained_state_dict = torch.load(args.model_path + '/model.pth')
        new_state_dict = {k.replace('module.', ''):
                              v for k, v in pretrained_state_dict.items()}
        net.load_state_dict(new_state_dict, strict=False)
    else:
        logging.info(f"model path is missing!")

    if USING_POST_PROCESSING:
        logging.info(f"Using the physical-constrained post processing!\n")

    test_net(net=net,
             data_dir=args.data_dir)
