import argparse

IMG_RESIZE = 256
USING_L1_LOSS = True
EPOCHS = 101
LEARNING_RATE = 5e-5
RELOAD_CHECKPOINT = True

PATH_TO_PTH_CHECKPOINT = "pretrained_models/LSMIU_galaxy"
DATA_PATH = 'LSMI_data/galaxy_512/'

BS = 2


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--l1_loss', type=bool, default=USING_L1_LOSS)

    # model hyer-parameters
    parser.add_argument('--use_pretrain', type=bool, default=RELOAD_CHECKPOINT)
    parser.add_argument('--model_path', type=str, default=PATH_TO_PTH_CHECKPOINT if RELOAD_CHECKPOINT else None)
    parser.add_argument('--model_type', type=str, default='U_Net')
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=2)
    # training hyper-parameters
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE * (BS / 32))
    parser.add_argument('--num_epochs_decay', type=int, default=100)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--seed', type=int, default=666)

    # dataset & loader config
    parser.add_argument('--trdir', type=str, default=DATA_PATH)
    parser.add_argument('--camera', type=str, default='galaxy')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--image_pool', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--input_type', type=str, default='uvl', choices=['rgb', 'uvl'])
    parser.add_argument('--output_type', type=str, default='uv', choices=['illumination', 'uv', 'mixmap'])
    parser.add_argument('--uncalculable', type=int, default=None)
    parser.add_argument('--mask_black', type=int, default=None)
    parser.add_argument('--mask_highlight', type=int, default=None)
    parser.add_argument('--mask_uncalculable', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=15)

    # data augmentation config
    parser.add_argument('--random_crop', type=str, default='yes', choices=['yes', 'no'])
    parser.add_argument('--illum_augmentation', type=str, default='yes', choices=['yes', 'no'])
    parser.add_argument('--sat_min', type=float, default=0.2)
    parser.add_argument('--sat_max', type=float, default=0.8)
    parser.add_argument('--val_min', type=float, default=1.0)
    parser.add_argument('--val_max', type=float, default=1.0)
    parser.add_argument('--hue_threshold', type=float, default=0.2)

    # path config
    parser.add_argument('--model_root', type=str, default='models')
    parser.add_argument('--result_root', type=str, default='results')
    parser.add_argument('--log_root', type=str, default='logs')

    # Misc
    parser.add_argument('--save_epoch', type=int, default=20,
                        help='number of epoch for auto saving, -1 for turn off')
    parser.add_argument('--multi_gpu', type=int, default=1, choices=[0, 1],
                        help='0 for single-GPU, 1 for multi-GPU')
    parser.add_argument('--save_result', type=str, default='no')
    parser.add_argument('--val_freq', type=int, default=2)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--change_log', type=str)
    config = parser.parse_args()

    return config
