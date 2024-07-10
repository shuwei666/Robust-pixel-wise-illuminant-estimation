import numpy as np
import os
import pprint
import torch
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import (
    set_seed, save_log, LossTracker, Evaluator, criterion_loss,
    EPS, DEVICE, get_MAE, apply_wb, print_metrics, criterion_l1_loss
)
from dataset import LSMI, RandomColor, aug_crop, val_resize
from settings import get_args
from model import U_Net

try:
    from torch.utils.tensorboard import SummaryWriter
    USE_TB = True
except ImportError:
    USE_TB = False


def train_net(net,
              epochs,
              batch_size,
              lr,
              validation_frequency,
              data_dir,
              stopping_patience,
              logdir,
              l1_loss):
    train_loader, val_loader = get_data_loaders(data_dir=data_dir,
                                                batch_size=batch_size)

    if USE_TB:
        writer = SummaryWriter(comment=f"_LR_{lr}_BS_{batch_size}",
                               log_dir=logdir)

    logging.info(f'''Start training:

            Epochs:                 {epochs} epochs
            Batch size:             {batch_size}
            Learning rate:          {lr}
            Validation freq:        {validation_frequency}
            Device:                 {DEVICE}
            Tensorboard:            {USE_TB}
            Train dir:              {data_dir}
            Log dir:                {logdir}

        ''')

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))

    # Select the loss function
    criterion = criterion_l1_loss() if l1_loss else criterion_loss()
    if l1_loss:
        logging.info(f"\t Using l1 loss for fine-tuning!\t")

    evaluator = Evaluator()
    best_val_error, best_metrics = 100.0, evaluator.get_best_metrics()
    train_loss = LossTracker()

    epochs_without_improvement = 0

    for epoch in range(epochs):
        net.train()
        train_loss.reset()

        for i, batch in enumerate(train_loader):
            inputs, gd_uv = batch['input_uvl'], batch['gt_uv']
            inputs, gd_uv = inputs.to(DEVICE), gd_uv.to(DEVICE)

            pred_uv = net(inputs)
            optimizer.zero_grad()

            loss = criterion(pred_uv, gd_uv)

            loss.backward()
            optimizer.step()

            input_rgb = batch['input_rgb'].to(DEVICE)
            pred_rgb = apply_wb(input_rgb, pred_uv, pred_type='uv')
            pred_illu = input_rgb / (pred_rgb + EPS)
            pred_illu[:, 1, :, :] = 1.

            gt_illu = batch['gt_illu'].to(DEVICE)
            ones = torch.ones_like(gt_illu[:, :1, :, :])
            gt_illu = torch.cat([gt_illu[:, :1, :, :], ones, gt_illu[:, 1:, :, :]], dim=1)

            mae_illu = get_MAE(pred_illu, gt_illu)

            train_loss.update(mae_illu.item())
            logging.info(f"[ Epoch:{epoch + 1}/{epochs} - Batch: {i + 1}/{len(train_loader)} ]|"
                         f"[Training loss {loss.item(): .4f} | Angular error: {mae_illu.item():.4f}]")

        if epoch % validation_frequency == 0:
            evaluator.reset_errors()
            logging.info('Start Validation...\n')

            val_loss = val_net(net, val_loader, evaluator)

            if USE_TB:
                writer.add_scalars("Loss",
                                   {'train': train_loss.avg,
                                    'eval': val_loss.avg}, epoch)

                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

            logging.info(f"\nThe current val error is {val_loss.avg:.2f}(Best is {best_val_error})\n")
            metrics = evaluator.compute_metrics()
            logging.info(f"The current metrics are: \n")
            print_metrics(metrics, best_metrics)

            if val_loss.avg < best_val_error:
                epochs_without_improvement = 0
                best_val_error = val_loss.avg
                best_metrics = evaluator.update_best_metrics()
                logging.info(f"Saving the best model...\n")
                torch.save(net.state_dict(), logdir + '/model.pth')

            else:
                epochs_without_improvement += 1

            logging.info(f"Epoch: [{epoch + 1}/{epochs}]: \n"
                         f"Train-loss:{train_loss.avg: .2f}\n"
                         f"Val-loss: {val_loss.avg: .2f}[Best:{best_val_error:.2f}]\n")

            if validation_frequency * epochs_without_improvement >= stopping_patience:
                logging.info(f"Training stopped! Best model has been saved!")
                logging.info(f"\n Best metrics:\n***************"
                             f"\n {pprint.pformat(best_metrics)}"
                             f"\n****************************\n")
                break

        if (epoch + 1) > (epochs * 9 // 10):
            lr -= (lr / 10)
            for param in optimizer.param_groups:
                param['lr'] = lr
            logging.info(f'Decay lr to {lr}')

    if USE_TB:
        writer.close()

    return best_metrics


def val_net(net, val_loader, evaluator):
    net.eval()
    val_loss = LossTracker()
    with tqdm(total=len(val_loader), desc="Validation Processing",
              bar_format='{l_bar}{bar:40}{r_bar}', colour='#00FF00') as pbar:
        for batch in val_loader:
            inputs, gd_uv = batch['input_uvl'], batch['gt_uv']
            inputs, gd_uv = inputs.to(DEVICE), gd_uv.to(DEVICE)

            with torch.no_grad():
                pred_uv = net(inputs)

                input_rgb = batch['input_rgb'].to(DEVICE)
                pred_rgb = apply_wb(input_rgb, pred_uv.detach(), pred_type='uv')
                pred_illu = input_rgb / (pred_rgb + EPS)
                pred_illu[:, 1, :, :] = 1.

                gt_illu = batch['gt_illu'].to(DEVICE)
                ones = torch.ones_like(gt_illu[:, :1, :, :])
                gt_illu = torch.cat([gt_illu[:, :1, :, :], ones, gt_illu[:, 1:, :, :]], dim=1)

                mae_illu = get_MAE(pred_illu, gt_illu).item()
                evaluator.add_error(mae_illu)
                val_loss.update(np.round(mae_illu, 2))
                pbar.update()
                pbar.set_postfix({"Error": mae_illu})

    net.train()

    return val_loss


def get_data_loaders(data_dir, batch_size):
    training_set = LSMI(root=data_dir,
                        split='train',
                        illum_augmentation=RandomColor(0.2, 0.8, 1.0, 1.0, 0.2),
                        transform=aug_crop())
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=15)

    val_set = LSMI(root=data_dir,
                   split='val',
                   illum_augmentation=None,
                   transform=val_resize())
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=15)
    logging.info(f" Train set /Val set ....... : {len(training_set)}/{len(val_set)}\n")

    return training_loader, val_loader


if __name__ == '__main__':

    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    log_dir = f"debug/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.info("Fine-tuning of LSMIU using L1 loss")

    save_log(log_dir, data_name=args.trdir)
    set_seed(args.seed)
    logging.info(f"Using seed {args.seed} for reproduction!")

    net = U_Net()
    net.to(DEVICE)

    if args.use_pretrain and args.model_path:
        logging.info(f"Using fine-tuning! \n"
                     f"Loading checkpoint from {args.model_path}")
        pretrained_state_dict = torch.load(args.model_path + '/model.pth')
        # pretrained_state_dict = torch.load(args.model_path + '/best.pt')

        new_state_dict = {k.replace('module.', ''):
                              v for k, v in pretrained_state_dict.items()}
        net.load_state_dict(new_state_dict, strict=False)
    else:
        logging.info(f"\nTrain model from scratch!!!\n")

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    logging.info(f'Using {torch.cuda.device_count()} GPUs!\n')

    train_net(net=net,
              epochs=args.epochs,
              batch_size=args.batch_size,
              lr=args.lr,
              validation_frequency=args.val_freq,
              data_dir=args.trdir,
              stopping_patience=args.patience,
              logdir=log_dir,
              l1_loss=args.l1_loss)
