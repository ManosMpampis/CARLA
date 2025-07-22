import os
import logging
import time
import errno

import torch
from termcolor import colored


def mkdir(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger=None):
        self.logger = EmptyLogger() if logger is None else logger
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.log(('\t'.join(entries)))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


@torch.no_grad()
def fill_ts_repository(p, loader, model, ts_repository, real_aug=False, ts_repository_aug=None, logger=None):
    """_summary_

    Args:
        p (dict): configuration dictionary
        loader (torch.utils.data.DataLoader): Dataset loader to be used to generate near and far neighbors
        model (torch.nn.Module): Trained model to use for filling time serires repository
        ts_repository (TSRepository): Time series repository to be filled.
        real_aug (bool, optional): Determines if the method save the new values to the Time Series Repositorys. Defaults to False.
        ts_repository_aug (TSRepository, optional): Time series repository filled with original anchors and negative neighbors. Defaults to None.
        device (torch.device, optional): Device to use for torch. Defaults to torch.device("cpu").
        verbose_dict (dict): Verbose variables for logging.
    """
    logger = EmptyLogger() if logger is None else logger
    
    model.eval()
    device = next(model.parameters()).device

    ts_repository.reset()
    if ts_repository_aug != None: ts_repository_aug.reset()
    if real_aug: ts_repository.resize(3)

    con_data = torch.tensor([], device=device)
    con_target = torch.tensor([], device=device)
    for i, batch in enumerate(loader): 
        ts_org = batch['ts_org'].to(device, non_blocking=True)
        targets = batch['target'].to(device, non_blocking=True)
        if ts_org.ndim == 3:
            b, w, h = ts_org.shape
        else:
            b, w = ts_org.shape
            h = 1

        output = model(ts_org.reshape(b, h, w))
        ts_repository.update(output, targets)
        if ts_repository_aug != None: ts_repository_aug.update(output, targets)
        if i % 100 == 0:
            logger.log(f'Fill TS Repository [{i}/{len(loader)}]')

        if real_aug:
            con_data = torch.cat((con_data, ts_org), dim=0)
            con_target = torch.cat((con_target, targets), dim=0)

            ts_w_augment = batch['ts_w_augment'].to(device, non_blocking=True)
            targets = torch.tensor([1]*ts_w_augment.shape[0], dtype=torch.long, device=device)

            output = model(ts_w_augment.reshape(b, h, w))
            ts_repository.update(output, targets)
            # ts_repository_aug.update(output, targets)


            ts_ss_augment = batch['ts_ss_augment'].to(device, non_blocking=True) #cuda
            targets = torch.tensor([4]*ts_ss_augment.shape[0], dtype=torch.long, device=device)
            
            con_data = torch.cat((con_data, ts_ss_augment), dim=0)
            con_target = torch.cat((con_target, targets), dim=0)
            output = model(ts_ss_augment.reshape(b, h, w))
            ts_repository.update(output, targets)
            ts_repository_aug.update(output, targets)


    if real_aug:
        from data.ra_dataset import SaveAugmentedDataset
        con_dataset = SaveAugmentedDataset(con_data.cpu(), con_target.cpu())
        con_loader = torch.utils.data.DataLoader(con_dataset, num_workers=p['num_workers'],
                                                 batch_size=p['batch_size'], pin_memory=True,
                                                 drop_last=False, shuffle=False)
        torch.save(con_loader, p['contrastive_dataset'])

def log(string, verbose=1, file_path=None, color=None):
    if verbose >= 1:
        if file_path is not None and verbose >= 2:
            with open(file_path, 'a') as f:
                f.write(string + '\n')
                f.flush()
        if color is not None:
            string = colored(string, color)
        print(string)
    return

class Logger:
    def __init__(self, verbose=1, file_path="./", use_tensorboard=True):
        
        self.verbose = verbose
        self.use_tensorboard = use_tensorboard

        self._name = "Self-Awareness"
        self._version = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.log_dir = os.path.join(file_path, f"-{self._version}")

        self._init_logger()

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version
        
    def _init_logger(self):
        self.logger = logging.getLogger(name=self.name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # create file handler
        if self.verbose == 0:
            h = logging.NullHandler()
        elif self.verbose == 1:
            h = logging.StreamHandler()
        elif self.verbose >= 2:
            mkdir(self.log_dir)
            h = logging.FileHandler(os.path.join(self.log_dir, "logs.txt"))
        h.setLevel(logging.INFO)
        # set file formatter
        fmt = "[%(asctime)s]: %(message)s"
        formatter = logging.Formatter(fmt, datefmt="%m-%d %H:%M:%S")
        h.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(h)

        self.log_metrics = self._do_nothing
        self.scalar_summary = self._do_nothing
        self.finalize = self._do_nothing
        # add tensorboard handler
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    "the dependencies to use torch.utils.tensorboard "
                    "(applicable to PyTorch 1.1 or higher)"
                ) from None
            self.log(
                "Using Tensorboard, logs will be saved in {}".format(self.log_dir)
            )
            self.experiment = SummaryWriter(log_dir=self.log_dir)
            self.log_metrics = self._log_metrics
            self.scalar_summary = self._scalar_summary
            self.finalize = self._finalize


    def info(self, string):
        self.logger.info(string)

    def log(self, string):
        self.logger.info(string)

    def dump_cfg(self, cfg_node):
        with open(os.path.join(self.log_dir, "train_cfg.yml"), "w") as f:
            cfg_node.dump(stream=f)

    def log_hyperparams(self, params):
        self.logger.info(f"hyperparams: {params}")

    def _log_metrics(self, metrics, step):
        self.logger.info(f"Val_metrics: {metrics}")
        for k, v in metrics.items():
            self.experiment.add_scalars("Val_metrics/" + k, {"Val": v}, step)

    def _finalize(self):
        self.experiment.flush()
        self.experiment.close()
        self.save()

    def _scalar_summary(self, tag, phase, value, step):
        self.experiment.add_scalars(tag, {phase: value}, step)

    def _do_nothing(self, *args, **kwargs):
        pass

class EmptyLogger:
    def __init__(self, verbose=None, file_path="./", use_tensorboard=True):
        self._name = "empty"
        self._version = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        self._init_logger()

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version
        
    def _init_logger(self):
        self.logger = logging.getLogger(name=self.name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        h = logging.NullHandler()
        h.setLevel(logging.INFO)
        # set file formatter
        fmt = "[%(asctime)s]: %(message)s"
        formatter = logging.Formatter(fmt, datefmt="%m-%d %H:%M:%S")
        h.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(h)

        self.log_metrics = self._do_nothing
        self.scalar_summary = self._do_nothing
        self.finalize = self._do_nothing
        


    def info(self, string):
        self.logger.info(string)

    def log(self, string):
        self.logger.info(string)

    def dump_cfg(self, cfg_node):
        pass

    def log_hyperparams(self, params):
        self.logger.info(f"hyperparams: {params}")


if __name__ == "__main__":
    logger = Logger(verbose=0, file_path="./", use_tensorboard=False)
    logger.log("test")
    logger.log('test2')