import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel
from einops import rearrange
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
from dadaptation import DAdaptAdam, DAdaptAdan
from adan_pytorch import Adan
from collections import OrderedDict
import wandb
import pickle as pkl
import gc
from torchinfo import summary
from collections import defaultdict
try:
    from data_utils.datasets import get_data_loader, DSET_NAME_TO_OBJECT
    from models.avit import build_avit
    from utils import logging_utils
    from utils.YParams import YParams
except:
    from .data_utils.datasets import get_data_loader, DSET_NAME_TO_OBJECT
    from .models.avit import build_avit
    from .utils import logging_utils
    from .utils.YParams import YParams


def add_weight_decay(model, weight_decay=1e-5, inner_lr=1e-3, skip_list=()):
    """ From Ross Wightman at:
        https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3 
        
        Goes through the parameter list and if the squeeze dim is 1 or 0 (usually means bias or scale) 
        then don't apply weight decay. 
        """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (len(param.squeeze().shape) <= 1 or name in skip_list):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
            {'params': no_decay, 'weight_decay': 0.,},
            {'params': decay, 'weight_decay': weight_decay}]

class Trainer:
    def __init__(self, params, global_rank, local_rank, device, sweep_id=None):
        self.device = device
        self.params = params
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.sweep_id = sweep_id
        self.log_to_screen = params.log_to_screen
        # Basic setup
        self.train_loss = nn.MSELoss()
        self.startEpoch = 0
        self.epoch = 0
        self.mp_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.half

        self.iters = 0
        self.initialize_data(self.params)
        print(f"Initializing model on rank {self.global_rank}")
        self.initialize_model(self.params)
        self.initialize_optimizer(self.params)
        if params.resuming:
            print("Loading checkpoint %s"%params.checkpoint_path)
            self.restore_checkpoint(params.checkpoint_path)
        if params.resuming == False and params.pretrained:
            print("Starting from pretrained model at %s"%params.pretrained_ckpt_path)
            self.restore_checkpoint(params.pretrained_ckpt_path)
            self.iters = 0
            self.startEpoch = 0
        # Do scheduler after checking for resume so we don't warmup every time
        self.initialize_scheduler(self.params)

    def single_print(self, *text):
        if self.global_rank == 0 and self.log_to_screen:
            print(' '.join([str(t) for t in text]))

    def initialize_data(self, params):
        if params.tie_batches:
            in_rank = 0
        else:
            in_rank = self.global_rank
        if self.log_to_screen:
            print(f"Initializing data on rank {self.global_rank}")
        self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(params, params.train_data_paths, 
                          dist.is_initialized(), split='train', rank=in_rank, train_offset=self.params.embedding_offset)
        self.valid_data_loader, self.valid_dataset, _ = get_data_loader(params, params.valid_data_paths,
                                                                                 dist.is_initialized(),
                                                                        split='val', rank=in_rank)
        if dist.is_initialized():
            self.train_sampler.set_epoch(0)


    def initialize_model(self, params):
        if self.params.model_type == 'avit':
            self.model = build_avit(params).to(device)
        
        if self.params.compile:
            print('WARNING: BFLOAT NOT SUPPORTED IN SOME COMPILE OPS SO SWITCHING TO FLOAT16')
            self.mp_type = torch.half
            self.model = torch.compile(self.model)
        
        if dist.is_initialized():
            self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                 output_device=[self.local_rank], find_unused_parameters=True)
        
        self.single_print(f'Model parameter count: {sum([p.numel() for p in self.model.parameters()])}')

    def initialize_optimizer(self, params): 
        parameters = add_weight_decay(self.model, self.params.weight_decay) # Dont use weight decay on bias/scaling terms
        if params.optimizer == 'adam':
            if self.params.learning_rate < 0:
                self.optimizer =  DAdaptAdam(parameters, lr=1., growth_rate=1.05, log_every=100, decouple=True )
            else:
                self.optimizer = optim.AdamW(parameters, lr=params.learning_rate)
        elif params.optimizer == 'adan':
            if self.params.learning_rate < 0:
                self.optimizer =  DAdaptAdan(parameters, lr=1., growth_rate=1.05, log_every=100)
            else:
                self.optimizer = Adan(parameters, lr=params.learning_rate)
        elif params.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=params.learning_rate, momentum=0.9)
        else: 
            raise ValueError(f"Optimizer {params.optimizer} not supported")
        self.gscaler = amp.GradScaler(enabled= (self.mp_type == torch.half and params.enable_amp))

    def initialize_scheduler(self, params):
        if params.scheduler_epochs > 0:
            sched_epochs = params.scheduler_epochs
        else:
            sched_epochs = params.max_epochs
        if params.scheduler == 'cosine':
            if self.params.learning_rate < 0:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                                            last_epoch = (self.startEpoch*params.epoch_size) - 1,
                                                                            T_max=sched_epochs*params.epoch_size, 
                                                                            eta_min=params.learning_rate / 100)
            else:
                k = params.warmup_steps
                if (self.startEpoch*params.epoch_size) < k:
                    warmup = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=.01, end_factor=1.0, total_iters=k)
                    decay = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, eta_min=params.learning_rate / 100, T_max=sched_epochs)
                    self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, [warmup, decay], [k], last_epoch=(params.epoch_size*self.startEpoch)-1)
        else:
            self.scheduler = None


    def save_checkpoint(self, checkpoint_path, model=None):
        """ Save model and optimizer to checkpoint """
        if not model:
            model = self.model

        torch.save({'iters': self.epoch*self.params.epoch_size, 'epoch': self.epoch, 'model_state': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, checkpoint_path)

    def restore_checkpoint(self, checkpoint_path):
        """ Load model/opt from path """
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.local_rank))
        if 'model_state' in checkpoint:
            model_state = checkpoint['model_state']
        else:
            model_state = checkpoint
        try: # Try to load with DDP Wrapper
            self.model.load_state_dict(model_state)
        except: # If that fails, either try to load into module or strip DDP prefix
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(model_state)
            else:
                new_state_dict = OrderedDict()
                for key, val in model_state.items():
                    # Failing means this came from DDP - strip the DDP prefix
                    name = key[7:]
                    new_state_dict[name] = val
                self.model.load_state_dict(new_state_dict)
        
        if self.params.resuming:  #restore checkpoint is used for finetuning as well as resuming. If finetuning (i.e., not resuming), restore checkpoint does not load optimizer state, instead uses config specified lr.
            self.iters = checkpoint['iters']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.startEpoch = checkpoint['epoch']
            self.epoch = self.startEpoch
        else:
            self.iters = 0
        if self.params.pretrained:
            if self.params.freeze_middle:
                self.model.module.freeze_middle()
            elif self.params.freeze_processor:
                self.model.module.freeze_processor()
            else:
                self.model.module.unfreeze()
            # See how much we need to expand the projections
            exp_proj = 0
            # Iterate through the appended datasets and add on enough embeddings for all of them. 
            for add_on in self.params.append_datasets:
                exp_proj += len(DSET_NAME_TO_OBJECT[add_on]._specifics()[2])
            self.model.module.expand_projections(exp_proj)
        checkpoint = None
        self.model = self.model.to(self.device)

    def train_one_epoch(self):
        self.model.train()
        self.epoch += 1
        tr_time = 0
        data_time = 0
        data_start = time.time()
        self.model.train()
        logs = {'train_rmse': torch.zeros(1).to(self.device),
                'train_nrmse': torch.zeros(1).to(self.device),
            'train_l1': torch.zeros(1).to(self.device)}
        steps = 0
        last_grads = [torch.zeros_like(p) for p in self.model.parameters()]
        grad_logs = defaultdict(lambda: torch.zeros(1, device=self.device))
        grad_counts = defaultdict(lambda: torch.zeros(1, device=self.device))
        loss_logs = defaultdict(lambda: torch.zeros(1, device=self.device))
        loss_counts = defaultdict(lambda: torch.zeros(1, device=self.device))
        self.single_print('train_loader_size', len(self.train_data_loader), len(self.train_dataset))
        for batch_idx, data in enumerate(self.train_data_loader):
            steps += 1
            inp, file_index, field_labels, bcs, tar = map(lambda x: x.to(self.device), data) 
            dset_type = self.train_dataset.sub_dsets[file_index[0]].type
            loss_counts[dset_type] += 1
            inp = rearrange(inp, 'b t c h w -> t b c h w')
            data_time += time.time() - data_start
            dtime = time.time() - data_start
            
            self.model.require_backward_grad_sync = ((1+batch_idx) % self.params.accum_grad == 0)
            with amp.autocast(self.params.enable_amp, dtype=self.mp_type):
                model_start = time.time()
                output = self.model(inp, field_labels, bcs)
                spatial_dims = tuple(range(output.ndim))[2:] # Assume 0, 1, 2 are T, B, C
                residuals = output - tar
                # Differentiate between log and accumulation losses
                tar_norm = (1e-7 + tar.pow(2).mean(spatial_dims, keepdim=True))
                raw_loss = ((residuals).pow(2).mean(spatial_dims, keepdim=True) 
                         / tar_norm)
                # Scale loss for accum
                loss = raw_loss.mean() / self.params.accum_grad
                forward_end = time.time()
                forward_time = forward_end-model_start
                # Logging
                with torch.no_grad():
                    logs['train_l1'] += F.l1_loss(output, tar)
                    log_nrmse = raw_loss.sqrt().mean()
                    logs['train_nrmse'] += log_nrmse # ehh, not true nmse, but close enough
                    loss_logs[dset_type] += loss.item()
                    logs['train_rmse'] += residuals.pow(2).mean(spatial_dims).sqrt().mean()
                # Scaler is no op when not using AMP
                self.gscaler.scale(loss).backward()
                backward_end = time.time()
                backward_time = backward_end - forward_end
                # Only take step once per accumulation cycle
                optimizer_step = 0
                if self.model.require_backward_grad_sync:
                    self.gscaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    self.gscaler.step(self.optimizer)
                    self.gscaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None:
                        self.scheduler.step()
                    optimizer_step = time.time() - backward_end
                tr_time += time.time() - model_start
                if self.log_to_screen and batch_idx % self.params.log_interval == 0 and self.global_rank == 0:
                    print(f"Epoch {self.epoch} Batch {batch_idx} Train Loss {log_nrmse.item()}")
                if self.log_to_screen:
                    print('Total Times. Batch: {}, Rank: {}, Data Shape: {}, Data time: {}, Forward: {}, Backward: {}, Optimizer: {}'.format(
                        batch_idx, self.global_rank, inp.shape, dtime, forward_time, backward_time, optimizer_step))
                data_start = time.time()
        logs = {k: v/steps for k, v in logs.items()}
        # If distributed, do lots of logging things
        if dist.is_initialized():
            for key in sorted(logs.keys()):
                dist.all_reduce(logs[key].detach()) 
                logs[key] = float(logs[key]/dist.get_world_size())
            for key in sorted(loss_logs.keys()):
                dist.all_reduce(loss_logs[key].detach())
            for key in sorted(grad_logs.keys()):
                dist.all_reduce(grad_logs[key].detach())
            for key in sorted(loss_counts.keys()):
                dist.all_reduce(loss_counts[key].detach())
            for key in sorted(grad_counts.keys()):
                dist.all_reduce(grad_counts[key].detach())
            
        for key in loss_logs.keys():
            logs[f'{key}/train_nrmse'] = loss_logs[key] / loss_counts[key]

        self.iters += steps
        if self.global_rank == 0:
            logs['iters'] = self.iters
        self.single_print('all reduces executed!')

        return tr_time, data_time, logs

    def validate_one_epoch(self, full=False):
        """
        Validates - for each batch just use a small subset to make it easier.

        Note: need to split datasets for meaningful metrics, but TBD. 
        """
        # Don't bother with full validation set between epochs
        self.model.eval()
        if full:
            cutoff = 999999999999
        else:
            cutoff = 40
        self.single_print('STARTING VALIDATION!!!')
        with torch.inference_mode():
            # There's something weird going on when i turn this off.
            with amp.autocast(False, dtype=self.mp_type):
                field_labels = self.valid_dataset.get_state_names()
                distinct_dsets = list(set([dset.title for dset_group in self.valid_dataset.sub_dsets 
                                           for dset in dset_group.get_per_file_dsets()]))
                counts = {dset: 0 for dset in distinct_dsets}
                logs = {} # 
                # Iterate through all folder specific datasets
                for subset_group in self.valid_dataset.sub_dsets:
                    for subset in subset_group.get_per_file_dsets():
                        dset_type = subset.title
                        self.single_print('VALIDATING ON', dset_type)
                        # Create data loader for each
                        if self.params.use_ddp:
                            temp_loader = torch.utils.data.DataLoader(subset, batch_size=self.params.batch_size,
                                                                    num_workers=self.params.num_data_workers,
                                                                    sampler=torch.utils.data.distributed.DistributedSampler(subset,
                                                                                                                            drop_last=True)
                                    )
                        else:
                            # Seed isn't important, just trying to mix up samples from different trajectories
                            temp_loader = torch.utils.data.DataLoader(subset, batch_size=self.params.batch_size,
                                                                    num_workers=self.params.num_data_workers, 
                                                                    shuffle=True, generator= torch.Generator().manual_seed(0),
                                                                    drop_last=True)
                        count = 0
                        for batch_idx, data in enumerate(temp_loader):
                            # Only do a few batches of each dataset if not doing full validation
                            if count > cutoff:
                                del(temp_loader)
                                break
                            count += 1
                            counts[dset_type] += 1
                            inp, bcs, tar = map(lambda x: x.to(self.device), data) 
                            # Labels come from the trainset - useful to configure an extra field for validation sets not included
                            labels = torch.tensor(self.train_dataset.subset_dict.get(subset.get_name(), [-1]*len(self.valid_dataset.subset_dict[subset.get_name()])),
                                                device=self.device).unsqueeze(0).expand(tar.shape[0], -1)
                            inp = rearrange(inp, 'b t c h w -> t b c h w')
                            output = self.model(inp, labels, bcs)
                            # I don't think this is the true metric, but PDE bench averages spatial RMSE over batches (MRMSE?) rather than root after mean
                            # And we want the comparison to be consistent
                            spatial_dims = tuple(range(output.ndim))[2:] # Assume 0, 1, 2 are T, B, C
                            residuals = output - tar
                            nmse = (residuals.pow(2).mean(spatial_dims, keepdim=True) 
                                    / (1e-7 + tar.pow(2).mean(spatial_dims, keepdim=True))).sqrt()#.mean()
                            logs[f'{dset_type}/valid_nrmse'] = logs.get(f'{dset_type}/valid_nrmse',0) + nmse.mean()
                            logs[f'{dset_type}/valid_rmse'] = (logs.get(f'{dset_type}/valid_mse',0) 
                                                                + residuals.pow(2).mean(spatial_dims).sqrt().mean())
                            logs[f'{dset_type}/valid_l1'] = (logs.get(f'{dset_type}/valid_l1', 0) 
                                                                + residuals.abs().mean())

                            for i, field in enumerate(self.valid_dataset.subset_dict[subset.type]):
                                field_name = field_labels[field]
                                logs[f'{dset_type}/{field_name}_valid_nrmse'] = (logs.get(f'{dset_type}/{field_name}_valid_nrmse', 0) 
                                                                                + nmse[:, i].mean())
                                logs[f'{dset_type}/{field_name}_valid_rmse'] = (logs.get(f'{dset_type}/{field_name}_valid_rmse', 0) 
                                                                                    + residuals[:, i:i+1].pow(2).mean(spatial_dims).sqrt().mean())
                                logs[f'{dset_type}/{field_name}_valid_l1'] = (logs.get(f'{dset_type}/{field_name}_valid_l1', 0) 
                                                                            +  residuals[:, i].abs().mean())
                        else:
                            del(temp_loader)

            self.single_print('DONE VALIDATING - NOW SYNCING')
            for k, v in logs.items():
                dset_type = k.split('/')[0]
                logs[k] = v/counts[dset_type]

            logs['valid_nrmse'] = 0
            for dset_type in distinct_dsets:
                logs['valid_nrmse'] += logs[f'{dset_type}/valid_nrmse']/len(distinct_dsets)
            
            if dist.is_initialized():
                for key in sorted(logs.keys()):
                    dist.all_reduce(logs[key].detach()) # There was a bug with means when I implemented this - dont know if fixed
                    logs[key] = float(logs[key].item()/dist.get_world_size())
                    if 'rmse' in key:
                        logs[key] = logs[key]
            self.single_print('DONE SYNCING - NOW LOGGING')
        return logs               


    def train(self):
        # This is set up this way based on old code to allow wandb sweeps
        if self.params.log_to_wandb:
            if self.sweep_id:
                wandb.init(dir=self.params.experiment_dir)
                hpo_config = wandb.config.as_dict()
                self.params.update_params(hpo_config)
                params = self.params
            else:
                wandb.init(dir=self.params.experiment_dir, config=self.params, name=self.params.name, group=self.params.group, 
                           project=self.params.project, entity=self.params.entity, resume=True)
                
        if self.sweep_id and dist.is_initialized():
            param_file = f"temp_hpo_config_{os.environ['SLURM_JOBID']}.pkl"
            if self.global_rank == 0:
                with open(param_file, 'wb') as f:
                    pkl.dump(hpo_config, f)
            dist.barrier() # Stop until the configs are written by hacky MPI sub
            if self.global_rank != 0: 
                with open(param_file, 'rb') as f:
                    hpo_config = pkl.load(f)
            dist.barrier() # Stop until the configs are written by hacky MPI sub
            if self.global_rank == 0:
                os.remove(param_file)
            # If tuning batch size, need to go from global to local batch size
            if 'batch_size' in hpo_config:
                hpo_config['batch_size'] = int(hpo_config['batch_size'] // self.world_size)
            self.params.update_params(hpo_config)
            params = self.params
            self.initialize_data(self.params) # This is the annoying redundant part - but the HPs need to be set from wandb
            self.initialize_model(self.params) 
            self.initialize_optimizer(self.params)
            self.initialize_scheduler(self.params)
        if self.global_rank == 0:
            summary(self.model)
        if self.params.log_to_wandb:
            wandb.watch(self.model)
        self.single_print("Starting Training Loop...")
        # Actually train now, saving checkpoints, logging time, and logging to wandb
        best_valid_loss = 1.e6
        for epoch in range(self.startEpoch, self.params.max_epochs):
            if dist.is_initialized():
                self.train_sampler.set_epoch(epoch)
            start = time.time()

            # with torch.autograd.detect_anomaly(check_nan=True):
            tr_time, data_time, train_logs = self.train_one_epoch()
            
            valid_start = time.time()
            # Only do full validation set on last epoch - don't waste time
            if epoch==self.params.max_epochs-1:
                valid_logs = self.validate_one_epoch(True)
            else:
                valid_logs = self.validate_one_epoch()
            
            post_start = time.time()
            train_logs.update(valid_logs)
            train_logs['time/train_time'] = valid_start-start
            train_logs['time/train_data_time'] = data_time
            train_logs['time/train_compute_time'] = tr_time
            train_logs['time/valid_time'] = post_start-valid_start
            if self.params.log_to_wandb:
                wandb.log(train_logs) 
            gc.collect()
            torch.cuda.empty_cache()

            if self.global_rank == 0:
                if self.params.save_checkpoint:
                    self.save_checkpoint(self.params.checkpoint_path)
                if epoch % self.params.checkpoint_save_interval == 0:
                    self.save_checkpoint(self.params.checkpoint_path + f'_epoch{epoch}')
                if valid_logs['valid_nrmse'] <= best_valid_loss:
                    self.save_checkpoint(self.params.best_checkpoint_path)
                    best_valid_loss = valid_logs['valid_nrmse']
                
                cur_time = time.time()
                self.single_print(f'Time for train {valid_start-start}. For valid: {post_start-valid_start}. For postprocessing:{cur_time-post_start}')
                self.single_print('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
                self.single_print('Train loss: {}. Valid loss: {}'.format(train_logs['train_nrmse'], valid_logs['valid_nrmse']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default='00', type=str)
    parser.add_argument("--use_ddp", action='store_true', help='Use distributed data parallel')
    parser.add_argument("--yaml_config", default='./config/multi_ds.yaml', type=str)
    parser.add_argument("--config", default='basic_config', type=str)
    parser.add_argument("--sweep_id", default=None, type=str, help='sweep config from ./configs/sweeps.yaml')
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params.use_ddp = args.use_ddp
    # Set up distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if args.use_ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank) # Torch docs recommend just using device, but I had weird memory issues without setting this.
    device = torch.device(local_rank) if torch.cuda.is_available() else torch.device("cpu")

    # Modify params
    params['batch_size'] = int(params.batch_size//world_size)
    params['startEpoch'] = 0
    if args.sweep_id:
        jid = os.environ['SLURM_JOBID'] # so different sweeps dont resume
        expDir = os.path.join(params.exp_dir, args.sweep_id, args.config, str(args.run_name), jid)
    else:
        expDir = os.path.join(params.exp_dir, args.config, str(args.run_name))

    params['old_exp_dir'] = expDir # I dont remember what this was for but not removing it yet
    params['experiment_dir'] = os.path.abspath(expDir)
    params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')
    params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
    params['old_checkpoint_path'] = os.path.join(params.old_exp_dir, 'training_checkpoints/best_ckpt.tar')
    
    # Have rank 0 check for and/or make directory
    if  global_rank==0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
            os.makedirs(os.path.join(expDir, 'training_checkpoints/'))
    params['resuming'] = True if os.path.isfile(params.checkpoint_path) else False

    # WANDB things
    params['name'] =  str(args.run_name)
    # params['group'] = params['group'] #+ args.config
    # params['project'] = "pde_bench"
    # params['entity'] = "flatiron-scipt"
    if global_rank==0:
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
        logging_utils.log_versions()
        params.log()

    if global_rank==0:
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
        logging_utils.log_versions()
        params.log()

    params['log_to_wandb'] = (global_rank==0) and params['log_to_wandb']
    params['log_to_screen'] = (global_rank==0) and params['log_to_screen']
    torch.backends.cudnn.benchmark = False

    if global_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
            yaml.dump(hparams,  hpfile )
    trainer = Trainer(params, global_rank, local_rank, device, sweep_id=args.sweep_id)
    if args.sweep_id and trainer.global_rank==0:
        print(args.sweep_id, trainer.params.entity, trainer.params.project)
        wandb.agent(args.sweep_id, function=trainer.train, count=1, entity=trainer.params.entity, project=trainer.params.project) 
    else:
        trainer.train()
    if params.log_to_screen:
        print('DONE ---- rank %d'%global_rank)
