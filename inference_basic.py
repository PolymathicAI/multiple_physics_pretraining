import argparse
import os
import torch
import torch.distributed as dist
from einops import rearrange
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
import numpy as np
import h5py
import matplotlib.pyplot as plt
import imageio

try:
    from data_utils.datasets import get_data_loader
    from models.avit import build_avit
    from utils.YParams import YParams
except:
    from .data_utils.datasets import get_data_loader
    from .models.avit import build_avit
    from .utils.YParams import YParams

class InferenceRunner:
    def __init__(self, params, global_rank, local_rank, device):
        self.device = device
        self.params = params
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.log_to_screen = params.log_to_screen
        self.initialize_data(self.params)
        self.initialize_model(self.params)
        self.restore_checkpoint(self.params.checkpoint_path)

    def initialize_data(self, params):
        # Load test data
        self.test_data_loader, self.test_dataset, _ = get_data_loader(
            params, params.test_data_paths, dist.is_initialized(), split='test', rank=self.global_rank
        )

    def initialize_model(self, params):
        self.model = build_avit(params).to(self.device)
        self.model.eval()

    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_state = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
        # If keys are prefixed with 'module.', strip them
        if any(k.startswith('module.') for k in model_state.keys()):
            new_state_dict = {}
            for k, v in model_state.items():
                new_k = k.replace('module.', '')
                new_state_dict[new_k] = v
            model_state = new_state_dict
            
        self.model.load_state_dict(model_state)
        self.model = self.model.to(self.device)

    # def run_inference(self, rollout_steps=10, history_length=10):
    #     results = []
    #     gt_results = []
    #     with torch.inference_mode():
    #         for batch_idx, data in enumerate(self.test_data_loader):
    #             print(f"Data length: {len(data)}")
    #             inp, file_index, field_labels, bcs, tar = map(lambda x: x.to(self.device), data)
                
    #             # Use only the first `history_length` frames as initial input
    #             input_seq = inp[:, :history_length]  # (batch, history_length, c, h, w)
    #             rollout_preds = []
    #             for step in range(rollout_steps):
    #                 model_input = rearrange(input_seq, 'b t c h w -> t b c h w')
    #                 pred = self.model(model_input, field_labels, bcs)  # (batch, c, h, w)
    #                 rollout_preds.append(pred.cpu().numpy())
    #                 input_seq = torch.cat([input_seq[:, 1:], pred.unsqueeze(1)], dim=1)
    #             results.append(np.stack(rollout_preds))  # (rollout_steps, batch, c, h, w)
    #             # Save corresponding ground truth frames for comparison
    #             # gt_results.append(tar[:, history_length:history_length+rollout_steps].cpu().numpy())
    #             gt_results.append(tar.cpu().numpy())
    #             if self.log_to_screen:
    #                 print(f"Batch {batch_idx}: Rollout completed, shape {np.stack(rollout_preds).shape}")
    #             break
    #     # Concatenate all batches
    #     predictions = np.concatenate(results, axis=1)  # (rollout_steps, total_batch, c, h, w)
    #     ground_truth = np.concatenate(gt_results, axis=1)  # (rollout_steps, total_batch, c, h, w)
    #     return predictions, ground_truth
    
    def run_inference(self, rollout_steps=10, history_length=10):
        results = []
        gt_results = []
        true_future_results = []
        with torch.inference_mode():
            for batch_idx, data in enumerate(self.test_data_loader):
                print(f"Data length: {len(data)}")
                inp, file_index, field_labels, bcs, tar = map(lambda x: x.to(self.device), data)
                
                # Use only the first `history_length` frames as initial input
                input_seq = inp[:, :history_length]  # (batch, history_length, c, h, w)
                # Save the true future frames from the input sequence
                true_future = inp[:, history_length:history_length+rollout_steps]  # (batch, rollout_steps, c, h, w)
                true_future_results.append(true_future.cpu().numpy())
                
                rollout_preds = []
                for step in range(rollout_steps):
                    model_input = rearrange(input_seq, 'b t c h w -> t b c h w')
                    pred = self.model(model_input, field_labels, bcs)  # (batch, c, h, w)
                    rollout_preds.append(pred.cpu().numpy())
                    input_seq = torch.cat([input_seq[:, 1:], pred.unsqueeze(1)], dim=1)
                results.append(np.stack(rollout_preds))  # (rollout_steps, batch, c, h, w)
                gt_results.append(tar.cpu().numpy())
                if self.log_to_screen:
                    print(f"Batch {batch_idx}: Rollout completed, shape {np.stack(rollout_preds).shape}")
                break
        predictions = np.concatenate(results, axis=1)  # (rollout_steps, total_batch, c, h, w)
        ground_truth = np.concatenate(gt_results, axis=1)  # (batch, c, h, w)
        true_future = np.concatenate(true_future_results, axis=0)  # (total_batch, rollout_steps, c, h, w)
        return predictions, ground_truth, true_future

def save_rollout_video(predictions, ground_truth, output_path, field_idx=0):
    # predictions: (rollout_steps, batch, c, h, w)
    # ground_truth: (batch, c, h, w)
    rollout_steps, batch, c, h, w = predictions.shape
    frames = []
    # Use the first ground truth frame for all steps
    gt_img = ground_truth[0, field_idx, :, :]
    
    for t in range(rollout_steps):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        pred_img = predictions[t, 0, field_idx, :, :]
        axes[0].imshow(gt_img, cmap='viridis')
        axes[0].set_title("Ground Truth (static)")
        axes[0].axis('off')
        axes[1].imshow(pred_img, cmap='viridis')
        axes[1].set_title(f"Prediction step {t}")
        axes[1].axis('off')
        plt.tight_layout()
        fig.canvas.draw()
        # img = np.frombuffer(fig.canvas.get_renderer().tostring_rgb(), dtype=np.uint8)
        # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = np.frombuffer(fig.canvas.get_renderer().tostring_argb(), dtype=np.uint8)
        height, width = fig.canvas.get_width_height()[::-1]
        img = img.reshape(height, width, 4)[:, :, 1:]  # Drop alpha from ARGB
        frames.append(img)
        plt.close(fig)
    # video_path = output_path.replace('.h5', '.mp4')
    imageio.mimsave(output_path, frames, fps=2)
    print(f"Saved rollout video to {output_path}")
    

def save_rollout_vs_true_video(predictions, true_future, output_path, field_idx=0):
    # predictions: (rollout_steps, batch, c, h, w)
    # true_future: (batch, rollout_steps, c, h, w)
    rollout_steps, batch, c, h, w = predictions.shape
    frames = []
    max_steps = min(rollout_steps, true_future.shape[1])
    for t in range(max_steps):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=200)  # Higher dpi for sharper images
        pred_img = predictions[t, 0, field_idx, :, :]
        true_img = true_future[0, t, field_idx, :, :]
        axes[0].imshow(true_img, cmap='viridis')
        axes[0].set_title(f"True Frame {t}")
        axes[0].axis('off')
        axes[1].imshow(pred_img, cmap='viridis')
        axes[1].set_title(f"Prediction step {t}")
        axes[1].axis('off')
        plt.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.get_renderer().tostring_argb(), dtype=np.uint8)
        height, width = fig.canvas.get_width_height()[::-1]
        img = img.reshape(height, width, 4)[:, :, 1:]  # Drop alpha from ARGB
        frames.append(img)
        plt.close(fig)
    imageio.mimsave(output_path, frames, fps=2)
    print(f"Saved rollout vs true video to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./config/mpp_avit_b_config.yaml', type=str)
    parser.add_argument("--config", default='basic_config', type=str)
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params.use_ddp = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up checkpoint path (update as needed)
    expDir = os.path.join(params.exp_dir, args.config, "demo_multi_gpus")
    os.makedirs(expDir, exist_ok=True)
    params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')

    # Add your test data paths to the config before running
    # params.test_data_paths = [...]

    runner = InferenceRunner(params, global_rank=0, local_rank=0, device=device)
    # predictions, ground_truth = runner.run_inference(rollout_steps=10, history_length=10)
    # print("Inference completed.")

    # # Save results to HDF5 file in expDir (recommended for SLURM jobs)
    # output_path = os.path.join(expDir, "inference_results.h5")
    # with h5py.File(output_path, "w") as f:
    #     f.create_dataset("predictions", data=predictions)
    #     f.create_dataset("ground_truth", data=ground_truth)
    #     f.attrs["rollout_steps"] = predictions.shape[0]
    #     f.attrs["history_length"] = 10
    # print(f"Saved predictions and ground truth to {output_path}")
    
    predictions, ground_truth, true_future = runner.run_inference(rollout_steps=10, history_length=10)
    print("Inference completed.")

    output_path = os.path.join(expDir, "inference_results.h5")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("predictions", data=predictions)
        f.create_dataset("ground_truth", data=ground_truth)
        f.create_dataset("true_future", data=true_future)
        f.attrs["rollout_steps"] = predictions.shape[0]
        f.attrs["history_length"] = 10
    print(f"Saved predictions, ground truth, and true future to {output_path}")

    video_path = "/parallel_scratch/jc02423/multiple_physics_pretraining/~/MPP/basic_config/demo_multi_gpus/rollout_video.mp4"
    # Save rollout video for first field/channel
    # save_rollout_video(predictions, ground_truth, video_path, field_idx=0)
    save_rollout_vs_true_video(predictions, true_future, video_path, field_idx=0)