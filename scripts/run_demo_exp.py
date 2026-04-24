# Adapted from RED-diff run_reddiff_gp.py for demo
import datetime
import logging
import os
import shutil
import sys
import time
from pathlib import Path
import torch.distributed as dist

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

# Add external dependencies paths
# sys.path.insert(0, "/home/xim22003/Diffusion_CLM/slips")
# sys.path.insert(1, "/home/xim22003/Diffusion_CLM/sde_sampler/sde_sampler")
sys.path.insert(0, os.path.abspath(os.path.join(f"{__file__}/", "../../")))
# sys.path.insert(0, "./")
import os
os.environ["GPyTorch_NO_KEOPS"] = "1"
# print(os.path.abspath(os.path.join(f"{__file__}/", "../../")))

from src.algo.build import build_gp_algo
from src.algo.dataset import build_loader
from src.algo.build import build_model
from src.algo.score_estimator import ReverseDiffusionModel
from src.utils.diffusion import Diffusion
from src.utils.functions import get_timesteps, strfdt

torch.set_printoptions(sci_mode=False)

@hydra.main(version_base="1.2", config_path="../configs", config_name="ddrmpp")
def main(cfg: DictConfig):
    print('cfg.exp.seed', cfg.exp.seed)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f'Experiment name is {cfg.exp.name}')

    exp_root = cfg.exp.root
    samples_root = cfg.exp.samples_root
    exp_name = cfg.exp.name
    samples_root = os.path.join(exp_root, samples_root, exp_name)
    dataset_name = cfg.dataset.name

    # Create output directory
    if cfg.exp.overwrite:
        if os.path.exists(samples_root):
            shutil.rmtree(samples_root)
        os.makedirs(samples_root)
    else:
        if not os.path.exists(samples_root):
            os.makedirs(samples_root)

    # Build models
    gp_model, model = build_model(cfg)

    # Very crucial for correctness of performance
    model.eval()
    gp_model.eval()

    # Build data loader
    loader, columns = build_loader(cfg)
    logger.info(f'Dataset size is {len(loader.dataset)}')

    # Create diffusion process
    diffusion = Diffusion(**cfg.diffusion)
    cg_model = ReverseDiffusionModel(model, diffusion, cfg)
    # Create algorithm
    algo = build_gp_algo(cg_model, gp_model, cfg)

    # Prepare results storage
    start_time = time.time()
    ts = get_timesteps(cfg)

    sample_indices = []
    true_params_all = []  # list of lists per sample
    pred_params_all = []  # list of lists per sample
    mse_values = []       # MSE for parameter columns per sample
    rmse_values = []      # RMSE for trajectory per sample

    # Iterate over all batches in loader
    for batch_idx, (x, y, info) in enumerate(loader):
        # Apply smoke test limit
        if cfg.exp.smoke_test > 0 and batch_idx >= cfg.exp.smoke_test:
            logger.info(f"Smoke test limit reached ({cfg.exp.smoke_test} batches)")
            break

        x, y = x.cuda(), y.cuda()
        batch_indices = info['idx']  # tensor of indices
        if isinstance(batch_indices, torch.Tensor):
            batch_indices = batch_indices.tolist()
        info['logger'] = logger

        # Run RED-diff sampling
        xt_s, mu, mu_list, _, _ = algo.sample(x, y, ts, **info) # type: ignore

        # Save mu_list (optimization history) as .npy file
        mu_history_dir = os.path.join(samples_root, "mu_history")
        os.makedirs(mu_history_dir, exist_ok=True)
        # score_history_dir = os.path.join(samples_root, "score_history")
        # os.makedirs(score_history_dir, exist_ok=True)
        # Convert mu_list from list of numpy arrays to single numpy array
        mu_list_array = np.array(mu_list)  # shape: (total_steps, batch_size, param_dim)
        # score_rmse_array = np.array(score_rmse_list)
        # Create filename with batch index and sample indices
        if not isinstance(batch_indices, list):
            batch_indices = [batch_indices]
        batch_indices_str = "_".join(str(idx) for idx in batch_indices)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        mu_history_file = os.path.join(mu_history_dir, f"mu_history_batch_{batch_idx}_{batch_indices_str}_{timestamp}.npy")
        np.save(mu_history_file, mu_list_array)
        logger.info(f"Saved mu_li   st to {mu_history_file}")
        # score_history_file = os.path.join(score_history_dir, f"score_history_batch_{batch_idx}_{batch_indices_str}_{timestamp}.npy")
        # np.save(score_history_file, score_rmse_array)
        # logger.info(f"Saved score_list to {score_history_file}")


        xo = xt_s.cpu()
        mu_0 = mu.cpu()

        # Compute trajectory RMSE using GP model
        with torch.no_grad():
            mu_P_final, mu_Q_final = gp_model(mu)
            predicted_final = torch.cat([mu_P_final, mu_Q_final], dim=1)
            # RMSE per sample
            batch_rmse = torch.sqrt(torch.mean((predicted_final - y) ** 2, dim=1))

        # Compute parameter MSE for selected columns
        batch_mse = torch.mean((mu_0[:, columns] - x.cpu()[:, columns]) ** 2, dim=1)

        # Store results for each sample in batch
        true_batch = x.cpu()[:, columns]
        pred_batch = mu_0[:, columns]

        for i in range(x.shape[0]):
            sample_idx = batch_indices[i] if isinstance(batch_indices, list) else batch_indices
            sample_indices.append(sample_idx)
            true_params_all.append(true_batch[i].tolist())
            pred_params_all.append(pred_batch[i].tolist())
            mse_values.append(batch_mse[i].item())
            rmse_values.append(batch_rmse[i].item())

        # Log progress
        if (batch_idx + 1) % max(1, len(loader) // 10) == 0 or batch_idx == 0:
            logger.info(f"Processed batch {batch_idx + 1}/{len(loader)}")

    # Save results to CSV
    import csv
    results_file = os.path.join(samples_root, f"results_{dataset_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    # Prepare header
    header = ["sample_idx"]
    for col in columns:
        header.append(f"true_param_col{col}")
    for col in columns:
        header.append(f"pred_param_col{col}")
    header.extend(["mse", "rmse"])

    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, idx in enumerate(sample_indices):
            row = [idx]
            row.extend(true_params_all[i])
            row.extend(pred_params_all[i])
            row.append(mse_values[i])
            row.append(rmse_values[i])
            writer.writerow(row)

    logger.info(f"Results saved results of {batch_idx} to {results_file}")

    # Print summary statistics
    if sample_indices:
        avg_mse = np.mean(mse_values)
        avg_rmse = np.mean(rmse_values)
        logger.info(f"Processed {len(sample_indices)} samples")
        logger.info(f"Average MSE (parameters): {avg_mse:.6f}")
        logger.info(f"Average RMSE (trajectory): {avg_rmse:.6f}")

        # Print first few samples details
        print("\nFirst 3 samples details:")
        for i in range(min(3, len(sample_indices))):
            print(f"\nSample {sample_indices[i]}:")
            for j, col_idx in enumerate(columns):
                true_val = true_params_all[i][j]
                pred_val = pred_params_all[i][j]
                diff = abs(true_val - pred_val)
                print(f"  Column {col_idx}: True={true_val:.4f}, Estimated={pred_val:.4f}, Diff={diff:.4f}")
            print(f"  MSE: {mse_values[i]:.6f}, RMSE: {rmse_values[i]:.6f}")

    logger.info("Done.")
    now = time.time() - start_time
    now_in_hours = strfdt(datetime.timedelta(seconds=now))
    logger.info(f"Total time: {now_in_hours}")


if __name__ == "__main__":
    main()