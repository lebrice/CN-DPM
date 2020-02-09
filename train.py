import os
import pickle
import torch
from tensorboardX import SummaryWriter
from models import NdpmModel
from data import DataScheduler
from typing import Dict
from torch import Tensor

def _write_summary(summary: Dict[str, Dict], writer: SummaryWriter, step):
    for summary_type, summary_dict in summary.items():
        if summary_type == "scalar":
            write_fn = writer.add_scalar
        elif summary_type == "image":
            write_fn = writer.add_image
        elif summary_type == "histogram":
            write_fn = writer.add_histogram
        else:
            raise RuntimeError("Unsupported summary type: %s" % summary_type)

        for tag, value in summary_dict.items():
            write_fn(tag, value, step)


def _make_collage(samples: Tensor, config: Dict, grid_h: int, grid_w: int) -> Tensor:
    s = samples.view(
        grid_h, grid_w,
        config["x_c"], config["x_h"], config["x_w"]
    )
    collage = s.permute(2, 0, 3, 1, 4).contiguous().view(
        config["x_c"],
        config["x_h"] * grid_h,
        config["x_w"] * grid_w
    )
    return collage


def train_model(config: Dict, model: NdpmModel,
                scheduler: DataScheduler,
                writer: SummaryWriter):
    saved_model_path = os.path.join(config["log_dir"], "ckpts")
    os.makedirs(saved_model_path, exist_ok=True)

    is_ndmp = isinstance(model, NdpmModel)

    for step, (x, y, t) in enumerate(scheduler):
        step += 1
        if is_ndmp:
            stm_item_count = len(model.ndpm.stm_x)
            stm_capacity = config["stm_capacity"]
            print(
                f"\r[Step {step:4}]",
                f"STM: {stm_item_count:5}/{stm_capacity}",
                f"| #Expert: {len(model.ndpm.experts) - 1}",
                end="",
            )
        else:
            print("\r[Step {:4}]".format(step), end="")

        summarize = step % config["summary_step"] == 0
        summarize_experts = summarize and isinstance(model, NdpmModel)
        summarize_samples = summarize and config["summarize_samples"]

        # learn the model
        model.learn(x, y, t, step)

        # Evaluate the model
        evaluatable = (
            not is_ndmp or len(model.ndpm.experts) > 1
        )
        if evaluatable and step % config["eval_step"] == 0:
            scheduler.eval(model, writer, step, "model")

        if step % config["ckpt_step"] == 0:
            print("\nSaving checkpoint... ", end="")
            ckpt_path = os.path.join(saved_model_path,
                                     "ckpt-{}.pt".format(str(step).zfill(6)))
            del model.writer
            if is_ndmp:
                del model.ndpm.writer
            with open(ckpt_path, "wb") as f:
                pickle.dump(model, f)
            model.writer = writer
            if is_ndmp:
                model.ndpm.writer = writer
            print("Saved to {}".format(ckpt_path))

        # Evaluate experts of the model"s DPMoE
        if summarize_experts:
            writer.add_scalar("num_experts", len(model.ndpm.experts) - 1, step)

        # Summarize samples
        if summarize_samples:
            if is_ndmp:
                comps = [e.g for e in model.ndpm.experts[1:]]
            else:
                comps = [model.component]
            
            if len(comps) == 0:
                continue
            grid_h, grid_w = config["sample_grid"]
            total_samples = []
            # Sample from each expert
            for i, expert in enumerate(comps):
                with torch.no_grad():
                    samples = expert.sample(grid_h * grid_w)
                total_samples.append(samples)
                collage = _make_collage(samples, config, grid_h, grid_w)
                writer.add_image(f"samples/{i + 1}", collage, step)

            if is_ndmp:
                counts = model.ndpm.prior.counts[1:]
                expert_w = counts / counts.sum()
                num_samples = torch.distributions.multinomial.Multinomial(
                    grid_h * grid_w, probs=expert_w).sample().type(torch.int)
                to_collage = []
                for i, samples in enumerate(total_samples):
                    to_collage.append(samples[:num_samples[i]])
                to_collage = torch.cat(to_collage, dim=0)
                collage = _make_collage(to_collage, config, grid_h, grid_w)
                writer.add_image("samples/ndpm", collage, step)
