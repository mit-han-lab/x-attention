import os
import argparse
import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--num_sampled_prompts", type=int, default=None)
parser.add_argument("--num_seed", type=int, default=1)
parser.add_argument("--attn", type=str, default="xattn")
parser.add_argument(
    "--attn_warmup_steps",
    type=int,
    default=5,
    help="Number of warmup steps for attention.",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.9,
    help="threshold for attention.",
)
parser.add_argument(
    "--stride",
    type=int,
    default=8,
    help="stride for attention.",
)

parser.add_argument("--prompt_path", type=str, default="all_dimension_longer.txt")
args = parser.parse_args()

prompt_path = args.prompt_path
prompt_set_name = prompt_path.split("/")[-1].split(".")[0]

prompt_list = []
with open(prompt_path, "r") as f:
    for line in f.readlines():
        prompt_list.append(line.strip())

prompt_index = list(range(len(prompt_list)))
if args.num_sampled_prompts is not None:
    # uniformly sample num_sampled_prompts
    prompt_list = prompt_list[:: len(prompt_list) // args.num_sampled_prompts]
    prompt_list = prompt_list[: args.num_sampled_prompts]
    prompt_index = prompt_index[:: len(prompt_index) // args.num_sampled_prompts]
    prompt_index = prompt_index[: args.num_sampled_prompts]

attn_name = (
    f"xattn_warmup={args.attn_warmup_steps}_threshold={args.threshold}_stride={args.stride}"
    if args.attn == "xattn"
    else args.attn
)

# slurm_script = "sbatch"

job_script = "python3 sample_video.py"

pbar = tqdm.tqdm(total=len(prompt_list) * args.num_seed)
for idx, prompt in enumerate(prompt_list):
    for seed in range(args.num_seed):
        pbar.set_description(f"Sampling attn {args.attn} prompt {prompt} seed {seed}")
        save_path = f"./results_720p/{attn_name}/{prompt_set_name}/{prompt_index[idx]}"
        # if has *.mp4 in save_path, skip
        if os.path.exists(save_path) and any(
            os.path.isfile(os.path.join(save_path, f))
            for f in os.listdir(save_path)
            if f.endswith(".mp4")
        ):
            pbar.update(1)
            print(f"Skipping {save_path} because it already exists")
            continue
        attn_args = f"--attn {args.attn} "
        if args.attn == "xattn":
            attn_args += f"--attn-warmup-steps {args.attn_warmup_steps} --threshold {args.threshold} --stride {args.stride} "
        # slurm_params = f"--job-name=hunyuan/{attn_name}/{prompt_set_name}/prompt_{prompt_index[idx]}_seed{seed} slurm_scripts/submit1.sh "
        os.system(
            # f"{slurm_script} "
            # f"{slurm_params} "
            f"{job_script} "
            f"--video-size 720 1280 "
            # f"--video-size 544 960 "
            f"--video-length 129 "
            f"--infer-steps 50 "
            f'--prompt "{prompt}" '
            f"--flow-reverse "
            f"--save-path ./results_720p/{attn_name}/{prompt_set_name}/{prompt_index[idx]} "
            f"--seed {seed} {attn_args}"
        )

print(len(prompt_list))
