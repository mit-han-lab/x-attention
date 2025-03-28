import os

import datasets
import torch
import torchmetrics
import torchvision
from PIL import Image
from torch.utils import data
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from tqdm import tqdm

__all__ = ["compute_image_similarity_metrics", "compute_video_similarity_metrics"]


class MultiImageDataset(data.Dataset):
    def __init__(
        self, gen_dirpath: str, ref_dirpath_or_dataset: str | datasets.Dataset
    ):
        super(data.Dataset, self).__init__()
        self.gen_names = sorted(
            [
                name
                for name in os.listdir(gen_dirpath)
                if name.endswith(".png") or name.endswith(".jpg")
            ]
        )
        self.gen_dirpath, self.ref_dirpath_or_dataset = (
            gen_dirpath,
            ref_dirpath_or_dataset,
        )
        if isinstance(ref_dirpath_or_dataset, str):
            self.ref_names = sorted(
                [
                    name
                    for name in os.listdir(ref_dirpath_or_dataset)
                    if name.endswith(".png") or name.endswith(".jpg")
                ]
            )
            assert len(self.ref_names) == len(self.gen_names)
        else:
            assert isinstance(ref_dirpath_or_dataset, datasets.Dataset)
            self.ref_names = self.gen_names
            assert len(ref_dirpath_or_dataset) == len(self.gen_names)
        self.transform = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.ref_names)

    def __getitem__(self, idx: int):
        if isinstance(self.ref_dirpath_or_dataset, str):
            name = self.ref_names[idx]
            assert name == self.gen_names[idx]
            ref_image = Image.open(
                os.path.join(self.ref_dirpath_or_dataset, name)
            ).convert("RGB")
        else:
            row = self.ref_dirpath_or_dataset[idx]
            ref_image = row["image"].convert("RGB")
            name = row["filename"] + ".png"
        gen_image = Image.open(os.path.join(self.gen_dirpath, name)).convert("RGB")
        gen_size = gen_image.size
        ref_size = ref_image.size
        if ref_size != gen_size:
            ref_image = ref_image.resize(gen_size, Image.Resampling.BICUBIC)
        gen_tensor = self.transform(gen_image)
        ref_tensor = self.transform(ref_image)
        return [gen_tensor, ref_tensor]


class MultiVideoDataset(data.Dataset):
    def __init__(self, gen_dirpath: str, ref_dirpath: str):
        super(data.Dataset, self).__init__()
        gen_filenames = os.listdir(gen_dirpath)
        self.gen_names = sorted(
            [
                os.path.relpath(os.path.join(root, name), gen_dirpath)
                for root, _, files in os.walk(gen_dirpath)
                for name in files
                if name.endswith(".mp4")
            ]
        )
        self.ref_names = sorted(
            [
                os.path.relpath(os.path.join(root, name), ref_dirpath)
                for root, _, files in os.walk(ref_dirpath)
                for name in files
                if name.endswith(".mp4")
            ]
        )
        self.ref_names_dict = {}
        for name in self.ref_names:
            idx = int(name.split("/")[0])
            self.ref_names_dict[idx] = name
        self.gen_names_dict = {}
        for name in self.gen_names:
            idx = int(name.split("/")[0])
            self.gen_names_dict[idx] = name
        # sort by idx
        self.ref_names_dict = sorted(self.ref_names_dict.items(), key=lambda x: x[0])
        self.gen_names_dict = sorted(self.gen_names_dict.items(), key=lambda x: x[0])
        self.ref_names = [name for _, name in self.ref_names_dict]
        print(len(self.ref_names))
        print(self.ref_names[:2])
        self.gen_names = [name for _, name in self.gen_names_dict]
        print(len(self.gen_names))
        print(self.gen_names[:2])
        assert len(self.ref_names) == len(self.gen_names)
        self.gen_dirpath, self.ref_dirpath = gen_dirpath, ref_dirpath

    def __len__(self):
        return len(self.ref_names)

    def __getitem__(self, idx: int):
        ref_name = self.ref_names[idx]
        gen_name = self.gen_names[idx]
        ref_video, ref_audio, ref_info = torchvision.io.read_video(
            os.path.join(self.ref_dirpath, ref_name),
            pts_unit="sec",
            output_format="TCHW",
        )
        if ref_video.max() > 1:
            ref_video = ref_video / 255.0
        gen_video, gen_audio, gen_info = torchvision.io.read_video(
            os.path.join(self.gen_dirpath, gen_name),
            pts_unit="sec",
            output_format="TCHW",
        )
        if gen_video.max() > 1:
            gen_video = gen_video / 255.0
        ref_size = gen_video.size()
        gen_size = ref_video.size()
        assert ref_size == gen_size
        return [gen_video, gen_info, ref_video, ref_info]


def compute_image_similarity_metrics(
    ref_dirpath_or_dataset: str | datasets.Dataset,
    gen_dirpath: str,
    metrics: tuple[str, ...] = ("psnr", "lpips", "ssim"),
    batch_size: int = 64,
    num_workers: int = 8,
    device: str | torch.device = "cuda",
) -> dict[str, float]:
    if len(metrics) == 0:
        return {}
    metric_names = metrics
    metrics: dict[str, torchmetrics.Metric] = {}
    for metric_name in metric_names:
        if metric_name == "psnr":
            metric = PeakSignalNoiseRatio(
                data_range=(0, 1), reduction="elementwise_mean", dim=(1, 2, 3)
            ).to(device)
        elif metric_name == "lpips":
            metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
        elif metric_name == "ssim":
            metric = StructuralSimilarityIndexMeasure(data_range=(0, 1)).to(device)
        else:
            raise NotImplementedError(f"Metric {metric_name} is not implemented")
        metrics[metric_name] = metric
    dataset = MultiImageDataset(gen_dirpath, ref_dirpath_or_dataset)
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )
    with torch.no_grad():
        desc = (
            ref_dirpath_or_dataset.config_name
            if isinstance(ref_dirpath_or_dataset, datasets.Dataset)
            else os.path.basename(ref_dirpath_or_dataset)
        ) + " similarity metrics"
        for batch in tqdm(dataloader, desc=desc):
            batch = [tensor.to(device) for tensor in batch]
            for metric in metrics.values():
                metric.update(batch[0], batch[1])
    result = {
        metric_name: metric.compute().item() for metric_name, metric in metrics.items()
    }
    return result


def compute_video_similarity_metrics(
    ref_dirpath: str,
    gen_dirpath: str,
    metrics: list[str] | tuple[str] = ("psnr", "lpips", "ssim"),
    batch_size: int = 2,
    num_workers: int = 8,
    device: str | torch.device = "cuda",
) -> dict[str, float]:
    if len(metrics) == 0:
        return {}
    metric_names = metrics
    metrics: dict[str, torchmetrics.Metric] = {}
    for metric_name in metric_names:
        if metric_name == "psnr":
            metric = PeakSignalNoiseRatio(
                data_range=(0, 1), reduction="elementwise_mean", dim=(1, 2, 3)
            ).to(device)
        elif metric_name == "lpips":
            metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
        elif metric_name == "ssim":
            metric = StructuralSimilarityIndexMeasure(data_range=(0, 1)).to(device)
        else:
            raise NotImplementedError(f"Metric {metric_name} is not implemented")
        metrics[metric_name] = metric
    dataset = MultiVideoDataset(gen_dirpath, ref_dirpath)
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )
    with torch.no_grad():
        desc = os.path.basename(ref_dirpath) + " similarity metrics"
        for i, batch in enumerate(tqdm(dataloader, desc=desc)):
            gen_video = batch[0].to(device)
            ref_video = batch[2].to(device)
            gen_frames = gen_video.view(-1, *gen_video.shape[-3:])
            ref_frames = ref_video.view(-1, *ref_video.shape[-3:])
            for metric in metrics.values():
                metric.update(gen_frames, ref_frames)
    result = {
        metric_name: metric.compute().item() for metric_name, metric in metrics.items()
    }
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_dirpath",
        type=str,
        default="results_small/flash/all_dimension_longer",
    )
    parser.add_argument(
        "--gen_dirpath",
        type=str,
        default="results_small/xattn_warmup=10_threshold=0.9/all_dimension_longer",
    )
    args = parser.parse_args()

    result = compute_video_similarity_metrics(args.ref_dirpath, args.gen_dirpath)
    print(result)
