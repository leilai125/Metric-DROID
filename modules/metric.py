# standard library
from pathlib import Path
from typing import *
import sys
# third party
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from mmengine import Config
# metric 3d
metric3d_path = Path(__file__).resolve().parent / 'metric3d'
metric3d_mono_path = metric3d_path / 'mono'
sys.path.append(str(metric3d_path))
sys.path.append(str(metric3d_mono_path))
from .metric3d.mono.model.monodepth_model import get_configured_monodepth_model
from .metric3d.mono.utils.running import load_ckpt
from .metric3d.mono.utils.do_test import transform_test_data_scalecano, get_prediction
from .metric3d.mono.utils.mldb import load_data_info, reset_ckpt_path
from .metric3d.mono.utils.transform import gray_to_colormap

__ALL__ = ['Metric3D']

class Metric3D:
    cfg_: Config
    model_: torch.nn.Module

    def __init__(
        self,
        checkpoint: Union[str, Path] = './weights/metric_depth_vit_large_800k.pth',
        model_name: str = 'v2-L',
    ) -> None:
        checkpoint = Path(checkpoint).resolve()
        cfg:Config = self._load_config_(model_name, checkpoint)
        # build model
        model = get_configured_monodepth_model(cfg, )
        model = torch.nn.DataParallel(model).cuda()
        model, _, _, _ = load_ckpt(cfg.load_from, model, strict_match=False)
        model.eval()
        # save to self
        self.cfg_ = cfg
        self.model_ = model

    @staticmethod
    def _intrinsic_from_layer(intrinsic_layer: torch.Tensor) -> torch.Tensor:
        if intrinsic_layer.ndim == 3:
            intrinsic_layer = intrinsic_layer.unsqueeze(0)
        if intrinsic_layer.ndim != 4 or intrinsic_layer.shape[1] != 2:
            raise ValueError(f"Expected intrinsic layer [B,2,H,W] or [2,H,W], got {tuple(intrinsic_layer.shape)}")

        dx = intrinsic_layer[:, 0:1, :, 1:] - intrinsic_layer[:, 0:1, :, :-1]
        dy = intrinsic_layer[:, 1:2, 1:, :] - intrinsic_layer[:, 1:2, :-1, :]

        fx = dx.abs().mean(dim=(1, 2, 3)).clamp_min(1e-8).reciprocal()
        fy = dy.abs().mean(dim=(1, 2, 3)).clamp_min(1e-8).reciprocal()
        cx = 0.5 - intrinsic_layer[:, 0, 0, 0] * fx
        cy = 0.5 - intrinsic_layer[:, 1, 0, 0] * fy
        return torch.stack((fx, fy, cx, cy), dim=-1)

    def _model_device(self) -> torch.device:
        return next(self.model_.parameters()).device

    def _as_intrinsic_tensor(
        self,
        intrinsic: Union[str, Path, np.ndarray, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if torch.is_tensor(intrinsic):
            if intrinsic.ndim in (3, 4) and intrinsic.shape[-3] == 2:
                intrinsic_tensor = self._intrinsic_from_layer(intrinsic)
            elif intrinsic.ndim == 1 and intrinsic.numel() >= 4:
                intrinsic_tensor = intrinsic[:4].unsqueeze(0)
            elif intrinsic.ndim == 2 and intrinsic.shape[-1] >= 4:
                intrinsic_tensor = intrinsic[:, :4]
            else:
                raise ValueError(f"Unsupported intrinsic tensor shape: {tuple(intrinsic.shape)}")
            intrinsic_tensor = intrinsic_tensor.to(device=device, dtype=torch.float32)
        else:
            if isinstance(intrinsic, (str, Path)):
                intrinsic = np.loadtxt(intrinsic)
            intrinsic_np = np.asarray(intrinsic, dtype=np.float32)
            if intrinsic_np.ndim == 1:
                intrinsic_tensor = torch.from_numpy(intrinsic_np[:4]).to(device=device, dtype=torch.float32).unsqueeze(0)
            elif intrinsic_np.ndim == 2 and intrinsic_np.shape[-1] >= 4:
                intrinsic_tensor = torch.from_numpy(intrinsic_np[:, :4]).to(device=device, dtype=torch.float32)
            elif intrinsic_np.ndim == 3 and intrinsic_np.shape[0] == 2:
                intrinsic_tensor = self._intrinsic_from_layer(torch.from_numpy(intrinsic_np)).to(device=device)
            else:
                raise ValueError(f"Unsupported intrinsic array shape: {intrinsic_np.shape}")

        if intrinsic_tensor.shape[0] == 1 and batch_size > 1:
            intrinsic_tensor = intrinsic_tensor.expand(batch_size, -1).clone()
        if intrinsic_tensor.shape[0] != batch_size:
            raise ValueError(f"Intrinsic batch size {intrinsic_tensor.shape[0]} does not match image batch size {batch_size}")
        return intrinsic_tensor

    def _as_rgb_tensor(
        self,
        rgb_image: Union[np.ndarray, Image.Image, str, Path, torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, bool]:
        if isinstance(rgb_image, (str, Path)):
            rgb_image = np.array(Image.open(rgb_image))
        elif isinstance(rgb_image, Image.Image):
            rgb_image = np.array(rgb_image)

        if torch.is_tensor(rgb_image):
            rgb_tensor = rgb_image.detach()
            if rgb_tensor.ndim == 3 and rgb_tensor.shape[0] in (1, 3):
                rgb_tensor = rgb_tensor.unsqueeze(0)
                is_batched = False
            elif rgb_tensor.ndim == 4 and rgb_tensor.shape[1] in (1, 3):
                is_batched = True
            elif rgb_tensor.ndim == 3 and rgb_tensor.shape[-1] in (1, 3):
                rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0)
                is_batched = False
            elif rgb_tensor.ndim == 4 and rgb_tensor.shape[-1] in (1, 3):
                rgb_tensor = rgb_tensor.permute(0, 3, 1, 2)
                is_batched = True
            else:
                raise ValueError(f"Unsupported rgb tensor shape: {tuple(rgb_tensor.shape)}")
            rgb_tensor = rgb_tensor.to(device=device, dtype=torch.float32)
        else:
            rgb_np = np.asarray(rgb_image)
            if rgb_np.ndim == 3:
                rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
                is_batched = False
            elif rgb_np.ndim == 4:
                rgb_tensor = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
                is_batched = True
            else:
                raise ValueError(f"Unsupported rgb image shape: {rgb_np.shape}")

        if float(rgb_tensor.amax().detach().cpu()) <= 1.5:
            rgb_tensor = rgb_tensor * 255.0
        return rgb_tensor, is_batched

    @staticmethod
    def _build_camera_model_tensor(height: int, width: int, intrinsics: torch.Tensor) -> torch.Tensor:
        fx, fy, u0, v0 = intrinsics.unbind(dim=-1)
        f = 0.5 * (fx + fy)

        x = torch.arange(width, device=intrinsics.device, dtype=intrinsics.dtype).view(1, 1, width)
        y = torch.arange(height, device=intrinsics.device, dtype=intrinsics.dtype).view(1, height, 1)

        x_center = ((x - u0[:, None, None]) / float(width)).expand(-1, height, -1)
        y_center = ((y - v0[:, None, None]) / float(height)).expand(-1, -1, width)

        fov_x = torch.atan(x_center / (f[:, None, None] / float(width)))
        fov_y = torch.atan(y_center / (f[:, None, None] / float(height)))
        return torch.stack((x_center, y_center, fov_x, fov_y), dim=1)

    def _resize_for_input_tensor(
        self,
        image: torch.Tensor,
        intrinsic: torch.Tensor,
        output_shape: Sequence[int],
        canonical_shape: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], torch.Tensor]:
        output_h, output_w = int(output_shape[0]), int(output_shape[1])
        canonical_h, canonical_w = int(canonical_shape[0]), int(canonical_shape[1])

        resize_ratio_h = output_h / canonical_h
        resize_ratio_w = output_w / canonical_w
        to_scale_ratio = min(resize_ratio_h, resize_ratio_w)

        _, _, h, w = image.shape
        reshape_h = max(int(to_scale_ratio * h), 1)
        reshape_w = max(int(to_scale_ratio * w), 1)

        pad_h = max(output_h - reshape_h, 0)
        pad_w = max(output_w - reshape_w, 0)
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2

        resized = F.interpolate(image, size=(reshape_h, reshape_w), mode="bilinear", align_corners=False)

        padding = image.new_tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
        padded = padding.expand(image.shape[0], -1, output_h, output_w).clone()
        padded[:, :, pad_h_half : pad_h_half + reshape_h, pad_w_half : pad_w_half + reshape_w] = resized

        intrinsic_scaled = intrinsic.clone()
        intrinsic_scaled[:, 2] = intrinsic_scaled[:, 2] * to_scale_ratio
        intrinsic_scaled[:, 3] = intrinsic_scaled[:, 3] * to_scale_ratio

        cam_model = self._build_camera_model_tensor(reshape_h, reshape_w, intrinsic_scaled)
        cam_model_padded = image.new_full((image.shape[0], 4, output_h, output_w), -1.0)
        cam_model_padded[:, :, pad_h_half : pad_h_half + reshape_h, pad_w_half : pad_w_half + reshape_w] = cam_model

        pad = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        label_scale_factor = image.new_full((image.shape[0],), 1.0 / to_scale_ratio)
        return padded, cam_model_padded, pad, label_scale_factor

    def _transform_test_data_scalecano_tensor(
        self,
        rgb: torch.Tensor,
        intrinsic: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[int], torch.Tensor, Tuple[int, int]]:
        canonical_space = self.cfg_.data_basic['canonical_space']
        forward_size = self.cfg_.data_basic.crop_size
        mean = rgb.new_tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
        std = rgb.new_tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)

        _, _, ori_h, ori_w = rgb.shape
        ori_focal = 0.5 * (intrinsic[:, 0] + intrinsic[:, 1])
        canonical_focal = float(canonical_space['focal_length'])
        cano_label_scale_ratio = canonical_focal / ori_focal

        canonical_intrinsic = intrinsic.clone()
        canonical_intrinsic[:, 0] = canonical_intrinsic[:, 0] * cano_label_scale_ratio
        canonical_intrinsic[:, 1] = canonical_intrinsic[:, 1] * cano_label_scale_ratio

        rgb, cam_model, pad, resize_label_scale_ratio = self._resize_for_input_tensor(
            rgb,
            canonical_intrinsic,
            forward_size,
            [ori_h, ori_w],
        )

        label_scale_factor = cano_label_scale_ratio * resize_label_scale_ratio
        rgb = torch.div(rgb - mean, std)
        cam_model_stacks = [
            F.interpolate(
                cam_model,
                size=(max(cam_model.shape[2] // i, 1), max(cam_model.shape[3] // i, 1)),
                mode='bilinear',
                align_corners=False,
            )
            for i in [2, 4, 8, 16, 32]
        ]
        return rgb, cam_model_stacks, pad, label_scale_factor, (ori_h, ori_w)

    @torch.no_grad()
    def predict_tensor(
        self,
        rgb_image: Union[np.ndarray, Image.Image, str, Path, torch.Tensor],
        intrinsic: Union[str, Path, np.ndarray, torch.Tensor],
        d_max: Optional[float] = 300,
        d_min: Optional[float] = 0,
    ) -> torch.Tensor:
        device = self._model_device()
        rgb_tensor, is_batched = self._as_rgb_tensor(rgb_image, device=device)
        intrinsic_tensor = self._as_intrinsic_tensor(intrinsic, batch_size=rgb_tensor.shape[0], device=device)

        rgb_input, cam_models_stacks, pad, label_scale_factor, ori_shape = self._transform_test_data_scalecano_tensor(
            rgb_tensor,
            intrinsic_tensor,
        )

        pred_depth, _, _ = self.model_.module.inference(
            dict(
                input=rgb_input,
                cam_model=cam_models_stacks,
            )
        )

        if pred_depth.ndim == 4 and pred_depth.shape[1] == 1:
            pred_depth = pred_depth[:, 0]
        elif pred_depth.ndim == 2:
            pred_depth = pred_depth.unsqueeze(0)
        elif pred_depth.ndim != 3:
            raise ValueError(f"Unexpected depth prediction shape: {tuple(pred_depth.shape)}")

        pad_top, pad_bottom, pad_left, pad_right = pad
        h_end = pred_depth.shape[-2] - pad_bottom if pad_bottom > 0 else pred_depth.shape[-2]
        w_end = pred_depth.shape[-1] - pad_right if pad_right > 0 else pred_depth.shape[-1]
        pred_depth = pred_depth[:, pad_top:h_end, pad_left:w_end]

        pred_depth = F.interpolate(
            pred_depth.unsqueeze(1),
            size=ori_shape,
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)

        normalize_scale = float(self.cfg_.data_basic.depth_range[1])
        pred_depth = pred_depth * normalize_scale / label_scale_factor[:, None, None]

        if d_max is not None:
            pred_depth = torch.where(pred_depth <= float(d_max), pred_depth, torch.zeros_like(pred_depth))
        if d_min is not None:
            pred_depth = torch.where(pred_depth >= float(d_min), pred_depth, torch.zeros_like(pred_depth))

        return pred_depth if is_batched else pred_depth[0]

    @torch.no_grad()
    def __call__(
        self,
        rgb_image: Union[np.ndarray, Image.Image, str, Path, torch.Tensor],
        intrinsic: Union[str, Path, np.ndarray, torch.Tensor],
        d_max: Optional[float] = 300,
        d_min: Optional[float] = 0
    ) -> Union[np.ndarray, torch.Tensor]:
        pred_depth = self.predict_tensor(
            rgb_image=rgb_image,
            intrinsic=intrinsic,
            d_max=d_max,
            d_min=d_min,
        )
        if torch.is_tensor(rgb_image):
            return pred_depth
        return pred_depth.detach().cpu().numpy()

    def _load_config_(
        self,
        model_name: str,
        checkpoint: Union[str, Path],
    ) -> Config:
        print(f'Loading model {model_name} from {checkpoint}')
        config_path = metric3d_path / 'mono/configs/HourglassDecoder'
        assert model_name in ['v2-L', 'v2-S', 'v2-g'], f"Model {model_name} not supported"
        # load config file
        cfg = Config.fromfile(
            str(config_path / 'vit.raft5.large.py') if model_name == 'v2-L' 
            else str(config_path / 'vit.raft5.small.py') if model_name == 'v2-S' 
            else str(config_path / 'vit.raft5.giant2.py')
        )
        cfg.load_from = str(checkpoint)
        # load data info
        data_info = {}
        load_data_info('data_info', data_info=data_info)
        cfg.mldb_info = data_info
        # update check point info
        reset_ckpt_path(cfg.model, data_info)
        # set distributed
        cfg.distributed = False
        
        return cfg
    
    @staticmethod
    def gray_to_colormap(depth: np.ndarray) -> np.ndarray:
        return gray_to_colormap(depth)
