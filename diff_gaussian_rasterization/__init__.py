#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# [ObjectLoopSplat - 솔로 래스터라이저 (Solo Rasterizer)]
# 이 패키지는 3DGS, w_pose, SemGaussSLAM의 래스터라이저를 통합한 버전
# Pytorch 레벨에서 포즈 최적화를 수행하기 위해 불필요한 C++ 레벨의 dL_dtau 로직을 제거
# JIT 로딩을 통해 시스템 설치 없이 동적으로 사용 가능하도록 구성

import os
import torch
import torch.nn as nn
from typing import NamedTuple
from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_C = load(
    name="diff_gaussian_rasterization_solo",
    sources=[
        os.path.join(_src_path, "ext.cpp"),
        os.path.join(_src_path, "rasterize_points.cu"),
        os.path.join(_src_path, "cuda_rasterizer", "backward.cu"),
        os.path.join(_src_path, "cuda_rasterizer", "forward.cu"),
        os.path.join(_src_path, "cuda_rasterizer", "rasterizer_impl.cu"),
    ],
    extra_cuda_cflags=["-I" + os.path.join(_src_path, "third_party/glm/")],
    verbose=True
)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    # semantic
    sh_sems,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        # semantic
        sh_sems,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        # semantic
        sh_sems,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            # semantic
            sh_sems,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        # num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
        # semantic
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, depth, alpha, semantics = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, sh_sems)
        return color, radii, depth, alpha, semantics

    # 输入的 grad_out_color, _, depth, grad_out_semantics 分别对应forward函数的输出 color, radii, depth, semantics
    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth, grad_out_alpha, grad_out_semantics):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, sh_sems = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                grad_out_depth,
                grad_out_alpha,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                # semantic
                grad_out_semantics,
                sh_sems)

        # Compute gradients for relevant tensors by invoking backward method
        (grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales,
         grad_rotations, grad_semantics) = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None, # raster_settings
            # semantic
            grad_semantics,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs=None, colors_precomp=None,
                scales=None, rotations=None, cov3D_precomp=None, sh_sems=None):
        """
        가우시안 래스터화 수행. 반환값: (color, radii, depth, alpha, semantics)

        Args:
            means3D (torch.Tensor): 3D 가우시안 중심 위치 (P, 3)
            means2D (torch.Tensor): 2D 화면 좌표 (P, 3), 그래디언트 추적용
            opacities (torch.Tensor): 불투명도 (P, 1)
            shs (torch.Tensor | None): SH 계수. colors_precomp와 상호 배타적
            colors_precomp (torch.Tensor | None): 사전 계산된 색상. shs와 상호 배타적
            scales (torch.Tensor | None): 가우시안 스케일. cov3D_precomp와 상호 배타적
            rotations (torch.Tensor | None): 가우시안 회전. cov3D_precomp와 상호 배타적
            cov3D_precomp (torch.Tensor | None): 사전 계산된 3D 공분산. scales/rotations와 상호 배타적
            sh_sems (torch.Tensor | None): 시맨틱 SH 피처 (P, NUM_LABELS). None이면 빈 텐서 전달

        Returns:
            (color, radii, depth, alpha, semantics) (tuple)
        """
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('shs 또는 colors_precomp 중 하나만 제공해야 합니다.')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or \
                ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('scales/rotations 또는 cov3D_precomp 중 하나만 제공해야 합니다.')

        # None 값을 빈 텐서로 변환 (C++ 인터페이스 호환)
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])
        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if sh_sems is None:
            sh_sems = torch.Tensor([])

        # C++/CUDA 래스터라이저 호출, 반환: color, radii, depth, alpha, semantics
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
            sh_sems,
        )

