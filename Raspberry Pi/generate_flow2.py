import argparse
import torch
import numpy as np
import sys
import os
import gc
import time
from pathlib import Path
import mcubes
import trimesh

# 1. CFM 경로 탐색 주입
sys.path.append(os.path.abspath('./CFM'))

# 2. Lightning 통합 모듈 및 솔버, 유틸리티 임포트
from CFM.flowmating_lightning import FlowMatchingLightning
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.path.scheduler.scheduler import CondOTScheduler, CosineScheduler
from flow_matching.path.scheduler.schedule_transform import ScheduleTransformedModel
from flow_matching.utils import ModelWrapper
from cod.utils.recon import create_grid_queries

PROMPT_TO_ID = {
    "airplane": 0, "car": 16, "chair": 18, "table": 49,
    "watercraft": 53, "sofa": 47, "rifle": 44, "lamp": 28
}

class FlowWrapper(ModelWrapper):
    """Meta flow_matching의 ModelWrapper 인터페이스에 맞춘 adapter.
    ScheduleTransformedModel이 이 인터페이스를 기대함."""
    def __init__(self, model):
        super().__init__(model=model)

    def forward(self, x, t, **extras):
        if t.dim() == 0:
            t = t.repeat(x.shape[0])
        return self.model(x, t, class_labels=extras.get('class_labels'))

def get_args():
    parser = argparse.ArgumentParser("Flow Matching Text-to-3D Generation (ScheduleTransformed)")
    parser.add_argument("--prompt", type=str, required=True,
                        help="생성할 카테고리 (예: airplane, car, chair, table)")
    parser.add_argument("--vae_config", type=str, required=True)
    parser.add_argument("--vae_weights", type=str, required=True)
    parser.add_argument("--fm_pth", type=str, required=True,
                        help="Flow Matching 체크포인트 경로")
    parser.add_argument("--output", type=str, default="output_flow2.obj")
    parser.add_argument("--density", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4,
                        help="ODE solver 스텝 수 (기본: 4)")
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ── 프롬프트 → 카테고리 ID 변환 ──
    prompt_lower = args.prompt.lower()
    if prompt_lower not in PROMPT_TO_ID:
        raise ValueError(f"지원하지 않는 프롬프트입니다: {prompt_lower}\n지원 목록: {list(PROMPT_TO_ID.keys())}")

    target_id = PROMPT_TO_ID[prompt_lower]
    print(f"🎯 프롬프트 '{prompt_lower}' -> 카테고리 ID [{target_id}] 변환 완료")

    # ── 모델 로딩 ──
    print("✅ 체크포인트 로딩 중...")
    model_load_start = time.time()

    model = FlowMatchingLightning(
        vae_config_path=args.vae_config,
        vae_weights_path=args.vae_weights
    )
    ckpt = torch.load(args.fm_pth, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.to(device)
    model.eval()

    n_latents = model.flow_model.n_latents
    channels = model.flow_model.channels
    print(f"   - 로딩 완료 (소요 시간: {time.time() - model_load_start:.2f}초)")
    print(f"   - Flow 모델 설정: n_latents={n_latents}, channels={channels}, steps={args.steps}")

    # ── ScheduleTransformedModel 적용 ──
    # 학습: CondOT (alpha_t=t, sigma_t=1-t)로 학습된 모델
    # sampling: CosineScheduler (alpha_t=sin(πt/2), sigma_t=cos(πt/2))로 변환
    # → t=0,1 근처에서 궤적이 부드러워져 적은 step으로도 정확한 sampling 가능
    wrapped_model = FlowWrapper(model.flow_model)
    transformed_model = ScheduleTransformedModel(
        velocity_model=wrapped_model,
        original_scheduler=CondOTScheduler(),
        new_scheduler=CosineScheduler()
    )
    solver = ODESolver(velocity_model=transformed_model)

    step_size = 1.0 / args.steps
    print(f"   - ScheduleTransformedModel: CondOT → CosineScheduler 변환 적용")

    # ── 전체 추론 시간 측정 시작 ──
    total_infer_start = time.time()

    with torch.no_grad():
        # 1단계: Flow Matching 샘플링
        print(f"⏳ 1단계: Flow Matching 샘플링 진행 중 ({args.steps}-Step Euler, CosineScheduler)...")
        flow_start = time.time()

        x_init = torch.randn([1, n_latents, channels], device=device)
        cond_tensor = torch.tensor([target_id], dtype=torch.long, device=device)

        sampled_z = solver.sample(
            x_init=x_init,
            step_size=step_size,
            method='euler',
            class_labels=cond_tensor
        )
        sampled_z = sampled_z.to(torch.float32).clone().contiguous()

        flow_time = time.time() - flow_start
        print(f"   - Flow 샘플링 완료 (소요 시간: {flow_time:.2f}초)")

        # Flow 모델 메모리 해제
        print("🧹 Flow 모델 메모리 해제 중...")
        del model.flow_model, wrapped_model, transformed_model, solver
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2단계: VAE 디코딩
        print("🧬 2단계: VAE 디코딩 및 3D 공간 밀도 맵(Logits) 변환 중...")
        vae_start = time.time()

        sampled_z = sampled_z.view(sampled_z.shape[0], n_latents, channels)
        decoded_latents = model.vae.decode_latents(sampled_z)
        context, _, _, _ = model.vae.decode_embed(decoded_latents)

        queries = create_grid_queries(args.density).to(device)
        logits = []
        chunk_size = 100000
        for i in range(0, queries.size(1), chunk_size):
            chunk = queries[:, i:i + chunk_size]
            logit = model.vae.decode_queries(context, chunk)
            logits.append(logit)
        logits = torch.cat(logits, dim=1)

        vae_time = time.time() - vae_start
        print(f"   - VAE 디코딩 완료 (소요 시간: {vae_time:.2f}초)")

    # 3단계: Marching Cubes
    print("🎨 3단계: Marching Cubes로 3D 메쉬 추출 중...")
    mc_start = time.time()

    density = args.density
    gap = 2. / density
    volume = logits.view(density, density, density).permute(2, 1, 0).cpu().numpy()
    vertices, triangles = mcubes.marching_cubes(volume, 0)
    vertices = (vertices - 0.5) * gap - 1.0
    mesh = trimesh.Trimesh(vertices, triangles)
    mesh.export(args.output)

    mc_time = time.time() - mc_start
    print(f"   - 3D 메쉬 추출 및 저장 완료 (소요 시간: {mc_time:.2f}초)")

    total_infer_time = time.time() - total_infer_start

    print("\n" + "=" * 40)
    print(f"🎉 3D 에셋 생성 완료! 파일 저장됨: {args.output}")
    print(f"⏱️ 총 순수 생성 시간: {total_infer_time:.2f}초 (모델 로딩 제외)")
    print(f"   - Flow 샘플링:    {flow_time:.2f}초 ({args.steps}스텝, CosineScheduler)")
    print(f"   - VAE 디코딩:     {vae_time:.2f}초")
    print(f"   - Marching Cubes: {mc_time:.2f}초")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    main()
