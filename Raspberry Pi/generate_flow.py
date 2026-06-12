import argparse
import torch
import torch.nn.functional
import numpy as np
import sys
import os
import gc
import time
from pathlib import Path

# ==========================================
# [환경 변수 레벨 차단] 스레드 충돌 및 백엔드 비활성화
# ==========================================
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKLDNN_DISABLED'] = '1'

torch.backends.mkldnn.enabled = False
if hasattr(torch.backends, 'xnnpack'):
    torch.backends.xnnpack.enabled = False
torch.set_num_threads(1)
torch.set_default_dtype(torch.float32)

# ==========================================
# [검증된 패치] C++ Matmul 엔진 붕괴 완전 우회 (Einsum)
# ==========================================
def safe_einsum_linear(input, weight, bias=None):
    orig_shape = input.shape
    x = input.reshape(-1, orig_shape[-1]).float()
    w = weight.float()
    
    out = torch.einsum('xi,ji->xj', x, w)
    
    if bias is not None:
        out = out + bias.float()
        
    return out.reshape(*orig_shape[:-1], out.shape[-1])

torch.nn.functional.linear = safe_einsum_linear
# ==========================================

# CFM 경로 탐색 주입
sys.path.append(os.path.abspath('./CFM'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'external/pointops')))

# Lightning 통합 모듈 및 솔버, 유틸리티 임포트
from CFM.flowmating_lightning import FlowMatchingLightning
from flow_matching.solver.ode_solver import ODESolver
from cod.utils.recon import create_grid_queries, occupancy_to_mesh

PROMPT_TO_ID = {
    "airplane": 0, "car": 16, "chair": 18, "table": 49,
    "watercraft": 53, "sofa": 47, "rifle": 44, "lamp": 28
}

class FlowWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t, **extras):
        if t.dim() == 0:
            t = t.repeat(x.shape[0])
        return self.model(x, t, class_labels=extras.get('class_labels'))

def get_args():
    parser = argparse.ArgumentParser("Flow Matching Text-to-3D Generation Pipeline (CPU)")
    parser.add_argument("--prompt", type=str, required=True,
                        help="생성할 카테고리 (예: airplane, car, chair, table)")
    parser.add_argument("--vae_config", type=str, required=True)
    parser.add_argument("--vae_weights", type=str, required=True)
    parser.add_argument("--fm_pth", type=str, required=True,
                        help="Flow Matching 체크포인트 경로")
    parser.add_argument("--output", type=str, default="output_flow.obj")
    parser.add_argument("--density", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4,
                        help="ODE solver 스텝 수 (기본: 4)")
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device('cpu')

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
    model = model.to(torch.float32)

    n_latents = model.flow_model.n_latents
    channels = model.flow_model.channels
    print(f"   - 로딩 완료 (소요 시간: {time.time() - model_load_start:.2f}초)")
    print(f"   - Flow 모델 설정: n_latents={n_latents}, channels={channels}, steps={args.steps}")

    # ODE Solver 세팅
    wrapped_model = FlowWrapper(model.flow_model)
    solver = ODESolver(velocity_model=wrapped_model)

    step_size = 1.0 / args.steps

    # ── 전체 추론 시간 측정 시작 ──
    total_infer_start = time.time()

    with torch.no_grad():
        # 1단계: Flow Matching 샘플링
        print(f"⏳ 1단계: Flow Matching 샘플링 진행 중 ({args.steps}-Step Euler)...")
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
        del model.flow_model, wrapped_model, solver
        gc.collect()

        # 2단계: VAE 디코딩
        print("🧬 2단계: VAE 디코딩 및 3D 공간 밀도 맵(Logits) 변환 중...")
        vae_start = time.time()

        sampled_z = sampled_z.view(sampled_z.shape[0], n_latents, channels)
        decoded_latents = model.vae.decode_latents(sampled_z)
        context, _, _, _ = model.vae.decode_embed(decoded_latents)

        queries = create_grid_queries(args.density).to(device)
        logits = []
        chunk_size = 50000
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

    mesh = occupancy_to_mesh(logits, r=args.density, threshold=0.0)
    mesh.export(args.output)

    mc_time = time.time() - mc_start
    print(f"   - 3D 메쉬 추출 및 저장 완료 (소요 시간: {mc_time:.2f}초)")

    total_infer_time = time.time() - total_infer_start

    print("\n" + "=" * 40)
    print(f"🎉 3D 에셋 생성 완료! 파일 저장됨: {args.output}")
    print(f"⏱️ 총 순수 생성 시간: {total_infer_time:.2f}초 (모델 로딩 제외)")
    print(f"   - Flow 샘플링:    {flow_time:.2f}초 ({args.steps}스텝)")
    print(f"   - VAE 디코딩:     {vae_time:.2f}초")
    print(f"   - Marching Cubes: {mc_time:.2f}초")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    main()