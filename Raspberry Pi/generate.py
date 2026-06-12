import argparse
import torch
import torch.nn.functional
import numpy as np
import sys
import os
import gc
import time  # [추가] 시간 측정을 위한 라이브러리

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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'external/pointops')))

from duffusion.diffusion_lightning import CategoryConditionedDiffusion
from cod.utils.recon import create_grid_queries, occupancy_to_mesh

PROMPT_TO_ID = {
    "airplane": 0, "car": 16, "chair": 18, "table": 49,
    "watercraft": 53, "sofa": 47, "rifle": 44, "lamp": 28
}

def get_args():
    parser = argparse.ArgumentParser("Text-to-3D Generation Pipeline")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--diffusion_ckpt", type=str, required=True)
    parser.add_argument("--vae_config", type=str, required=True)
    parser.add_argument("--vae_weights", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.obj")
    parser.add_argument("--density", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.0)
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device('cpu')

    prompt_lower = args.prompt.lower()
    if prompt_lower not in PROMPT_TO_ID:
        raise ValueError(f"지원하지 않는 프롬프트입니다: {prompt_lower}")

    target_id = PROMPT_TO_ID[prompt_lower]
    cond_tensor = torch.tensor([target_id], dtype=torch.long, device=device)
    print(f"🎯 프롬프트 '{prompt_lower}' -> 카테고리 ID [{target_id}] 변환 완료")

    print("✅ 체크포인트 로딩 중...")
    model_load_start = time.time()
    model = CategoryConditionedDiffusion.load_from_checkpoint(
        checkpoint_path=args.diffusion_ckpt,
        vae_config_path=args.vae_config,
        vae_weights_path=args.vae_weights,
        map_location=device
    )
    print(f"   - 로딩 완료 (소요 시간: {time.time() - model_load_start:.2f}초)")

    model.to(device)
    model.eval()
    model = model.to(torch.float32)

    # 전체 추론 시간 측정 시작
    total_infer_start = time.time()

    with torch.no_grad():
        print("⏳ 1단계: 디퓨전 모델 샘플링 진행 중...")
        diff_start = time.time()
        batch_seeds = torch.randint(0, 1000000, (1,)).to(device)
        sampled_z = model.diffusion_model.sample(cond=cond_tensor, batch_seeds=batch_seeds)
        sampled_z = sampled_z.to(torch.float32).clone().contiguous()
        diff_time = time.time() - diff_start
        print(f"   - 디퓨전 샘플링 완료 (소요 시간: {diff_time:.2f}초)")

        print("🧹 디퓨전 모델 VRAM 점유 해제 중...")
        del model.diffusion_model
        gc.collect()

        print("🧬 2단계: VAE 디코딩 및 3D 공간 밀도 맵(Logits) 변환 중...")
        vae_start = time.time()
        decoded_latents = model.vae.decode_latents(sampled_z)
        context, _, _, _ = model.vae.decode_embed(decoded_latents)

        queries = create_grid_queries(args.density).to(device)
        logits = []
        chunk_size = 50000 
        for i in range(0, queries.size(1), chunk_size):
            chunk = queries[:, i:i+chunk_size]
            logit = model.vae.decode_queries(context, chunk)
            logits.append(logit)
        logits = torch.cat(logits, dim=1)
        vae_time = time.time() - vae_start
        print(f"   - VAE 디코딩 완료 (소요 시간: {vae_time:.2f}초)")

    print("🎨 3단계: Marching Cubes로 3D 메쉬 추출 중...")
    mc_start = time.time()
    mesh = occupancy_to_mesh(logits, r=args.density, threshold=args.threshold)
    mesh.export(args.output)
    mc_time = time.time() - mc_start
    print(f"   - 3D 메쉬 추출 및 저장 완료 (소요 시간: {mc_time:.2f}초)")

    total_infer_time = time.time() - total_infer_start
    
    print("\n" + "="*40)
    print(f"🎉 3D 에셋 생성 완료! 파일 저장됨: {args.output}")
    print(f"⏱️ 총 순수 생성 시간: {total_infer_time:.2f}초 (모델 로딩 제외)")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()