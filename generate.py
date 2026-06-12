import argparse
import torch
import numpy as np

# COD-VAE 및 우리가 만든 모듈 임포트
from duffusion.diffusion_lightning import CategoryConditionedDiffusion

# [수정] 정확한 폴더 경로를 반영한 임포트
from cod.utils.recon import create_grid_queries, occupancy_to_mesh

# ==========================================
# [맵핑 딕셔너리] 텍스트 -> ShapeNet 정수 ID 변환
# ==========================================
PROMPT_TO_ID = {
    "airplane": 0,
    "car": 16,
    "chair": 18,
    "table": 49,
    "watercraft": 53,
    "sofa": 47,
    "rifle": 44,
    "lamp": 28
}

def get_args():
    parser = argparse.ArgumentParser("Text-to-3D Generation Pipeline")
    parser.add_argument("--prompt", type=str, required=True, help="생성할 에셋 텍스트 (예: chair)")
    parser.add_argument("--diffusion_ckpt", type=str, required=True, help="학습된 디퓨전 모델의 .ckpt 파일 경로")
    parser.add_argument("--vae_config", type=str, required=True, help="VAE config 파일 경로")
    parser.add_argument("--vae_weights", type=str, required=True, help="VAE 가중치 파일 경로")
    parser.add_argument("--output", type=str, default="output.obj", help="저장할 3D 메쉬 파일명 (.obj)")
    parser.add_argument("--density", type=int, default=128, help="Marching Cubes 해상도 (기본: 128)")
    parser.add_argument("--threshold", type=float, default=0.0, help="메쉬 추출 밀도 임계값")
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 텍스트 프롬프트를 카테고리 ID로 변환
    prompt_lower = args.prompt.lower()
    if prompt_lower not in PROMPT_TO_ID:
        raise ValueError(f"지원하지 않는 프롬프트입니다: {prompt_lower}. 지원 목록: {list(PROMPT_TO_ID.keys())}")
    
    target_id = PROMPT_TO_ID[prompt_lower]
    cond_tensor = torch.tensor([target_id], dtype=torch.long, device=device)
    print(f"🎯 프롬프트 '{prompt_lower}' -> 카테고리 ID [{target_id}] 변환 완료")

    # 2. 통합 모델 로드
    print("✅ 체크포인트 로딩 중...")
    model = CategoryConditionedDiffusion.load_from_checkpoint(
        checkpoint_path=args.diffusion_ckpt,
        vae_config_path=args.vae_config,
        vae_weights_path=args.vae_weights
    )
    model.to(device)
    model.eval()

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            # 3. 디퓨전 모델로 잠재 벡터(z) 샘플링
            print("⏳ 디퓨전 모델 샘플링 진행 중 (EDM Sampler)...")
            batch_seeds = torch.randint(0, 1000000, (1,)).to(device)
            # sampled_z 형태: [1, 32, channels]
            sampled_z = model.diffusion_model.sample(cond=cond_tensor, batch_seeds=batch_seeds).float()

            # 4. VAE 디코더 역추적 (Latent -> Embed -> Context)
            print("🧬 VAE 디코딩 및 3D 공간 밀도 맵(Logits) 변환 중...")
            decoded_latents = model.vae.decode_latents(sampled_z)
            context, _, _, _ = model.vae.decode_embed(decoded_latents)

            # 5. 3D Query 그리드 생성 및 청크(Chunk) 단위 밀도 예측
            queries = create_grid_queries(args.density).to(device) # [1, N, 3]
            
            logits = []
            chunk_size = 100000 # OOM 방지를 위한 분할 계산
            for i in range(0, queries.size(1), chunk_size):
                chunk = queries[:, i:i+chunk_size]
                logit = model.vae.decode_queries(context, chunk)
                logits.append(logit)
            
            logits = torch.cat(logits, dim=1) # [1, N]

    # 6. Marching Cubes로 표면 추출 및 파일 저장
    print("🎨 Marching Cubes로 3D 메쉬 추출 중...")
    mesh = occupancy_to_mesh(logits, r=args.density, threshold=args.threshold)
    mesh.export(args.output)
    
    print(f"🎉 3D 에셋 생성 완료! 파일 저장됨: {args.output}")

if __name__ == "__main__":
    main()