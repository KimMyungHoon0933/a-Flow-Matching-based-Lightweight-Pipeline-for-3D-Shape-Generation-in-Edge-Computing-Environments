"""
sample_class_cond.py — 수정 버전
========================================
변경 사항:
  1. 초기 노이즈 shape을 [iters, 512, 16] 하드코딩 → 모델에서 동적으로 가져옴
  2. 출력 디렉토리 자동 생성
  3. 팩토리 함수 이름 호환 (cod_m32_d32_flow)
========================================
"""

import argparse
import numpy as np
import torch
import mcubes
import trimesh
from pathlib import Path

import models_class_cond, models_ae

# Meta 라이브러리 임포트
from flow_matching.ode_solver import ODESolver
from flow_matching.model_wrapper import ModelWrapper


# ODE Solver가 모델을 인식할 수 있도록 감싸는 Wrapper 클래스
class FlowWrapper(ModelWrapper):
    def forward(self, x, t, **extras):
        # t가 스칼라로 넘어올 경우 배치 사이즈에 맞춰 확장
        if t.dim() == 0:
            t = t.repeat(x.shape[0])
        return self.model(x, t, class_labels=extras.get('class_labels'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Flow Matching Sampling')
    parser.add_argument('--ae', type=str, required=True,
                        help='VAE 모델 이름 (예: kl_d512_m512_l8)')
    parser.add_argument('--ae-pth', type=str, required=True,
                        help='VAE 체크포인트 경로')
    parser.add_argument('--fm', type=str, required=True,
                        help='Flow 모델 이름 (예: cod_m32_d32_flow)')
    parser.add_argument('--fm-pth', type=str, required=True,
                        help='Flow 모델 체크포인트 경로')
    parser.add_argument('--steps', type=int, default=4,
                        help='ODE solver 스텝 수 (기본: 4)')
    parser.add_argument('--category', type=int, default=18,
                        help='생성할 카테고리 ID (기본: 18=chair)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='생성할 샘플 수 (기본: 10)')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ── VAE 로드 ──
    print(f"[1/4] VAE 로드: {args.ae}")
    ae = models_ae.__dict__[args.ae]()
    ae.load_state_dict(torch.load(args.ae_pth, map_location='cpu')['model'])
    ae.to(device).eval()

    # ── Flow 모델 로드 ──
    print(f"[2/4] Flow 모델 로드: {args.fm}")
    model = models_class_cond.__dict__[args.fm]()
    
    # 체크포인트에서 state_dict 추출 (Lightning 체크포인트 호환)
    ckpt = torch.load(args.fm_pth, map_location='cpu')
    if 'state_dict' in ckpt:
        # Lightning 체크포인트: 'flow_model.' 접두사 제거
        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            if k.startswith('flow_model.'):
                state_dict[k.replace('flow_model.', '', 1)] = v
        model.load_state_dict(state_dict)
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    
    model.to(device).eval()

    # [핵심] 모델에서 n_latents, channels를 가져옴 → 하드코딩 제거
    n_latents = model.n_latents
    channels = model.channels
    print(f"    모델 설정: n_latents={n_latents}, channels={channels}")

    # ── ODE Solver 세팅 ──
    wrapped_model = FlowWrapper(model)
    solver = ODESolver(velocity_model=wrapped_model)

    # ── 3D 그리드 생성 ──
    density = 128
    gap = 2. / density
    x_lin = np.linspace(-1, 1, density + 1)
    y_lin = np.linspace(-1, 1, density + 1)
    z_lin = np.linspace(-1, 1, density + 1)
    xv, yv, zv = np.meshgrid(x_lin, y_lin, z_lin)
    grid = torch.from_numpy(
        np.stack([xv, yv, zv]).astype(np.float32)
    ).view(3, -1).transpose(0, 1)[None].to(device)

    # ── 출력 디렉토리 생성 ──
    output_dir = Path(f"class_cond_obj/{args.fm}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 생성 ──
    step_size = 1.0 / args.steps  # 4스텝이면 0.25
    category_id = args.category
    num_samples = args.num_samples

    print(f"[3/4] 생성 시작: category={category_id}, samples={num_samples}, steps={args.steps}")

    with torch.no_grad():
        # 배치 단위로 생성
        batch_size = min(num_samples, 16)  # 메모리 고려

        for start_idx in range(0, num_samples, batch_size):
            current_batch = min(batch_size, num_samples - start_idx)

            # [수정] 모델에서 가져온 n_latents, channels 사용
            x_init = torch.randn([current_batch, n_latents, channels], device=device)
            labels = torch.full((current_batch,), category_id, dtype=torch.long, device=device)

            # Flow Matching ODE solver 추론
            sampled_array = solver.sample(
                x_init=x_init,
                step_size=step_size,
                method='euler',
                class_labels=labels
            )

            # VAE 디코딩 + Marching Cubes
            for j in range(sampled_array.shape[0]):
                logits = ae.decode(sampled_array[j:j + 1], grid).detach()
                volume = logits.view(density + 1, density + 1, density + 1).permute(1, 0, 2).cpu().numpy()
                verts, faces = mcubes.marching_cubes(volume, 0)

                verts *= gap
                verts -= 1.

                m = trimesh.Trimesh(verts, faces)
                sample_idx = start_idx + j
                out_path = output_dir / f"{category_id:02d}_{sample_idx:05d}.obj"
                m.export(str(out_path))
                print(f"    저장: {out_path}")

    print(f"[4/4] 완료! {num_samples}개 생성됨 → {output_dir}/")