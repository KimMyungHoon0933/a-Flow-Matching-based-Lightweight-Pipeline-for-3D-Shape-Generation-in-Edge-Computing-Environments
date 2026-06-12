"""
benchmark_flow_only.py — Flow Matching 단독 벤치마크
===========================================================
사용법:
  python benchmark_flow_only.py \
    --fm_pth <Flow 체크포인트> \
    --vae_config <VAE config> \
    --vae_weights <VAE 가중치> \
    --runs 10
===========================================================
"""

import argparse
import torch
import torch.nn.functional
import numpy as np
import sys
import os
import gc
import time

# ==========================================
# 환경 변수 설정
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

# einsum 패치
def safe_einsum_linear(input, weight, bias=None):
    orig_shape = input.shape
    x = input.reshape(-1, orig_shape[-1]).float()
    w = weight.float()
    out = torch.einsum('xi,ji->xj', x, w)
    if bias is not None:
        out = out + bias.float()
    return out.reshape(*orig_shape[:-1], out.shape[-1])

torch.nn.functional.linear = safe_einsum_linear

sys.path.append(os.path.abspath('./CFM'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'external/pointops')))

from cod.utils.recon import create_grid_queries, occupancy_to_mesh

PROMPT_TO_ID = {
    "airplane": 0, "car": 16, "chair": 18, "table": 49,
    "watercraft": 53, "sofa": 47, "rifle": 44, "lamp": 28
}

def get_args():
    parser = argparse.ArgumentParser("Flow Matching 단독 벤치마크")
    parser.add_argument("--fm_pth", type=str, required=True)
    parser.add_argument("--vae_config", type=str, required=True)
    parser.add_argument("--vae_weights", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="chair",
                        help="생성할 카테고리 (기본: chair)")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--density", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4, help="ODE solver step 수")
    parser.add_argument("--method", type=str, default="midpoint",
                        help="ODE solver 방법 (euler, midpoint)")
    parser.add_argument("--save_mesh", action="store_true",
                        help="각 run마다 메쉬 파일 저장")
    return parser.parse_args()


def benchmark_flow(args):
    from CFM.flowmating_lightning import FlowMatchingLightning
    from flow_matching.solver.ode_solver import ODESolver

    device = torch.device('cpu')
    prompt_lower = args.prompt.lower()
    category_id = PROMPT_TO_ID[prompt_lower]

    # 모델 로드
    print("모델 로딩 중...")
    load_start = time.time()
    model = FlowMatchingLightning(
        vae_config_path=args.vae_config,
        vae_weights_path=args.vae_weights
    )
    ckpt = torch.load(args.fm_pth, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.to(device).eval().to(torch.float32)

    n_latents = model.flow_model.n_latents
    channels = model.flow_model.channels
    load_time = time.time() - load_start
    print(f"모델 로딩 완료 ({load_time:.2f}s)")
    print(f"Flow 설정: n_latents={n_latents}, channels={channels}, steps={args.steps}, method={args.method}")

    class FlowWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m
        def forward(self, x, t, **extras):
            if t.dim() == 0:
                t = t.repeat(x.shape[0])
            return self.model(x, t, class_labels=extras.get('class_labels'))

    cond = torch.tensor([category_id], dtype=torch.long, device=device)
    queries = create_grid_queries(args.density).to(device)

    sampling_times = []
    vae_times = []
    mc_times = []
    total_times = []

    for i in range(args.runs):
        gc.collect()
        total_start = time.time()

        with torch.no_grad():
            # 샘플링
            t0 = time.time()
            wrapped = FlowWrapper(model.flow_model)
            solver = ODESolver(velocity_model=wrapped)
            x_init = torch.randn([1, n_latents, channels], device=device)
            sampled_z = solver.sample(
                x_init=x_init,
                step_size=1.0 / args.steps,
                method=args.method,
                class_labels=cond
            )
            sampled_z = sampled_z.to(torch.float32).clone().contiguous()
            t_sampling = time.time() - t0

            # VAE 디코딩
            t0 = time.time()
            sampled_z = sampled_z.view(1, n_latents, channels)
            decoded = model.vae.decode_latents(sampled_z)
            context, _, _, _ = model.vae.decode_embed(decoded)
            logits = []
            for j in range(0, queries.size(1), 50000):
                chunk = queries[:, j:j+50000]
                logit = model.vae.decode_queries(context, chunk)
                logits.append(logit)
            logits = torch.cat(logits, dim=1)
            t_vae = time.time() - t0

        # Marching Cubes
        t0 = time.time()
        mesh = occupancy_to_mesh(logits, r=args.density, threshold=0.0)
        t_mc = time.time() - t0

        if args.save_mesh:
            mesh.export(f"flow_bench_{i+1:02d}.obj")

        t_total = time.time() - total_start

        sampling_times.append(t_sampling)
        vae_times.append(t_vae)
        mc_times.append(t_mc)
        total_times.append(t_total)

        del wrapped, solver
        print(f"  [{i+1}/{args.runs}] 샘플링={t_sampling:.2f}s  VAE={t_vae:.2f}s  MC={t_mc:.2f}s  전체={t_total:.2f}s")

    del model
    gc.collect()

    return {
        'sampling': sampling_times,
        'vae': vae_times,
        'mc': mc_times,
        'total': total_times,
    }


def print_results(results, args):
    def avg(lst):
        return sum(lst) / len(lst)

    def std(lst):
        m = avg(lst)
        return (sum((x - m) ** 2 for x in lst) / len(lst)) ** 0.5

    print("\n")
    print("=" * 55)
    print(f"Flow Matching 벤치마크 결과 ({args.runs}회 평균)")
    print(f"카테고리: {args.prompt} (ID={PROMPT_TO_ID[args.prompt.lower()]})")
    print(f"스텝: {args.steps}, 방법: {args.method}, 해상도: {args.density}")
    print("=" * 55)
    print(f"{'단계':<20} {'평균':<15} {'표준편차':<15}")
    print("-" * 55)

    for stage, name in [('sampling', 'Flow 샘플링'), ('vae', 'VAE 디코딩'), ('mc', 'Marching Cubes'), ('total', '전체')]:
        a = avg(results[stage])
        s = std(results[stage])
        if stage == 'total':
            print("-" * 55)
        print(f"{name:<20} {a:.2f}s           ±{s:.2f}s")

    print("=" * 55)

    # 파일 저장
    out_file = "benchmark_flow_results.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"Flow Matching 벤치마크 결과 ({args.runs}회 평균)\n")
        f.write(f"카테고리: {args.prompt} (ID={PROMPT_TO_ID[args.prompt.lower()]})\n")
        f.write(f"스텝: {args.steps}, 방법: {args.method}, 해상도: {args.density}\n")
        f.write("=" * 55 + "\n")
        f.write(f"{'단계':<20} {'평균':<15} {'표준편차':<15}\n")
        f.write("-" * 55 + "\n")

        for stage, name in [('sampling', 'Flow 샘플링'), ('vae', 'VAE 디코딩'), ('mc', 'Marching Cubes'), ('total', '전체')]:
            a = avg(results[stage])
            s = std(results[stage])
            if stage == 'total':
                f.write("-" * 55 + "\n")
            f.write(f"{name:<20} {a:.2f}s           ±{s:.2f}s\n")

        f.write("=" * 55 + "\n\n")

        f.write("개별 실행 결과:\n")
        for i, (s, v, m, t) in enumerate(zip(
            results['sampling'], results['vae'], results['mc'], results['total']
        )):
            f.write(f"  [{i+1}] 샘플링={s:.2f}s  VAE={v:.2f}s  MC={m:.2f}s  전체={t:.2f}s\n")

    print(f"\n결과가 {out_file}에 저장되었습니다.")


if __name__ == "__main__":
    args = get_args()

    prompt_lower = args.prompt.lower()
    if prompt_lower not in PROMPT_TO_ID:
        print(f"지원하지 않는 프롬프트: {prompt_lower}")
        print(f"지원 목록: {list(PROMPT_TO_ID.keys())}")
        sys.exit(1)

    print("=" * 55)
    print(f"Flow Matching 단독 벤치마크 ({args.runs}회)")
    print(f"카테고리: {args.prompt} (ID={PROMPT_TO_ID[prompt_lower]})")
    print(f"스텝: {args.steps}, 방법: {args.method}, 해상도: {args.density}")
    print("=" * 55)

    results = benchmark_flow(args)
    print_results(results, args)
