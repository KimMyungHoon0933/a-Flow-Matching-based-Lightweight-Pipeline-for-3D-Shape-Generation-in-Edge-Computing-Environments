"""
benchmark_flow_resource.py — Flow Matching 리소스 사용량 측정
===========================================================
측정 항목:
  - 모델 크기 (파라미터 수, 메모리)
  - 최대 RAM 사용량 (추론 중)
  - 최대 CPU 점유율 (추론 중)
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
import threading
import resource

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


import subprocess


def get_rss_mb():
    """현재 프로세스의 RSS(Resident Set Size)를 MB 단위로 반환"""
    with open('/proc/self/status', 'r') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024  # kB -> MB
    return 0.0


class ResourceMonitor:
    """RSS는 스레드로 샘플링, CPU 사용률은 mpstat로 측정
    
    mpstat 1초 간격으로 백그라운드 실행 → %idle 파싱 → 사용률 = 100 - idle
    커널 레벨 측정이라 100%를 절대 초과하지 않음
    """
    def __init__(self, interval=0.1):
        self.interval = interval
        self.peak_rss_mb = 0.0
        self._stop = threading.Event()
        self._thread = None
        self._mpstat_proc = None
        self.cpu_samples = []

    def start(self):
        self.peak_rss_mb = get_rss_mb()

        # mpstat 백그라운드 실행 (1초 간격, 종료될 때까지)
        self._mpstat_proc = subprocess.Popen(
            ['mpstat', '1'],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )

        # mpstat 출력 파싱 스레드
        self._mpstat_thread = threading.Thread(target=self._parse_mpstat, daemon=True)
        self._mpstat_thread.start()

        # RSS 모니터링 스레드
        self._stop.clear()
        self._thread = threading.Thread(target=self._monitor_rss, daemon=True)
        self._thread.start()

    def _parse_mpstat(self):
        """mpstat 출력에서 %idle을 읽어 CPU 사용률 계산"""
        for line in self._mpstat_proc.stdout:
            line = line.strip()
            if not line or line.startswith('Linux') or line.startswith('Average'):
                continue
            # 헤더 행에서 %idle 컬럼 위치 찾기
            if '%idle' in line:
                parts = line.split()
                try:
                    self._idle_idx = parts.index('%idle')
                except ValueError:
                    pass
                continue
            # 데이터 행: "HH:MM:SS  all  ..."
            parts = line.split()
            if len(parts) > 2 and 'all' in parts and hasattr(self, '_idle_idx'):
                try:
                    # 'all' 다음부터 데이터이므로, 실제 데이터 부분만 추출
                    # mpstat 출력: 시간 AM/PM all usr nice sys iowait irq soft steal guest gnice idle
                    idle = float(parts[-1])  # %idle은 항상 마지막 컬럼
                    cpu_used = 100.0 - idle
                    self.cpu_samples.append(cpu_used)
                except (ValueError, IndexError):
                    pass

    def _monitor_rss(self):
        """RSS만 주기적으로 샘플링"""
        while not self._stop.is_set():
            rss = get_rss_mb()
            if rss > self.peak_rss_mb:
                self.peak_rss_mb = rss
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._mpstat_proc:
            self._mpstat_proc.terminate()
            self._mpstat_proc.wait()
        if self._thread:
            self._thread.join()

    @property
    def max_cpu_percent(self):
        return max(self.cpu_samples) if self.cpu_samples else 0.0

    @property
    def avg_cpu_percent(self):
        return sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0


def count_parameters(model):
    """모델 파라미터 수와 메모리 크기 계산"""
    total = 0
    total_bytes = 0
    for p in model.parameters():
        total += p.numel()
        total_bytes += p.numel() * p.element_size()
    return total, total_bytes


def get_args():
    parser = argparse.ArgumentParser("Flow Matching 리소스 사용량 측정")
    parser.add_argument("--fm_pth", type=str, required=True)
    parser.add_argument("--vae_config", type=str, required=True)
    parser.add_argument("--vae_weights", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="chair")
    parser.add_argument("--density", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--method", type=str, default="euler")
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device('cpu')
    prompt_lower = args.prompt.lower()

    if prompt_lower not in PROMPT_TO_ID:
        print(f"지원하지 않는 프롬프트: {prompt_lower}")
        print(f"지원 목록: {list(PROMPT_TO_ID.keys())}")
        sys.exit(1)

    category_id = PROMPT_TO_ID[prompt_lower]

    print("=" * 60)
    print("Flow Matching 리소스 사용량 측정")
    print(f"카테고리: {args.prompt} (ID={category_id})")
    print(f"스텝: {args.steps}, 방법: {args.method}, 해상도: {args.density}")
    print("=" * 60)

    # ── 1. 모델 로드 전 기준 메모리 ──
    gc.collect()
    rss_before_load = get_rss_mb()
    print(f"\n모델 로드 전 RSS: {rss_before_load:.1f} MB")

    # ── 2. 모델 로드 ──
    print("모델 로딩 중...")
    from CFM.flowmating_lightning import FlowMatchingLightning
    from flow_matching.solver.ode_solver import ODESolver

    model = FlowMatchingLightning(
        vae_config_path=args.vae_config,
        vae_weights_path=args.vae_weights
    )
    ckpt = torch.load(args.fm_pth, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    del ckpt
    gc.collect()
    model.to(device).eval().to(torch.float32)

    rss_after_load = get_rss_mb()

    n_latents = model.flow_model.n_latents
    channels = model.flow_model.channels

    # ── 3. 모델 크기 분석 ──
    flow_params, flow_bytes = count_parameters(model.flow_model)
    vae_params, vae_bytes = count_parameters(model.vae)
    total_params = flow_params + vae_params
    total_bytes = flow_bytes + vae_bytes

    print("\n" + "-" * 60)
    print("모델 크기")
    print("-" * 60)
    print(f"  Flow 모델:  {flow_params:>12,} params  ({flow_bytes / 1e6:>7.2f} MB)")
    print(f"  VAE:        {vae_params:>12,} params  ({vae_bytes / 1e6:>7.2f} MB)")
    print(f"  합계:       {total_params:>12,} params  ({total_bytes / 1e6:>7.2f} MB)")
    print(f"  체크포인트:  {os.path.getsize(args.fm_pth) / 1e6:.2f} MB ({args.fm_pth})")
    print(f"  모델 로드 후 RSS: {rss_after_load:.1f} MB (증가분: +{rss_after_load - rss_before_load:.1f} MB)")

    # ── 4. 추론 중 리소스 모니터링 ──
    print("\n" + "-" * 60)
    print("추론 실행 중 (리소스 모니터링)...")
    print("-" * 60)

    class FlowWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m
        def forward(self, x, t, **extras):
            if t.dim() == 0:
                t = t.repeat(x.shape[0])
            return self.model(x, t, class_labels=extras.get('class_labels'))

    cond_tensor = torch.tensor([category_id], dtype=torch.long, device=device)

    # ODE Solver 세팅
    wrapped_model = FlowWrapper(model.flow_model)
    solver = ODESolver(velocity_model=wrapped_model)
    step_size = 1.0 / args.steps

    monitor = ResourceMonitor(interval=0.05)
    monitor.start()

    with torch.no_grad():
        # 1단계: Flow Matching 샘플링
        x_init = torch.randn([1, n_latents, channels], device=device)
        sampled_z = solver.sample(
            x_init=x_init,
            step_size=step_size,
            method='euler',
            class_labels=cond_tensor
        )
        sampled_z = sampled_z.to(torch.float32).clone().contiguous()

        rss_after_flow = get_rss_mb()

        # Flow 모델 메모리 해제
        del model.flow_model, wrapped_model, solver
        gc.collect()

        # 2단계: VAE 디코딩
        sampled_z = sampled_z.view(sampled_z.shape[0], n_latents, channels)
        decoded_latents = model.vae.decode_latents(sampled_z)
        context, _, _, _ = model.vae.decode_embed(decoded_latents)

        queries = create_grid_queries(args.density).to(device)
        logits = []
        chunk_size = 50000
        for j in range(0, queries.size(1), chunk_size):
            chunk = queries[:, j:j + chunk_size]
            logit = model.vae.decode_queries(context, chunk)
            logits.append(logit)
        logits = torch.cat(logits, dim=1)

        rss_after_vae = get_rss_mb()

    # 3단계: Marching Cubes
    mesh = occupancy_to_mesh(logits, r=args.density, threshold=0.0)
    rss_after_mc = get_rss_mb()

    monitor.stop()

    # ── 5. 결과 출력 ──
    peak_rss = monitor.peak_rss_mb
    max_cpu = monitor.max_cpu_percent
    avg_cpu = monitor.avg_cpu_percent

    # rusage로도 peak RSS 확인 (크로스체크)
    rusage_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # kB -> MB

    print("\n" + "=" * 60)
    print("리소스 사용량 결과")
    print("=" * 60)

    print(f"\n{'항목':<30} {'값':>20}")
    print("-" * 60)
    print(f"{'[모델 크기]':<30}")
    print(f"  {'Flow 파라미터 수':<28} {flow_params:>15,}")
    print(f"  {'Flow 파라미터 메모리':<28} {flow_bytes / 1e6:>14.2f} MB")
    print(f"  {'VAE 파라미터 수':<28} {vae_params:>15,}")
    print(f"  {'VAE 파라미터 메모리':<28} {vae_bytes / 1e6:>14.2f} MB")
    print(f"  {'전체 파라미터 수':<28} {total_params:>15,}")
    print(f"  {'전체 파라미터 메모리':<28} {total_bytes / 1e6:>14.2f} MB")
    print("-" * 60)
    print(f"{'[메모리 (RSS)]':<30}")
    print(f"  {'모델 로드 전':<28} {rss_before_load:>14.1f} MB")
    print(f"  {'모델 로드 후':<28} {rss_after_load:>14.1f} MB")
    print(f"  {'Flow 샘플링 후':<28} {rss_after_flow:>14.1f} MB")
    print(f"  {'VAE 디코딩 후':<28} {rss_after_vae:>14.1f} MB")
    print(f"  {'Marching Cubes 후':<28} {rss_after_mc:>14.1f} MB")
    print(f"  {'최대 RSS (모니터링)':<28} {peak_rss:>14.1f} MB")
    print(f"  {'최대 RSS (rusage)':<28} {rusage_peak:>14.1f} MB")
    print("-" * 60)
    print(f"{'[CPU 점유율]':<30} (최대 100%)")
    print(f"  {'최대 CPU 사용률':<28} {max_cpu:>13.1f} %")
    print(f"  {'평균 CPU 사용률':<28} {avg_cpu:>13.1f} %")
    print("=" * 60)

    # 파일 저장
    out_file = "benchmark_flow_resource.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("Flow Matching 리소스 사용량 결과\n")
        f.write(f"카테고리: {args.prompt} (ID={category_id})\n")
        f.write(f"스텝: {args.steps}, 방법: {args.method}, 해상도: {args.density}\n")
        f.write("=" * 60 + "\n\n")

        f.write("[모델 크기]\n")
        f.write(f"  Flow 파라미터:  {flow_params:,} ({flow_bytes / 1e6:.2f} MB)\n")
        f.write(f"  VAE 파라미터:   {vae_params:,} ({vae_bytes / 1e6:.2f} MB)\n")
        f.write(f"  전체 파라미터:  {total_params:,} ({total_bytes / 1e6:.2f} MB)\n")
        f.write(f"  체크포인트:     {os.path.getsize(args.fm_pth) / 1e6:.2f} MB\n\n")

        f.write("[메모리 사용량 (RSS)]\n")
        f.write(f"  모델 로드 전:     {rss_before_load:.1f} MB\n")
        f.write(f"  모델 로드 후:     {rss_after_load:.1f} MB\n")
        f.write(f"  Flow 샘플링 후:   {rss_after_flow:.1f} MB\n")
        f.write(f"  VAE 디코딩 후:    {rss_after_vae:.1f} MB\n")
        f.write(f"  Marching Cubes 후: {rss_after_mc:.1f} MB\n")
        f.write(f"  최대 RSS:         {max(peak_rss, rusage_peak):.1f} MB\n\n")

        f.write(f"[CPU 점유율] (최대 100%)\n")
        f.write(f"  최대: {max_cpu:.1f}%\n")
        f.write(f"  평균: {avg_cpu:.1f}%\n")

    print(f"\n결과가 {out_file}에 저장되었습니다.")


if __name__ == "__main__":
    main()