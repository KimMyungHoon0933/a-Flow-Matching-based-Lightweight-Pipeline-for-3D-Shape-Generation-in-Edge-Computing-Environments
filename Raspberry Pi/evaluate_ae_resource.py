"""
evaluate_ae_resource.py — VAE 평가 + 리소스 사용량 측정
===========================================================
evaluate_ae.py 평가 과정에서의 리소스(메모리, CPU 점유율) 측정

사용법:
  python evaluate_ae_resource.py <model_dir> \
    --data shapenet \
    --seed 123456
===========================================================
"""

import os
from os import path
from argparse import ArgumentParser
import sys
import gc
import time
import threading
import subprocess
import resource

# ==========================================
# 환경 변수 설정
# ==========================================
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKLDNN_DISABLED'] = '1'

import lightning.pytorch as pl
import torch
import torch.nn.functional

torch.backends.mkldnn.enabled = False
if hasattr(torch.backends, 'xnnpack'):
    torch.backends.xnnpack.enabled = False
torch.set_num_threads(1)
torch.set_default_dtype(torch.float32)

# ==========================================
# einsum 패치
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

sys.path.append('./external/pointops')
from pointops.functions import pointops
import engine
from cod.solvers.recon_eval import ReconEvaluator

DEFAULT_OUTPUT_DIR = 'results/'

# ==========================================
# 리소스 모니터링 (benchmark_flow_resource.py 동일)
# ==========================================

def get_rss_mb():
    with open('/proc/self/status', 'r') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    return 0.0


def count_parameters(model):
    total = 0
    total_bytes = 0
    for p in model.parameters():
        total += p.numel()
        total_bytes += p.numel() * p.element_size()
    return total, total_bytes


class ResourceMonitor:
    """RSS는 스레드로 샘플링, CPU 사용률은 mpstat로 측정"""
    def __init__(self, interval=0.1):
        self.interval = interval
        self.peak_rss_mb = 0.0
        self._stop = threading.Event()
        self._thread = None
        self._mpstat_proc = None
        self.cpu_samples = []

    def start(self):
        self.peak_rss_mb = get_rss_mb()

        self._mpstat_proc = subprocess.Popen(
            ['mpstat', '1'],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )

        self._mpstat_thread = threading.Thread(target=self._parse_mpstat, daemon=True)
        self._mpstat_thread.start()

        self._stop.clear()
        self._thread = threading.Thread(target=self._monitor_rss, daemon=True)
        self._thread.start()

    def _parse_mpstat(self):
        for line in self._mpstat_proc.stdout:
            line = line.strip()
            if not line or line.startswith('Linux') or line.startswith('Average'):
                continue
            if '%idle' in line:
                parts = line.split()
                try:
                    self._idle_idx = parts.index('%idle')
                except ValueError:
                    pass
                continue
            parts = line.split()
            if len(parts) > 2 and 'all' in parts and hasattr(self, '_idle_idx'):
                try:
                    idle = float(parts[-1])
                    cpu_used = 100.0 - idle
                    self.cpu_samples.append(cpu_used)
                except (ValueError, IndexError):
                    pass

    def _monitor_rss(self):
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


# ==========================================
# 메인
# ==========================================

parser = ArgumentParser('VAE 평가 + 리소스 측정')
parser.add_argument('model_dir', type=str, help='path to the saved weights dir')
parser.add_argument('--save_dir', '-sd', type=str, default=None, help='path to the output dir')
parser.add_argument('--data', '-d', type=str, default=None, help='name of the data config (e.g., shapenet)')
parser.add_argument('--eval', '-e', type=str, default=None, help='name of the evaluator config')
parser.add_argument('--seed', '-s', type=int, default=123456, help='evaluation random seed')
parser.add_argument('--num_samples', '-n', type=int, default=10,
                    help='평가에 사용할 샘플 수 (기본: 10)')
parser.add_argument('--gpus', '-g', default='[0]',
                    help='GPU to use')

def main():
    args = parser.parse_args()
    pl.seed_everything(args.seed)

    print("=" * 60)
    print("VAE 평가 + 리소스 사용량 측정")
    print(f"모델 디렉토리: {args.model_dir}")
    print(f"평가 샘플 수: {args.num_samples}")
    print("=" * 60)

    # ── 1. 모델 로드 전 메모리 ──
    gc.collect()
    rss_before_load = get_rss_mb()
    print(f"\n모델 로드 전 RSS: {rss_before_load:.1f} MB")

    # ── 2. 모델 로드 (evaluate_ae.py 동일) ──
    print("모델 로딩 중...")
    load_start = time.time()

    cfg = engine.load_config(path.join(args.model_dir, 'config.yaml'))
    pt_files = [x for x in os.listdir(args.model_dir) if x.endswith('.pt')]
    output_dir = None
    if len(pt_files) > 0:
        checkpoint_path = path.join(args.model_dir, pt_files[0])
    else:
        checkpoint_path = engine.find_best_checkpoint_path(path.join(args.model_dir, 'checkpoints'))
        output_dir = args.model_dir

    model_name = args.model_dir.strip('/').split('/')[-1]
    if args.save_dir is not None:
        output_dir = path.join(args.save_dir, model_name)
    if output_dir is None:
        output_dir = path.join(DEFAULT_OUTPUT_DIR, model_name)

    engine.set_context_from_existing(output_dir)

    from cod.models.vae.vae import CompactLatentVAE
    vae_params_cfg = cfg.model.get('params', cfg.model)
    filtered_params = {k: v for k, v in vae_params_cfg.items() if k not in ['_target_', '_base_', '_overwrite_']}
    model = CompactLatentVAE(**filtered_params)

    if args.data is not None:
        data_cfg = engine.load_config(path.join('config/data', f'{args.data}.yaml'))
    elif 'data' in cfg:
        data_cfg = cfg.data
    else:
        raise Exception('data config should be specified either from command line or config file')
    dm = engine.instantiate(data_cfg)

    eval_cfg = {}
    if args.eval is not None and path.exists(args.eval):
        eval_cfg = engine.load_config(args.eval)
    evaluator = engine.instantiate(eval_cfg, ReconEvaluator, dm=dm, model=model)
    evaluator.num_cd_workers = 4  # 라즈베리파이 OOM 방지
    evaluator.restore_checkpoint(checkpoint_path)

    load_time = time.time() - load_start
    rss_after_load = get_rss_mb()

    print(f"모델 로딩 완료 ({load_time:.2f}s)")

    # ── 3. 모델 크기 분석 ──
    vae_params, vae_bytes = count_parameters(model)
    ckpt_size = os.path.getsize(checkpoint_path) / 1e6

    print("\n" + "-" * 60)
    print("모델 크기")
    print("-" * 60)
    print(f"  VAE:        {vae_params:>12,} params  ({vae_bytes / 1e6:>7.2f} MB)")
    print(f"  체크포인트:  {ckpt_size:.2f} MB ({checkpoint_path})")
    print(f"  모델 로드 후 RSS: {rss_after_load:.1f} MB (증가분: +{rss_after_load - rss_before_load:.1f} MB)")

    # ── 4. 평가 실행 + 리소스 모니터링 ──
    print("\n" + "-" * 60)
    print("평가 실행 중 (리소스 모니터링)...")
    print("-" * 60)

    monitor = ResourceMonitor(interval=0.05)
    monitor.start()

    eval_start = time.time()

    n = args.num_samples

    # 데이터셋을 num_samples개로 제한
    test_dataset = dm.get_dataset('test')
    if hasattr(test_dataset, 'items'):
        test_dataset.items = test_dataset.items[:n]
    print(f"테스트 데이터: {len(test_dataset)}개 사용")

    # trainer.test (제한된 데이터셋으로 직접 DataLoader 생성)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    trainer = pl.Trainer(accelerator='cpu', devices=1)
    trainer.test(model=evaluator, dataloaders=test_loader)
    rss_after_test = get_rss_mb()

    # CD metric (같은 제한된 데이터셋 재사용)
    test_dataset.use_queries = False
    test_dataset.use_full_surface = True
    evaluator.measure_cd(test_dataset)
    rss_after_cd = get_rss_mb()

    eval_time = time.time() - eval_start

    monitor.stop()

    # ── 5. 결과 출력 ──
    peak_rss = monitor.peak_rss_mb
    max_cpu = monitor.max_cpu_percent
    avg_cpu = monitor.avg_cpu_percent
    rusage_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    print("\n" + "=" * 60)
    print("리소스 사용량 결과")
    print("=" * 60)

    print(f"\n{'항목':<30} {'값':>20}")
    print("-" * 60)
    print(f"{'[모델 크기]':<30}")
    print(f"  {'VAE 파라미터 수':<28} {vae_params:>15,}")
    print(f"  {'VAE 파라미터 메모리':<28} {vae_bytes / 1e6:>14.2f} MB")
    print(f"  {'체크포인트 크기':<28} {ckpt_size:>14.2f} MB")
    print("-" * 60)
    print(f"{'[메모리 (RSS)]':<30}")
    print(f"  {'모델 로드 전':<28} {rss_before_load:>14.1f} MB")
    print(f"  {'모델 로드 후':<28} {rss_after_load:>14.1f} MB")
    print(f"  {'trainer.test 후':<28} {rss_after_test:>14.1f} MB")
    print(f"  {'CD 측정 후':<28} {rss_after_cd:>14.1f} MB")
    print(f"  {'최대 RSS (모니터링)':<28} {peak_rss:>14.1f} MB")
    print(f"  {'최대 RSS (rusage)':<28} {rusage_peak:>14.1f} MB")
    print("-" * 60)
    print(f"{'[CPU 점유율]':<30} (최대 100%)")
    print(f"  {'최대 CPU 사용률':<28} {max_cpu:>13.1f} %")
    print(f"  {'평균 CPU 사용률':<28} {avg_cpu:>13.1f} %")
    print("-" * 60)
    print(f"{'[소요 시간]':<30}")
    print(f"  {'모델 로딩':<28} {load_time:>13.2f} s")
    print(f"  {'평가 (test + CD)':<28} {eval_time:>13.2f} s")
    print("=" * 60)

    # 파일 저장
    out_file = "evaluate_ae_resource.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("VAE 평가 리소스 사용량 결과\n")
        f.write(f"모델: {args.model_dir}\n")
        f.write(f"체크포인트: {checkpoint_path}\n")
        f.write("=" * 60 + "\n\n")

        f.write("[모델 크기]\n")
        f.write(f"  VAE 파라미터:  {vae_params:,} ({vae_bytes / 1e6:.2f} MB)\n")
        f.write(f"  체크포인트:    {ckpt_size:.2f} MB\n\n")

        f.write("[메모리 사용량 (RSS)]\n")
        f.write(f"  모델 로드 전:     {rss_before_load:.1f} MB\n")
        f.write(f"  모델 로드 후:     {rss_after_load:.1f} MB\n")
        f.write(f"  trainer.test 후:  {rss_after_test:.1f} MB\n")
        f.write(f"  CD 측정 후:       {rss_after_cd:.1f} MB\n")
        f.write(f"  최대 RSS:         {max(peak_rss, rusage_peak):.1f} MB\n\n")

        f.write(f"[CPU 점유율] (최대 100%)\n")
        f.write(f"  최대: {max_cpu:.1f}%\n")
        f.write(f"  평균: {avg_cpu:.1f}%\n\n")

        f.write("[소요 시간]\n")
        f.write(f"  모델 로딩:       {load_time:.2f}s\n")
        f.write(f"  평가 (test+CD):  {eval_time:.2f}s\n")

    print(f"\n결과가 {out_file}에 저장되었습니다.")


if __name__ == '__main__':
    main()
