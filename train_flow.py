import sys
import os
# 파이썬아, 모듈을 찾을 때 'CFM' 폴더 안쪽도 루트 경로처럼 취급해서 뒤져봐!
sys.path.append(os.path.abspath('./CFM'))
import torch
import argparse
import lightning.pytorch as pl
# COD-VAE 자체 엔진 (YAML 파싱 및 DataModule 인스턴스화 용도)
import engine 

# [수정됨] 기존 디퓨전 대신 새로 작성한 Flow Matching Lightning 모듈 임포트
from CFM.flowmating_lightning import FlowMatchingLightning
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

def get_args_parser():
    parser = argparse.ArgumentParser('Category-Conditioned Flow Matching Training (OT-CFM)')
    
    parser.add_argument('--data_config', type=str, default='config/data/shapenet.yaml', 
                        help='Path to the data config yaml file')
    
    parser.add_argument('--vae_config', type=str, required=True, 
                        help='Path to VAE config (e.g., /root/re/COD-VAE/vae_m32 copy/config.yaml)')
    parser.add_argument('--vae_weights', type=str, required=True, 
                        help='Path to VAE weights (e.g., /root/re/COD-VAE/vae_m32 copy/weights.pt)')
    
    # [추가] 이전 체크포인트에서 이어서 학습
    parser.add_argument('--resume', type=str, default=None,
                        help='이전 체크포인트 경로 (예: ./checkpoints_flow/last.ckpt)')
    
    parser.add_argument('--gpus', '-g', default='[0]', help='GPU to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for Flow Matching training')
    parser.add_argument('--epochs', type=int, default=2000, help='Total epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    return parser.parse_args()

def main():
    args = get_args_parser()
    pl.seed_everything(123456) 

    print("✅ 1. 데이터 모듈(DataModule) 로딩 중...")
    data_cfg = engine.load_config(args.data_config)
    data_cfg.batch_size = args.batch_size 
    dm = engine.instantiate(data_cfg)

    print("✅ 2. 동결된 VAE 및 Flow Matching 모델(Vector Field Predictor) 초기화 중...")
    # 새로 만든 OT-CFM 기반 Lightning 모듈 호출
    model = FlowMatchingLightning(
        vae_config_path=args.vae_config,
        vae_weights_path=args.vae_weights,
        lr=args.lr
    )

    print("✅ 3. PyTorch Lightning Trainer (Flow Matching 최적화) 세팅 중...")
    # 체크포인트 저장 경로 및 이름 변경 (diffusion -> flow)
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints_flow',
        filename='flow-{epoch:04d}-{train_loss:.4f}',
        save_last=True,
        every_n_epochs=30,
        save_weights_only=True
    )

    trainer = pl.Trainer(
        devices=engine.parse_gpus_str(args.gpus),
        accelerator="gpu",
        strategy="ddp" if len(engine.parse_gpus_str(args.gpus)) > 1 else "auto",
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        # A100/H200 GPU 성능 극대화를 위한 Mixed Precision 적용
        precision="16-mixed" 
    )

    if args.resume:
        print(f"📂 체크포인트에서 가중치 로드: {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print("   ✅ 가중치 로드 완료 (옵티마이저 상태는 초기화)")

    trainer.fit(model, datamodule=dm)  # ckpt_path 없이 실행

if __name__ == "__main__":
    main()