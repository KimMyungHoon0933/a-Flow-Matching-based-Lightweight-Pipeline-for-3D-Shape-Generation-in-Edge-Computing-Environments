import argparse
import numpy as np
import torch
import mcubes
import trimesh
import sys
import os
from pathlib import Path

# Meta AI 라이브러리(flow_matching) 모듈 탐색을 위한 경로 강제 주입
sys.path.append(os.path.abspath('./CFM'))

# 명시된 경로(/root/re/COD-VAE/CFM/flow_set/)에 맞춘 정확한 임포트
from CFM.flow_set import models_ae
from CFM.flow_set import models_class_cond 

# Meta AI의 Flow Matching ODE 솔버
from flow_matching.solver.ode_solver import ODESolver

# 모델 입출력 규격을 솔버에 맞추기 위한 래퍼 클래스
class FlowWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t, **extras):
        # t가 스칼라로 들어올 경우 배치 사이즈(Batch Size)만큼 확장
        if t.dim() == 0:
            t = t.repeat(x.shape[0])
        return self.model(x, t, class_labels=extras.get('class_labels'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ae', type=str, required=True) 
    parser.add_argument('--ae-pth', type=str, required=True) 
    parser.add_argument('--fm', type=str, required=True) 
    parser.add_argument('--fm-pth', type=str, required=True) 
    args = parser.parse_args()

    device = torch.device('cuda:0')

    print("✅ 1. 지정된 경로(CFM.flow_set)에서 VAE 모델 로딩 중...")
    ae = models_ae.__dict__[args.ae]()
    ae.load_state_dict(torch.load(args.ae_pth, map_location='cpu')['model'])
    ae.to(device)
    ae.eval()

    # 3D 공간 좌표계 세팅
    density = 128
    gap = 2. / density
    x = np.linspace(-1, 1, density+1)
    y = np.linspace(-1, 1, density+1)
    z = np.linspace(-1, 1, density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].to(device)

    print("✅ 2. 지정된 경로(CFM.flow_set)에서 Flow 모델 및 솔버 초기화 중...")
    model = models_class_cond.__dict__[args.fm]()
    
    # PyTorch Lightning 체크포인트 로드 (접두사 'flow_model.' 제거 처리)
    ckpt = torch.load(args.fm_pth, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
    state_dict = {k.replace('flow_model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # ODE 솔버 세팅
    wrapped_model = FlowWrapper(model)
    solver = ODESolver(velocity_model=wrapped_model)

    total = 1000
    iters = 100 
    
    # 채널 수 동적 획득 (에러 방지용)
    channels = model.channels if hasattr(model, 'channels') else 2

    print(f"🚀 [OT-CFM 추론 시작] {args.fm} 모델 | 4-Step Euler | 채널 차원: {channels}")

    Path(f"class_cond_obj/{args.fm}").mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for category_id in [18, 19, 20]: 
            for i in range(total // iters):
                # 1. 100% 가우시안 노이즈(x_0) 생성
                x_init = torch.randn([iters, 512, channels], device=device) 
                labels = torch.Tensor([category_id] * iters).long().to(device)

                # 2. 4-Step Euler 수치적분으로 3D 잠재 벡터 직선 궤적 계산
                sampled_array = solver.sample(
                    x_init=x_init,
                    step_size=0.25,      # 4스텝 (0.0 -> 0.25 -> 0.50 -> 0.75 -> 1.0)
                    method='euler',      
                    class_labels=labels
                )

                # 3. VAE 디코딩 및 Marching Cubes 메쉬 추출
                for j in range(sampled_array.shape[0]):
                    logits = ae.decode(sampled_array[j:j+1], grid).detach()
                    volume = logits.view(density+1, density+1, density+1).permute(2, 1, 0).cpu().numpy()
                    vertices, triangles = mcubes.marching_cubes(volume, 0)
                    
                    vertices = (vertices - 0.5) * gap - 1.0
                    mesh = trimesh.Trimesh(vertices, triangles)
                    mesh.export(f"class_cond_obj/{args.fm}/{category_id}_{i*iters+j}.obj")
                
                print(f"   -> 카테고리 {category_id} | Batch {i+1}/{(total//iters)} 추출 완료")