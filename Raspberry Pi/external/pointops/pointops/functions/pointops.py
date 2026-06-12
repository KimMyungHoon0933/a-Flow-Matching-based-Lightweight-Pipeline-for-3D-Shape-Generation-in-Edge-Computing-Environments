import torch

def knn(x, src, k, transpose=False):
    """
    순수 PyTorch 기반 KNN 연산
    """
    if transpose:
        x = x.transpose(1, 2).contiguous()
        src = src.transpose(1, 2).contiguous()
    
    # x: [B, N, 3], src: [B, M, 3]
    # 두 포인트 클라우드 간의 모든 점의 거리를 계산 (유클리디안 거리)
    dists = torch.cdist(x, src)
    
    # 가장 거리가 짧은 k개의 인덱스와 거리를 추출
    knn_dists, knn_idx = torch.topk(dists, k=k, dim=-1, largest=False)
    
    return knn_idx, knn_dists


def fps(x, k):
    """
    순수 PyTorch 기반 FPS (Farthest Point Sampling) 연산
    """
    b, n, _ = x.shape
    device = x.device
    
    sampled_indices = torch.zeros((b, k), dtype=torch.long, device=device)
    distances = torch.ones((b, n), device=device) * 1e10
    
    # 첫 번째 점은 0번 인덱스로 고정 (또는 무작위 선택 가능)
    farthest_idx = torch.zeros((b,), dtype=torch.long, device=device)
    batch_indices = torch.arange(b, dtype=torch.long, device=device)
    
    for i in range(k):
        sampled_indices[:, i] = farthest_idx
        centroid = x[batch_indices, farthest_idx, :].view(b, 1, 3)
        
        # 선택된 중심점과 나머지 모든 점들 간의 거리 제곱 계산
        dist = torch.sum((x - centroid) ** 2, dim=-1)
        distances = torch.min(distances, dist)
        
        # 가장 거리가 먼 점을 다음 중심점으로 선택
        farthest_idx = torch.max(distances, dim=-1)[1]
    
    # 추출된 인덱스를 바탕으로 실제 좌표 반환
    idx_expanded = sampled_indices.unsqueeze(-1).expand(-1, -1, 3)
    sampled_points = torch.gather(x, 1, idx_expanded)
    
    return sampled_points


def index_points(points, idx):
    """
    기존과 동일 (이미 순수 PyTorch로 작성되어 있음)
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)