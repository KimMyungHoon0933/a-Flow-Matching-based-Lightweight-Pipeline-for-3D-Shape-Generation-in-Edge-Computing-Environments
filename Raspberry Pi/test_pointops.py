import sys
import torch

# 패키지 최상단 경로 추가
sys.path.append('./external/pointops')

# COD-VAE가 실제로 사용하는 방식으로 import
from pointops.functions import pointops

print("--- 연산 테스트 시작 ---")

points = torch.rand(2, 100, 3)
src_points = torch.rand(2, 50, 3)

try:
    print("1. KNN (K-최근접 이웃) 테스트 중...")
    idx, dist = pointops.knn(points, src_points, k=3)
    print(f"-> 통과. 반환된 인덱스 형태: {idx.shape}")

    print("\n2. FPS (최장거리 샘플링) 테스트 중...")
    fps_points = pointops.fps(points, k=10)
    print(f"-> 통과. 반환된 샘플 형태: {fps_points.shape}")

    print("\n3. Index Points (포인트 인덱싱) 테스트 중...")
    indexed = pointops.index_points(points, idx)
    print(f"-> 통과. 반환된 인덱싱 형태: {indexed.shape}")

    print("\n결론: 순수 CPU 환경에서 정상 작동합니다.")
except Exception as e:
    print(f"\n오류 발생: {e}")
