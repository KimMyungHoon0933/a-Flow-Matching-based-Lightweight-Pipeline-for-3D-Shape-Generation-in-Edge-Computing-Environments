"""
verify_exp2_fixed.py — 실험 2: 청크 분할 필요 위치 검증 (수정본)
===========================================================
CompactLatentVAE 구조:
  vae.autoencoder.decoder  ← CompactTriplaneDecoder가 여기에 있음

사용법 (Raspberry Pi에서):
  python verify_exp2_fixed.py \
    --diffusion_ckpt <디퓨전 체크포인트> \
    --vae_config <VAE config 경로> \
    --vae_weights <VAE 가중치 경로>
===========================================================
"""

import argparse
import torch
import torch.nn.functional
import sys
import os
import gc
import time
import types
import traceback

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

# einsum 패치 적용
_original_linear = torch.nn.functional.linear

def safe_einsum_linear(input, weight, bias=None):
    orig_shape = input.shape
    x = input.reshape(-1, orig_shape[-1]).float()
    w = weight.float()
    out = torch.einsum('xi,ji->xj', x, w)
    if bias is not None:
        out = out + bias.float()
    return out.reshape(*orig_shape[:-1], out.shape[-1])

torch.nn.functional.linear = safe_einsum_linear

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'external/pointops')))

from cod.utils.recon import create_grid_queries

# ==========================================
# 결과 기록
# ==========================================
results = []

def log(msg):
    print(msg)
    results.append(msg)

def save_results():
    with open("verify_exp2_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    print(f"\n결과가 verify_exp2_results.txt에 저장되었습니다.")


# ==========================================
# 청크 없이 직접 호출
# ==========================================
def direct_forward(module, x, chunk_size=512):
    B, N, C = x.shape
    x_2d = x.reshape(B * N, C)
    out_2d = module(x_2d)
    return out_2d.reshape(B, N, -1)


# ==========================================
# 실험 2 메인
# ==========================================
def run_experiment_2(args):
    log("=" * 60)
    log("실험 2: 기여 ② 검증 — 청크 분할이 어디에 필요한가")
    log("=" * 60)
    log(f"시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("")

    device = torch.device('cpu')

    # 모델 로드 (diffusion 체크포인트 사용)
    log("모델 로딩 중...")
    from duffusion.diffusion_lightning import CategoryConditionedDiffusion
    model = CategoryConditionedDiffusion.load_from_checkpoint(
        checkpoint_path=args.diffusion_ckpt,
        vae_config_path=args.vae_config,
        vae_weights_path=args.vae_weights,
        map_location=device
    )
    model.to(device).eval().to(torch.float32)

    # VAE 참조
    vae = model.vae

    # [핵심 수정] CompactLatentVAE → autoencoder → decoder
    decoder = vae.autoencoder.decoder
    log(f"  decoder 타입: {type(decoder).__name__}")
    log(f"  decoder 설정: embed_dim={decoder.embed_dim}, query_dim={decoder.query_dim}")
    log(f"  plane_resolution={decoder.plane_resolution}, num_output_patches={decoder.num_output_patches}")

    # 가짜 latent z 생성 (Flow 모델 샘플링 없이 테스트)
    log("\n가짜 latent z 생성 (랜덤 [1, 32, 32])...")
    sampled_z = torch.randn(1, 32, 32, dtype=torch.float32, device=device)

    # 원본 _chunked_forward 백업
    original_chunked_forward = decoder._chunked_forward

    # 테스트 조합
    test_configs = [
        ("1) 전부 청크 (현재 상태)",            True,  True,  True),
        ("2) uncertainty_out만 직접",           True,  True,  False),
        ("3) init_out만 직접",                  True,  False, True),
        ("4) decoder_out만 직접",               False, True,  True),
        ("5) decoder_out + init_out 직접",      False, False, True),
        ("6) 전부 직접 (청크 없음)",             False, False, False),
    ]

    for config_name, chunk_decoder, chunk_init, chunk_uncertainty in test_configs:
        log(f"\n--- 테스트: {config_name} ---")
        log(f"    decoder_out:     {'청크' if chunk_decoder else '직접'}")
        log(f"    init_out:        {'청크' if chunk_init else '직접'}")
        log(f"    uncertainty_out: {'청크' if chunk_uncertainty else '직접'}")

        gc.collect()

        # ---- decode_tokens 패치 (uncertainty_out 제어) ----
        def make_patched_decode_tokens(use_chunk_unc):
            def patched_decode_tokens(self, tokens, z):
                init_tokens = self.init_transformer(tokens, z)
                init_tokens = init_tokens.contiguous()

                if use_chunk_unc:
                    uncertainty = original_chunked_forward(self.uncertainty_out, init_tokens)
                else:
                    uncertainty = direct_forward(self.uncertainty_out, init_tokens)
                uncertainty = self.apply_uncertainty_activation(uncertainty)

                L_full = tokens.size(1)
                tokens_sel, indices, prune_indices = self._select_by_uncertainty(
                    init_tokens, uncertainty, self.keep_ratio)
                if self.mask_token is not None:
                    tokens_sel = tokens_sel + self.mask_token.unsqueeze(0)

                L = tokens_sel.size(1)
                if self.merging_module is not None:
                    pruned = torch.gather(init_tokens, 1,
                                           prune_indices.expand(-1, -1, tokens_sel.size(-1)))
                    merged = self.merging_module(pruned)
                    tokens_sel = torch.cat([tokens_sel, merged], dim=1)

                x = torch.cat([tokens_sel, z], dim=1)
                tokens_out = self.transformer(x)
                tokens_out = tokens_out[:, :L]
                tokens_out = tokens_out.contiguous()

                full_tokens = torch.zeros(tokens_out.size(0), L_full, tokens_out.size(2),
                                           device=tokens_out.device, dtype=tokens_out.dtype)
                mask = torch.ones(tokens_out.size(0), L_full,
                                   device=tokens_out.device, dtype=torch.bool)
                mask.scatter_(1, indices.squeeze(-1), 0)
                full_tokens[mask] = init_tokens[mask]
                _indices = indices.expand(-1, -1, full_tokens.size(-1))
                full_tokens.scatter_(1, _indices, tokens_out)

                return init_tokens, full_tokens, uncertainty
            return patched_decode_tokens

        # ---- decode 패치 (decoder_out, init_out 제어) ----
        def make_patched_decode(use_chunk_dec, use_chunk_ini):
            def patched_decode(z_input):
                B = z_input.size(0)
                tokens = decoder.mask_pos.unsqueeze(0).expand(B, -1, -1)
                init_tokens, tokens, uncertainty = decoder.decode_tokens(tokens, z_input)

                tokens = tokens.contiguous()
                init_tokens = init_tokens.contiguous()

                if use_chunk_dec:
                    patches = original_chunked_forward(decoder.decoder_out, tokens)
                else:
                    patches = direct_forward(decoder.decoder_out, tokens)

                if use_chunk_ini:
                    init_patches = original_chunked_forward(decoder.init_out, init_tokens)
                else:
                    init_patches = direct_forward(decoder.init_out, init_tokens)

                patches = init_patches + uncertainty * patches
                planes = decoder.patches_to_planes(patches)
                uncertainty_view = uncertainty.view(
                    uncertainty.size(0), 3, 1,
                    decoder.plane_resolution, decoder.plane_resolution)
                init_planes = decoder.patches_to_planes(init_patches)

                if decoder.conv_refine is not None:
                    planes = planes.flatten(0, 1)
                    planes = decoder.conv_refine(planes) + planes
                    planes = planes.view(B, 3, decoder.query_dim,
                                          decoder.output_resolution, decoder.output_resolution)
                    init_planes = init_planes.flatten(0, 1)
                    init_planes = decoder.conv_refine(init_planes) + init_planes
                    init_planes = init_planes.view(B, 3, decoder.query_dim,
                                                    decoder.output_resolution, decoder.output_resolution)

                return planes, init_planes, uncertainty_view
            return patched_decode

        # 백업
        orig_decode = decoder.decode
        orig_decode_tokens = decoder.decode_tokens

        try:
            # 패치 적용
            decoder.decode_tokens = types.MethodType(
                make_patched_decode_tokens(chunk_uncertainty), decoder)
            decoder.decode = make_patched_decode(chunk_decoder, chunk_init)

            start = time.time()

            with torch.no_grad():
                # VAE 디코딩 경로: decode_latents → decode_embed → decode_queries
                decoded_latents = vae.decode_latents(sampled_z)
                context, _, _, _ = vae.decode_embed(decoded_latents)

                queries = create_grid_queries(128).to(device)
                logits = []
                for i in range(0, queries.size(1), 50000):
                    chunk = queries[:, i:i+50000]
                    logit = vae.decode_queries(context, chunk)
                    logits.append(logit)
                logits = torch.cat(logits, dim=1)

            elapsed = time.time() - start
            log(f"    ✅ 성공! (소요: {elapsed:.2f}초, logits shape: {list(logits.shape)})")

        except RuntimeError as e:
            err_str = str(e).lower()
            if "memory" in err_str or "alloc" in err_str:
                log(f"    ❌ 메모리 초과! — {e}")
            elif "primitive" in err_str or "matmul" in err_str:
                log(f"    ❌ matmul 호환성 에러! — {e}")
            else:
                log(f"    ❌ RuntimeError — {e}")
        except Exception as e:
            log(f"    ❌ 실패 — {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            # 복원
            decoder.decode = orig_decode
            decoder.decode_tokens = orig_decode_tokens
            gc.collect()

    log("\n" + "=" * 60)
    log("실험 2 완료!")
    log("=" * 60)


# ==========================================
# 메인
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("실험 2: 청크 분할 검증 (수정본)")
    parser.add_argument("--diffusion_ckpt", type=str, required=True,
                        help="디퓨전 체크포인트 (VAE 포함)")
    parser.add_argument("--vae_config", type=str, required=True)
    parser.add_argument("--vae_weights", type=str, required=True)
    args = parser.parse_args()

    run_experiment_2(args)
    save_results()