# Edge Deployment Notes

This note collects non-runtime guidance for deploying the lightweight
flow-matching 3D generation pipeline on edge devices such as Raspberry Pi 5.
It is documentation only and is not imported by any training, generation, or
evaluation script.

## Deployment Checklist

- Confirm that the target device has enough free memory for both latent
  sampling and COD-VAE decoding.
- Install dependencies in an isolated Python environment before running
  generation or benchmark scripts.
- Use the Raspberry Pi benchmark scripts to measure local runtime before
  comparing against reported results.
- Keep generated meshes, logs, and intermediate outputs outside version
  control unless they are intentionally added as reproducible examples.

## Validation Notes

- Run a short category-conditioned generation test before long benchmark runs.
- Compare CPU and memory usage across multiple runs because edge devices can
  vary with thermal state and background processes.
- Record device model, RAM size, Python version, PyTorch version, and command
  arguments when reporting performance numbers.

## Scope

This file is intended to help contributors document edge-device checks without
changing model code, configuration files, dependencies, or runtime behavior.
