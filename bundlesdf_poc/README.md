# BundleSDF proof-of-concept

## Current decision (August 2026)

**Pause BundleSDF work for now.** The synthetic data generator is built and
verified, but BundleSDF itself has not been run end-to-end in this environment.
Its difficult CUDA/C++ dependency stack is buying capabilities that the first
version of our real-to-sim pipeline probably does not need.

BundleSDF jointly tracks and reconstructs an unknown object while it moves and
may be occluded. Our immediate goal is simpler: capture metric object geometry,
combine it with the existing tipping/3D-COM and parameter-estimation work, and
produce a simulation-ready asset. If camera/object poses can be measured or
controlled during capture, ordinary RGB-D fusion is a much smaller and more
reliable solution.

### Preferred next approach

1. Capture masked RGB-D observations in several object orientations. Keep the
   object fixed and move a calibrated camera, or use a fixed camera with known
   turntable/object transforms. Robot FK, fiducials, or an encoder can provide
   the transforms.
2. Fuse the registered, metric depth maps with Open3D TSDF integration. Retain
   the original point clouds and confidence/probability values; initially use
   confidence thresholds, then add weighted TSDF updates if the uncertainty is
   useful downstream.
3. Clean, crop, scale-check, and make the reconstructed mesh watertight. Scan at
   least two stable orientations so the underside/support surfaces are observed.
4. Create a separate low-resolution collision mesh using convex decomposition
   (CoACD or VHACD). Do not use a NeRF or Gaussian-splat representation directly
   for contact physics.
5. Combine the visual/collision geometry with estimated mass, COM, inertia,
   friction, and other identified parameters. Export URDF first, and optionally
   MJCF/USD for simulators that support richer contact properties.

Open3D TSDF should be the geometry baseline because it preserves metric RGB-D
measurements and has far fewer dependencies. Nerfstudio or photogrammetry can be
added later for better appearance/texture, while the depth-derived mesh remains
the authoritative collision geometry.

### When to revisit BundleSDF

Return to BundleSDF only if the intended capture procedure truly requires most
of the following:

- the object moves with an unknown 6-DoF pose during capture;
- neither camera poses nor object poses can be supplied externally;
- hands or the robot cause substantial occlusion;
- the object is low-texture and ordinary registration repeatedly loses track;
- a controlled multi-view/turntable scan is not feasible.

Until then, do not spend more time rebuilding BundleSDF. The remaining sections
are retained as a reproducibility record in case those requirements change.

## 1. Setup / install

```bash
git clone https://github.com/NVlabs/BundleSDF.git bundlesdf_poc/BundleSDF   # already done

cd bundlesdf_poc/BundleSDF/docker
docker build --network host -t nvcr.io/nvidian/bundlesdf .
bash run_container.sh
# inside the container, one-time, machine-dependent compile step:
bash build.sh
```

Also required, per BundleSDF's own README (`bundlesdf_poc/BundleSDF/readme.md`):
download LoFTR weights (`outdoor_ds.ckpt`) into `BundleTrack/LoFTR/weights/`.
The XMem segmentation weights it also lists are **not needed** here — our
synthetic data already includes a ground-truth mask for every frame
(`--use_segmenter 0` below), so BundleSDF never needs to segment anything
itself.

## 2. Generate synthetic data

```bash
source ~/.virtualenvs/robot_learning/bin/activate
PYTHONPATH=$PWD python bundlesdf_poc/generate_synthetic_data.py
```

Writes `bundlesdf_poc/data/synthetic_cube/` (`rgb/`, `depth/`, `masks/`,
`cam_K.txt` — BundleSDF's exact input format, confirmed against
`BundleTrack/scripts/data_reader.py`) plus `gt_mesh.obj`, the ground-truth
mesh for comparison.

## 3. Run BundleSDF

Inside the docker container, against the data from step 2:

```bash
cd /path/to/bundlesdf_poc/BundleSDF
python run_custom.py --mode run_video \
  --video_dir /path/to/bundlesdf_poc/data/synthetic_cube \
  --out_folder /path/to/bundlesdf_poc/data/synthetic_cube_out \
  --use_segmenter 0 --use_gui 0

python run_custom.py --mode global_refine \
  --video_dir /path/to/bundlesdf_poc/data/synthetic_cube \
  --out_folder /path/to/bundlesdf_poc/data/synthetic_cube_out

python -c "from run_custom import postprocess_mesh; postprocess_mesh('/path/to/bundlesdf_poc/data/synthetic_cube_out')"
```

The third command isn't in BundleSDF's own CLI (`--mode` only covers
`run_video`/`global_refine`/`draw_pose`) even though its README implies a
finished mesh just appears — `postprocess_mesh()` exists in `run_custom.py`
but nothing calls it. Calling it directly is the adapter needed to actually
get the real-scale mesh out, without touching BundleSDF's source.

## 4. Reconstructed mesh

`{out_folder}/mesh/mesh_real_scale.obj` (from `postprocess_mesh`, step 3).
Compare against `bundlesdf_poc/data/synthetic_cube/gt_mesh.obj`, the
ground-truth cube from step 2.

---

## Status (not one of the 4 items above, but don't skip it)

- **Data generation is fully built and tested**, not just written: ran it,
  inspected the output numerically and visually (RGB has real shading, mask
  precisely covers the object and nothing else, depth/intrinsics are sane).
  One real bug found and fixed in the process — Genesis's segmentation
  buffer reserves `0` for background and offsets real entities by `+1`, not
  `entity.idx` directly; confirmed empirically, not assumed.
- **Running BundleSDF has not been attempted end-to-end.** This machine has
  a capable GPU (RTX 4060, 8GB, CUDA 13 driver) but **no Docker**, and
  BundleSDF's only supported install path is Docker — its dependency stack
  (custom-built OpenCV/PCL/pybind11/yaml-cpp, a C++ pybind extension in
  `BundleTrack/`, a separate CUDA extension in `mycuda/`, pinned
  torch==2.6.0 + kaolin==0.17.0 + pytorch3d) is exactly the kind of thing
  that's fragile to reproduce natively, which is presumably why NVlabs
  containerized it. A native build wasn't attempted without checking first —
  it would need its own isolated environment regardless (its pinned
  torch/CUDA versions would conflict with what's already installed for
  Genesis), and installing Docker itself needs `sudo`, which needs your
  password interactively, which isn't something this session can do.
- **Untextured cube is a real risk for BundleSDF specifically**, not just a
  cosmetic detail: LoFTR (its frame-to-frame feature matcher) needs visual
  texture to find correspondences, and the generated cube is flat-colored
  per face. If reconstruction quality is poor once BundleSDF actually runs,
  this is the first thing to try changing — a textured/checkered material
  instead of a solid color.

This section records where the experiment stopped; it is not a pending request
to install Docker or attempt another native build. Follow the "Current
decision" above unless the capture requirements change enough to justify
revisiting BundleSDF.
