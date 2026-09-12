# SeabedNet runnable package

Run the SeabedNet completion model on your own soundings. Nothing leaves your machine; no GPU is required.

## What it does
Given a grid of soundings in EPSG:3857 (100 m cells, depth in metres, negative below sea level, NaN where you have none),
`seabednet_complete.py` returns, for every cell within 6 km of a sounding: the completed depth, a 1-sigma uncertainty, and the
provenance mask (which cells were yours). The models are masked-completion U-Nets trained on the CHS NONNA-100 archive with a
Sandwell/SRTM15+ gravity anchor; the temporal benchmark (Exhibit H/M of the atlas) is the held-out test that applies to them.

## Files
- `seabednet_complete.py` — the tool (this repo, `validation_repo/package/`)
- `v5_model.py`, `v5_data.py` — model and prior code (same repo)
- weights: `v5_small.pt` (published v5 map), `v5_small_ctlall.pt`, `v5_small_s1all.pt` (radar member; only used if a winter Sentinel-1 file exists for the block) — GitHub release `ship`
- `planetary/gravity_prior_canada.npz` — gravity anchor, Canada window — release `ship`

## Run
```
pip install torch numpy scipy rasterio
python3 seabednet_complete.py --in block.npz --out block_completed            # NONNA-style npz: z [H,W] float32, bbox3857 [x0,y0,x1,y1]
python3 seabednet_complete.py --tif soundings.tif --out completed              # EPSG:3857 GeoTIFF, nodata = no sounding; writes completed.tif (depth, sigma)
python3 seabednet_complete.py --in block.npz --out out --models v5_small.pt --cpu --bw 4   # CPU only
```
Output `out.npz`: `complete`, `sigma` (float32 metres), `known` (bool), `bbox3857`, `members` (number of models averaged).

## Check it reproduces the published map
Corridor block b0033_0033 on CPU with the single published model: 630,765 completed cells; median difference to the published
national_v5_out block 0.3 m, 90th percentile 3.0 m (float16 storage in the published file and a slightly different fill mask).

## How to test it on soundings the model has never seen
Hide a survey you hold: set its cells to NaN in the input, run the tool, compare `complete` at those cells with the survey. The
temporal benchmark in the atlas is exactly this protocol with post-2016 soundings hidden. Report MAE by depth band and the share of
errors inside 1 sigma; the published numbers to beat are in Exhibit M.
