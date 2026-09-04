# Research statement — Emilio Girard, SeabedNet (Montréal, September 2026)

**One sentence.** I built, validated and published a machine-learning completion of Canada's national bathymetric archive that predicts depth, calibrated uncertainty and a shallowest-point grounding hazard for every 100 m cell of the country's charted water, and I put its claims on the public record where they can be checked.

**The problem.** Canada's Arctic is 15.8% adequately surveyed; under the Churchill trade corridor, 17% of the seabed has a published sounding and the median survey year is 1974. The C$262.5M Churchill Plus program funds port and rail, not charting. Ships ground on what is between the soundings.

**What is new.**
1. *Temporal validation as the standard.* Random or geographic splits of a survey archive leak along survey lines. I split on the CHS Survey Index date instead: train on soundings from before 2016, score on the 20.4 million soundings CHS collected afterwards. Model 12.6 m; nearest sounding 19.7 m; gravity-derived bathymetry (SRTM15+) 15.7 m; gravity-trend-plus-interpolation 16.2–16.8 m; 74% of errors inside the model's own 1σ. Released as a public benchmark (NONNA-Temporal-Churchill v1) with a scorer and leaderboard.
2. *The extreme, not the mean.* Ships ground on the shallowest point within the swept path, which a mean-depth model cannot represent. Using NONNA-10/100 pairs as ground truth for sub-cell extremes, a second model predicts the minimum depth within 500 m (2.6 m error on held-out tiles vs 10.7 m if the archive depth is taken as the shoal) and outputs P(shoal < draft) per cell.
3. *A grounding hindcast benchmark.* Every TSB Arctic grounding with a published position (seven), scored with only pre-incident soundings and, in the blind variant, no sounding within 10 km of the strike: 4 of 7 sites in the top decile of hazard among water the chart called safe (binomial p = 0.003), 5 of 7 with all soundings visible (p = 0.0002).
4. *Independent tests and negative results, all published.* 109k research-cruise multibeam cells CHS never published (4.9 m vs 13.4 m nearest sounding); gravity-leakage measured and stated; an ICESat-2 ATL24 laser test attempted and reported as a negative result; an era audit reported as null.
5. *Sealed, falsifiable claims.* A forecast of 1,314 depths at unsounded cells and a Shoal List of 40 suspected uncharted keel-depth shoals, both SHA-256 + OpenTimestamps sealed, graded automatically as the archive updates.

**Why it matters.** The pipeline runs on one desktop from public data; the full-size model trains in 17 minutes on a consumer GPU. It converts a decade-scale survey backlog into a ranked, priced survey plan (C$23–40M for the corridor's worst water; C$103–181M nationally) and gives navigators, insurers and CHS a hazard field where the archive is blind.

**What I want to do next.** Fuse satellite-derived bathymetry (ICESat-2, Sentinel-2) into the shallow band where the archive and the mean model are weakest; extend the hindcast to the TSB's full occurrence record (74 groundings, 2000–2018) once released; and formalize the temporal benchmark as a community standard for bathymetric completion.

**Links.** Atlas: https://girardemilio3-svg.github.io/churchill-corridor-atlas/ · Technical report: https://girardemilio3-svg.github.io/churchill-corridor-atlas/report/ · Code, results, benchmark: https://github.com/girardemilio3-svg/seabednet-validation
