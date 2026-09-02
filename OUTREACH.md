# Outreach drafts — send-ready, numbers pulled from the live page (2026-09-02)

Fill the `[…]` slots after the hindcast lands. Nothing here claims more than the page shows.

---

## 1. Transport Canada / MPO — Churchill Plus "Project intake form" (Major Projects Office)

**Project name:** SeabedNet Churchill Corridor Atlas — model-completed bathymetry and charting-risk map for the Churchill trade corridor

**Proponent:** SeabedNet (Emilio Girard), Montréal, QC — sole proprietor, ML engineer

**Sector:** Marine transportation / hydrography / trade-corridor infrastructure

**One-paragraph summary:**
Canada, Manitoba and Saskatchewan have committed C$262.5M to reopen the Churchill corridor, yet CHS reports only 15.8% of Canadian Arctic waters and 44.7% of key routes are adequately surveyed, and under the Churchill route itself 17% of the seabed has a published sounding with a median survey year of 1974. SeabedNet is a machine-learning completion of the CHS NONNA archive that produces depth, calibrated uncertainty and a shoal-hazard probability for every 100 m cell of the corridor, with the provenance of every pixel kept explicit. It has been scored on two independent tests: (1) 109,044 research-cruise multibeam depths CHS never published (mean error 4.9 m vs 13.4 m for the nearest published sounding); (2) a model trained only on pre-2016 soundings predicting the 6.46 million soundings CHS collected after 2016 (13.5 m vs 18.1 m for the gravity-derived bathymetry global charts fall back on). [Hindcast sentence.] All code and validation data are public.

**What we are asking for (three items):**
1. **A validation agreement with CHS:** access to CHS multibeam for the Churchill corridor not yet released to NONNA, under an agreement that we score the model on it and publish the score whatever it is.
2. **A charting-risk budget line in Churchill Plus**, sized by the σ-ranked survey plan on the atlas: ~219 ship-days, ~C$40M at the Coast Guard's July 2026 polar-icebreaker charter day rate (C$183k/day), to de-risk the ten highest-uncertainty boxes on the route in one season.
3. **Arctic Gateway Group as the file owner** on the proponent side; SeabedNet as the technical contributor.

**Readiness / stage:** Working prototype, publicly validated; national completion of all 437 NONNA blocks already run. Not for navigation; planning prior.

**Links:** https://girardemilio3-svg.github.io/churchill-corridor-atlas/ · https://github.com/girardemilio3-svg/seabednet-validation

**Contact:** girardemilio3@gmail.com

---

## 2. CHS courtesy note (chsinfo@dfo-mpo.gc.ca, cc CHS Central & Arctic)

Subject: Independent validation of a model-completed NONNA product for the Churchill corridor — sharing before wider circulation

Hello,

I'm sharing this with CHS first, before it goes to Transport Canada's Churchill Plus intake, because it is built on your data and I would rather you see it from me.

SeabedNet is a machine-learning completion of NONNA-100/10 that predicts depth, calibrated uncertainty and a shallowest-point hazard probability for every 100 m cell of the Churchill corridor. It is explicitly a planning prior, stamped "not for navigation", and it never claims "unsurveyed" — only "no published sounding", since I know NONNA is not the whole holdings.

Two things you may find directly useful:

1. **A temporal test using your own Survey Index.** Training only on soundings inside the 1832–2016 index polygons, the model predicts the post-2016 soundings in the corridor to 13.5 m mean error (gravity prior 18.1 m; nearest old sounding 18.8 m). It also exposes where the archive's mean depth is the wrong quantity: under 50 m of water, mean-depth models read 12–14 m too deep. The hazard model targets the shallowest point within 500 m instead, learned from NONNA-10/100 pairs.
2. **A grounding hindcast** on TSB cases M18C0225, M10H0006, M96H0016, M12H0012, M14C0219 and M25C0241, using only pre-incident soundings: [one sentence with the result].

I'd welcome a correction on anything, and I'd welcome more: if CHS holds corridor multibeam not yet in NONNA, I would score the model on it under any agreement you prefer and publish the result, good or bad. There is also a sealed forecast file (SHA-256 on the page) of 1,314 predicted depths at unsounded cells that the next survey through the corridor can grade.

Page: https://girardemilio3-svg.github.io/churchill-corridor-atlas/
Code and validation cells: https://github.com/girardemilio3-svg/seabednet-validation

With respect for the work your surveyors do in that water,
Emilio Girard, Montréal

---

## 3. Arctic Gateway Group — one paragraph for the intro email

Subject: A charting-risk map for the Churchill route, validated on CHS's own data — would AGG carry it into Churchill Plus?

Under the route your ships run, 17% of the seabed has a published sounding and the median survey year is 1974. I built and publicly validated a model that fills the rest with depth, uncertainty and a shoal-hazard probability, then priced the survey season that removes the worst of it (~C$40M, ~219 ship-days). It is not a chart and says so; it is the document that lets the corridor ask for a charting budget line with a number on it. I would like AGG to own the file on the proponent side — I do the technical work, you carry it to the table. Ten minutes on a call and I'll show you the route under keel, kilometre by kilometre.

---

## 4. Transport Canada Corridors office (OPPCorridorsPPO@tc.gc.ca)

Subject: Request — Northern Low-Impact Shipping Corridors polygons for the Hudson Bay / Hudson Strait / Labrador route

Requesting the current NLISC corridor polygons (or a pointer to their public release) to overlay the model-found least-risk channel on the official corridors and report where they disagree. Purpose: a public charting-risk atlas for the Churchill corridor (link). Happy to share the overlay back.
