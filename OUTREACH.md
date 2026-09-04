# Outreach drafts — send-ready, numbers match the live page (2026-09-03)

Nothing here claims more than the page shows.

## 0. Verified contacts (checked 3 Sept 2026; official pages only, no guessed personal addresses)

| # | Who | Send to | Draft |
|---|---|---|---|
| 1 | **Manon Larocque**, Hydrographer General of Canada / DG CHS (since Sept 2023) | chsinfo@dfo-mpo.gc.ca, attention Ms Larocque · 613-998-4931 · 200 Kent St, Station 12W090, Ottawa K1A 0E6 | 2 |
| 2 | **TSB ATIP Coordinator** (legal 30-day reply); cc Clifford Harvey, Director of Investigations (Marine) | atip.aiprp@tsb-bst.gc.ca | 5 |
| 3 | **Chris Avery**, CEO Arctic Gateway Group (since July 2024) | web form arcticgateway.com/contact · (204) 805-7042 · 728 Bignell Ave, The Pas MB R9A 1L8 (no public email) | 3 |
| 4 | **Major Projects Office** (Privy Council) intake | canada.ca/en/privy-council/major-projects-office/proponent-intake-form.html · info@mpo-bgp.gc.ca | 1 |
| 4b | **TC Northern Low-Impact Shipping Corridors** | OPPCorridorsPPO@tc.gc.ca | 4 |
| 5 | **Martin Jakobsson**, Stockholm U (IBCAO / Seabed 2030 Arctic) | martin.jakobsson@geo.su.se | 6 |
| 6 | **Larry Mayer**, UNH CCOM | Larry.Mayer@unh.edu | 6 |
| 7 | **Ian Church**, UNB Ocean Mapping Group | ian.church@unb.ca | 6 |
| 8 | **Alexandre Forest**, Amundsen Science, U Laval | alexandre.forest@as.ulaval.ca | 6 |
| 9 | **Ocean Frontier Institute** | ofi@dal.ca (media: oficomms@dal.ca) | 6 |
| Day 7 | **Nunatsiaq News** | editors@nunatsiaq.com (Corey Larocque, coreyl@nunatsiaq.com) | 7 |
| Day 7 | **CBC North** | nunavut@cbc.ca · 867-979-6135 | 7 |
| Day 7 | **Hakai Magazine** | news@hakaimagazine.com / editor@hakaimagazine.com | 7 |
| Day 7 | **gCaptain** | editorial@gcaptain.com | 7 |
| Day 7 | **The Globe and Mail** | tips@globeandmail.com | 7 |

Flags: Harvey's 2026 tenure confirmed only via cached TSB pages; Arctic Gateway publishes no email (use the form and follow with a call).

---

## 1. Transport Canada / MPO — Churchill Plus "Project intake form" (Major Projects Office)

**Project name:** SeabedNet Churchill Corridor Atlas — model-completed bathymetry and charting-risk map for the Churchill trade corridor

**Proponent:** SeabedNet (Emilio Girard), Montréal, QC — sole proprietor, ML engineer

**Sector:** Marine transportation / hydrography / trade-corridor infrastructure

**One-paragraph summary:**
Canada, Manitoba and Saskatchewan have committed C$262.5M to reopen the Churchill corridor, yet CHS reports only 15.8% of Canadian Arctic waters and 44.7% of key routes are adequately surveyed, and under the Churchill route itself 17% of the seabed has a published sounding with a median survey year of 1974. SeabedNet is a machine-learning completion of the CHS NONNA archive that produces depth, calibrated uncertainty and a shoal-hazard probability for every 100 m cell of the corridor, with the provenance of every pixel kept explicit. It has been scored on two independent tests: (1) 109,044 research-cruise multibeam depths CHS never published (mean error 4.9 m vs 13.4 m for the nearest published sounding); (2) a model trained only on pre-2016 soundings predicting the 6.46 million soundings CHS collected after 2016 (13.3 m vs 18.1 m for the gravity-derived bathymetry global charts fall back on, and 16.2–16.8 m for gravity-trend-plus-interpolation). A hazard model predicting the shallowest point within 500 m, given only pre-incident soundings, ranked the strike site of 4 of 7 TSB-documented Arctic groundings in the top 10% of danger among water that looked safe (binomial p = 0.001), including Thamesborg 2025 with every sounding within 10 km withheld. All code, validation cells and a public benchmark are online.

**What we are asking for (three items):**
1. **A validation agreement with CHS:** access to CHS multibeam for the Churchill corridor not yet released to NONNA, under an agreement that we score the model on it and publish the score whatever it is.
2. **A charting-risk budget line in Churchill Plus**, sized by the σ-ranked survey plan on the atlas: ~219 ship-days, C$23M at the rate implied by CHS's 2023–25 Arctic survey contract to C$40M at the Coast Guard's July 2026 polar-icebreaker charter day rate, to de-risk the ten highest-uncertainty boxes on the route in one season.
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

1. **A temporal test using your own Survey Index.** Training only on soundings inside the 1832–2016 index polygons, the model predicts the 6.46 million post-2016 soundings in the corridor to 13.3 m mean error (gravity prior 18.1 m; gravity trend plus interpolated residuals 16.2–16.8 m; nearest old sounding 18.8 m). It also exposes where the archive's mean depth is the wrong quantity: under 50 m of water, mean-depth models read 6–9 m too deep. The hazard model targets the shallowest point within 500 m instead, learned from NONNA-10/100 pairs.
2. **A grounding hindcast** on TSB cases M96H0016, M00N0098, M10H0006, M12H0012, M14C0219, M18C0225 and M25C0241, using only pre-incident soundings and, in the blind variant, no sounding within 10 km of the strike: the shallowest-point hazard field ranks 4 of the 7 strike sites in the top 10% of danger among water the mean map called safe (binomial p = 0.001; 3 of the 4 uncharted-shoal cases). The TSB counts 74 Arctic groundings for 2000–2018; these seven are the ones with a published position, and I have asked the Board for the rest.

I'd welcome a correction on anything, and I'd welcome more: if CHS holds corridor multibeam not yet in NONNA, I would score the model on it under any agreement you prefer and publish the result, good or bad. There is also a sealed forecast file (SHA-256 on the page) of 1,314 predicted depths at unsounded cells that the next survey through the corridor can grade.

Page: https://girardemilio3-svg.github.io/churchill-corridor-atlas/
Code and validation cells: https://github.com/girardemilio3-svg/seabednet-validation

With respect for the work your surveyors do in that water,
Emilio Girard, Montréal

---

## 3. Arctic Gateway Group — one paragraph for the intro email

Subject: A charting-risk map for the Churchill route, validated on CHS's own data — would AGG carry it into Churchill Plus?

Under the route your ships run, 17% of the seabed has a published sounding and the median survey year is 1974. I built and publicly validated a model that fills the rest with depth, uncertainty and a shoal-hazard probability, then priced the survey season that removes the worst of it (C$23–40M, ~219 ship-days, bracketed by CHS's own Arctic survey contract and the Coast Guard's icebreaker charter). It is not a chart and says so; it is the document that lets the corridor ask for a charting budget line with a number on it. I would like AGG to own the file on the proponent side — I do the technical work, you carry it to the table. Ten minutes on a call and I'll show you the route under keel, kilometre by kilometre.

---

## 4. Transport Canada Corridors office (OPPCorridorsPPO@tc.gc.ca)

Subject: Request — Northern Low-Impact Shipping Corridors polygons for the Hudson Bay / Hudson Strait / Labrador route

Requesting the current NLISC corridor polygons (or a pointer to their public release) to overlay the model-found least-risk channel on the official corridors and report where they disagree. Purpose: a public charting-risk atlas for the Churchill corridor (link). Happy to share the overlay back.

---

## 5. Transportation Safety Board of Canada — data request (Marine occurrence records)

To: ATIP Coordinator, Transportation Safety Board of Canada — atip.aiprp@tsb-bst.gc.ca (Place du Centre, 200 Promenade du Portage, 4th floor, Gatineau QC K1A 1K8); cc Director of Investigations (Marine), Clifford Harvey, via the TSB switchboard.

Subject: Request for occurrence-level records of vessel groundings and bottom contacts in Canadian Arctic waters, 1990–2026

I am requesting, in machine-readable form (CSV or spreadsheet), the occurrence-level records held by the TSB for marine groundings and bottom contacts in Canadian Arctic waters (Hudson Bay, Hudson Strait, the Labrador coast north of 55°N, and the Arctic Archipelago) from 1 January 1990 to the present. For each occurrence: occurrence number, date, vessel name and type, position (latitude/longitude as recorded), reported draft, water depth or chart information where recorded, and the TSB's classification (grounding / bottom contact / striking). The TSB's report M18C0225 cites 74 such occurrences for 2000–2018; I am seeking the underlying list.

Purpose: a public, reproducible hindcast benchmark for seabed-hazard models (see https://girardemilio3-svg.github.io/churchill-corridor-atlas/report/ §5), in which each grounding is scored against a model given only the soundings that existed before the incident. The benchmark currently uses the seven Arctic groundings for which a position is published in a TSB report; the full occurrence list would make it complete. Results will be published whatever they show.

I am content to receive the records under the Open Government Licence or with any redaction the Board considers necessary for personal information; vessel names may be withheld if positions and dates are retained.

Emilio Girard, SeabedNet, Montréal · girardemilio3@gmail.com

---

## 6. Academic / scholarship cold email (Jakobsson, Mayer, Church, Forest, OFI)

Subject: A temporal-holdout benchmark for bathymetric completion on CHS NONNA, plus a grounding hindcast — would you look?

Dear Professor [Name],

I'm an independent ML engineer in Montréal. Over the past week I built and validated a completion model for Canada's NONNA archive and released the part I think your field is missing: a temporal-holdout benchmark. Training only on soundings inside the CHS Survey Index (1832–2016) and scoring on the 20.4 million soundings CHS collected afterwards, the model reaches 12.6 m against 15.7 m for SRTM15+ and 16.2–16.8 m for gravity-trend-plus-interpolation; the split, targets and scorer are public (NONNA-Temporal-Churchill v1).

Two things you may find directly relevant: [for Jakobsson: an audit of where the GEBCO/IBCAO grid disagrees with the CHS archive by >10 m along the Churchill route, and the caveat that GEBCO contains the test set via IBCAO v5, which I state rather than exploit] [for Mayer: the benchmark and the shallowest-point hazard model, hindcast on seven TSB groundings] [for Church/Forest: the independent test on Amundsen multibeam and the hazard field for Hudson Strait].

Everything, including the negative results (an ICESat-2 ATL24 attempt that cannot referee turbid water; an era audit that came out null), is on the record: [links]. I'd value ten minutes of your criticism more than praise, and I'm looking for a place to do this properly — a graduate position, a lab, or a project.

Emilio Girard · Montréal · girardemilio3@gmail.com

---

## 7. Press pitch (Day 7: Nunatsiaq News, CBC North, Hakai, Globe, gCaptain) — send with the sentence "CHS and the TSB received this on [date]"

Subject: A desktop model says it can see the Arctic shoals Canada hasn't charted — and it has put 40 of them in writing

A solo engineer in Montréal has published a machine-learning completion of Canada's official seabed archive and, with it, a sealed list of 40 places in Arctic and coastal shipping water where the official depth map says safe and the model says a keel-depth rock lies within 500 m. The list is timestamped on a public ledger; each entry is settled by a single survey line.

The numbers behind it are public and checkable: trained only on soundings from before 2016, the model predicted 20 million soundings the Canadian Hydrographic Service collected afterwards more accurately than the gravity-based charts the world falls back on; run against seven real Arctic groundings using only the soundings that existed before each ship hit, it ranked four of the seven strike sites in the top 10% of danger among water the chart called safe. It also grades every Nunavut resupply lane: Naujaat's approach rests on surveys with a median year of 1955; Deception Bay's, where ore ships dock, on 1960.

Ten years ago CHS said full Arctic charting was "more than a decade" away. The decade has passed; Ottawa, Manitoba and Saskatchewan are investing C$262.5M in the Churchill corridor with no charting line in the budget. The atlas prices one: C$23–40M for the corridor's worst water.

Page, report, code, and the sealed list: [links]. Available for interview; the TSB and CHS were informed on [date].
