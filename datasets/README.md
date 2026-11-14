# Datasets

No raw data is redistributed in this repository. Download each dataset from its official source and
unpack it under one raw-data root (`<raw_root>`) with the layout below, then run

    python -m src.preprocessing <raw_root> datasets/processed

| Stream | Source (DOI) | Expected location under `<raw_root>` |
|---|---|---|
| Gas drift | UCI, Gas Sensor Array Drift at Different Concentrations, doi:10.24432/C5MK6M | `drift/batch1.dat` ... `drift/batch10.dat` |
| Home activity | UCI, Gas sensors for home activity monitoring, doi:10.24432/C5BS4F | `home/HT_Sensor_metadata.dat`, `home_x/HT_Sensor_dataset.dat` |
| Wine spoilage | Mendeley Data, doi:10.17632/vpc887d53s.3 | `wine_x/{AQ_Wines,HQ_Wines,LQ_Wines}/*.txt` |
| Fermentation | Mendeley Data, doi:10.17632/pdnjz7zz5x.2 | `ips/indpensim_process.csv` (see below) |
| Water reuse | Eawag, doi:10.25678/000CCT | `eawag/{sensor_data,fcm_data,lab_data}.csv` |

## Fermentation file

`100_Batches_IndPenSim_V3.csv` is 2.5 GB because it contains 2,200 simulated Raman channels. Only the
39 process columns are used. Extract them once with

    import pandas as pd
    f = "100_Batches_IndPenSim_V3.csv"
    cols = [c for c in pd.read_csv(f, nrows=0).columns if not c.strip().lstrip("-").isdigit()]
    pd.read_csv(f, usecols=cols).to_csv("ips/indpensim_process.csv", index=False)

The published header is displaced by one position (a column name contains commas). `src/preprocessing.py`
addresses columns by position: 31 = step-level fault flag, 34 = batch number, 36 = batch-level fault flag.

## Curation steps (all documented in the manuscript)

* Gas drift: within-batch order is unpublished and is permuted per seed.
* Home activity: five-minute windows; one recording has metadata but no signal and is skipped.
* Wine: bottle-held-out streams (two bottles per class form the initial pool, drawn per seed).
* Fermentation: every fourth time step; campaign re-ordered so that one fault batch follows nine normal batches.
* Water reuse: sensor readings aligned to each flow-cytometry time (one-hour look-back);
  hazard label = intact cell concentration >= 1e5 cells/mL (operational proxy, not a regulatory limit).

No measured value or original label is modified. See `dataset_information.md` for licences and sizes.
