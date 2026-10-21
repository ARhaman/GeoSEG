# GeoSEG

Code for a controlled comparison of training-data strategies for seismic facies segmentation on the F3 Netherlands benchmark. The accompanying manuscript is in preparation; results, trained models and figures will be released with the publication.

This repository contains the code so that reviewers and readers can inspect the implementation. For questions or access requests, contact the corresponding author (below).

## Layout

```
geoseg/      core package: configuration, data, models, augmentation, training, metrics
scripts/     numbered pipeline, run in order
analysis/    paired statistics (bootstrap) from per-checkpoint metric tables
```

| Script | Purpose |
|---|---|
| `00_prepare_data.py` | Patch extraction, splits, standardisation |
| `01_fig2_class_imbalance.py` | Class frequencies of the training volume |
| `02_train_pix2pix.py` | Pix2Pix and L1-only generators, one per budget and seed |
| `03_run_experiment_a.py` | Fixed-epoch segmentation training |
| `04`–`08` | Uncertainty analysis, diagnostics and figure scripts |
| `09_step_matched_control.py` | Step-matched training of all arms |
| `10_ablation_summary.py` | Paired comparisons |
| `11_uncertainty_quality.py` | MC-Dropout, ensembles, calibration and error-detection metrics |
| `12_second_backbone.py` | Second segmentation network |
| `13_stepmatched_illustrations.py` | Example predictions and uncertainty maps |
| `run_when_free.py` | Waits for free GPU memory, then launches the sweeps |

## Running

```bash
conda create -n geoseg python=3.11 -y && conda activate geoseg
pip install -r requirements.txt            # install the CUDA build of PyTorch for your system
export GEOSEG_DATA_ROOT=/path/to/f3        # folder containing train/ and test_once/
export GEOSEG_RUN_ROOT=/path/for/outputs
python scripts/00_prepare_data.py
```

On Windows PowerShell use `$env:GEOSEG_DATA_ROOT = "..."`. Stages are resumable: a run whose output file exists is skipped. If your copy of the data has extra folder levels, set `GEOSEG_TRAIN_DIR` and `GEOSEG_TEST_DIR`.

**Data.** F3 Netherlands is third-party data and is not redistributed. It is available from the [Alaudah et al. (2019) benchmark release](https://github.com/yalaudah/facies_classification_benchmark) and the [Open Seismic Repository](https://opendtect.org/osr/pmwiki.php/Main/NetherlandsOffshoreF3BlockComplete); their licence terms apply.

**Hardware.** Developed with PyTorch 2.6 on a single NVIDIA RTX 4500 Ada (24 GB).

## License

Code is released under the [MIT License](LICENSE).

## Contact

Abdulrahman Al-Fakih, Center for Integrative Petroleum Research, King Fahd University of Petroleum and Minerals.
