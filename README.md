# End-to-End Multimodal Detection of Lunar and Martian Lava Tubes

Optical and thermal orbital data, fused end to end, for classifying candidate
cave entrances on Mars.

## The problem

Lava-tube skylights are the strongest candidate shelters for long-term
exploration: they shield against radiation and micrometeoroid impacts. Finding
them means telling a **cave-connected pit** from an ordinary collapse pit, and
optical imagery alone struggles because the two look alike from above.

Cushing et al. (2007) give a physical discriminant: a cave-connected pit swings
*less* in temperature over a Martian day than the pits around it, being warmer
than its surroundings before dawn and cooler than sunlit ground in the
afternoon. That suggests fusing high-resolution imagery with thermal
observations.

## The approach

A two-branch network over the 1846 annotated landforms of DeepLandforms
(Nodjoumi et al. 2023):

* an **optical branch** — a fixed 960 m HiRISE crop at 2.5 m/pixel through an
  ImageNet-pretrained ResNet-18;
* a **thermal branch** — three THEMIS windows at native 100 m/pixel, tagged with
  their local solar times and read by a dilated temporal convolution.

The projected embeddings are concatenated and classified. Every result is
scored against baselines that deliberately cannot see morphology, with splits
grouped at HiRISE-product level so no landform, resolution replica or shared
acquisition straddles a fold.

## What it finds

* **Optical detection works.** 91.3% accuracy and 90.4 macro F1 on
  skylight-versus-rest, against a 77.8% baseline that sees only a permuted
  intensity histogram and an 84.3% oracle holding each landform's true size.
* **The crop policy is itself a result.** Sizing a crop in proportion to its
  target manufactures a scale cue worth 11 accuracy points. Report a
  window-only baseline or part of the headline number is pipeline artefact.
* **THEMIS cannot contribute at 100 m/pixel.** Fusion *costs* 3.3 macro F1
  points, and a control in which the thermal branch is fed Gaussian noise costs
  nothing — so the penalty is the data, not the fusing. Only 3.2% of available
  observations fall in the pre-dawn window the physics needs, and the median
  skylight is 1.5 THEMIS pixels across.

## Repository guide

| Path | What it is |
|---|---|
| `data_overview.ipynb` | **Start here.** Sources, crop policy, splitting, what the data set contains and what the thermal branch can actually see. Builds `annotations.json`. |
| `training.ipynb` | Baselines, the cross-validated modality ablation, the noise control, and the tables the paper quotes. |
| `data/optical/` | Crop planning (`landform_crops.py`) and HiRISE product geometry from PDS ODE. |
| `data/thermal/` | THEMIS retrieval: site definition, footprint queries, window extraction. `thermal_data_collection.ipynb` drives it. |
| `model/` | Dataset (`landform_dataset.py`), architecture (`model.py`), baselines, training and cross-validation utilities. |

## Running it

```bash
pip install -r requirements.txt
```

Two inputs are not in the repository: the DeepLandforms tiles
(<https://zenodo.org/records/6818611>) go in
`data/optical/DeepLandforms_dataset/`, and the THEMIS windows are downloaded by
`data/thermal/thermal_data_collection.ipynb`. The MGC³ cave catalogue is
included.

Then, in order: `data_overview.ipynb` to build the annotation table,
`thermal_data_collection.ipynb` to fetch the windows, `training.ipynb` to report. 
Every run is cached as it finishes, so interrupting and re-running is free.
