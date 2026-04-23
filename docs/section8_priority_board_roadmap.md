# Section 8 Priority Board and Roadmap

## Priority Board (GitHub Project cards)

| Card | Priority | Status | Depends on | Definition of done |
|---|---|---:|---|---|
| Freeze baseline dataset + config version | P0 | Ready | - | Dataset YAML + train config paths locked in repo |
| Baseline variance run (N seeds) | P0 | Ready | dataset freeze | `kfold_runs/variance_metrics.csv` generated |
| K-fold seed rebuild validation | P0 | Ready | dataset freeze | 5 seeded datasets + uniqueness report CSVs |
| K-fold train array run on HPC | P0 | Ready | seed rebuild | per-seed kfold metrics CSV outputs |
| Per-class/per-size HPC evaluation | P0 | Ready | baseline model + dataset | `per_class_size_metrics.csv` + counts/events CSV |
| Error taxonomy HPC evaluation | P0 | Ready | baseline model + dataset | `error_taxonomy_catalog.csv` + summary CSV |
| Local notebook: baseline + kfold summary | P0 | Ready | variance + kfold | notebook tables/plots for mAP50/F1/Recall/Precision/Epochs |
| Local notebook: per-class/per-size analysis | P0 | Ready | per-class/per-size eval | class x size plots + key weak bins identified |
| Local notebook: error taxonomy analysis | P0 | Ready | error taxonomy eval | ranked error types + top failure examples list |
| Student-teacher tiny-set prototype (3 sub-images) | P1 | Backlog | baseline metrics | tiny prototype train/infer complete |
| Pseudo-label quality gate (manual spot-check sheet) | P1 | Backlog | tiny prototype | acceptance thresholds defined + pass/fail result |
| Semi-supervised scale run | P1 | Backlog | quality gate | scaled run metrics + annotation time estimate |
| Unsupervised tiny prototype | P1 | Backlog | baseline metrics | tiny clustering/self-training run complete |
| Unsupervised scale run | P1 | Backlog | unsup tiny | scaled run metrics + compute cost |
| Comparative analysis (baseline vs semi vs unsup) | P1 | Backlog | both scale runs | one comparison table incl mIoU/F1/GPU-hours/annotation-hours |
| Stakeholder review pack (figures + actions) | P1 | Backlog | comparative analysis | meeting notes + agreed actions logged |
| Final tuned model selection | P2 | Backlog | stakeholder review | chosen model + rationale documented |
| Delivery report + defence asset pack | P2 | Backlog | final model | final report figures, scripts, reproducible commands |

## Roadmap (Section 8 aligned)

| Phase | Section 8 mapping | Goal | Exit criteria |
|---|---|---|---|
| Phase 1: Baseline lock-in | 1, 2 | Confirm baseline quality + stability | variance + kfold + qualitative failures complete |
| Phase 2: Progress reporting | 3, 4 | Mid-project checkpoint | written/oral material generated from baseline outputs |
| Phase 3: Prototype tracks | 5, 6 | Build tiny then scaled semi/unsup tracks | both tracks have reproducible runs + metrics |
| Phase 4: Comparative analysis | 7, 8 | Choose strongest tuned candidates | side-by-side comparison + tuned finalists |
| Phase 5: Stakeholder + final delivery | 9, 10 | Final decisions + submission | feedback integrated, final report/defence delivered |

## Suggested GitHub Project columns

1. Backlog
2. Ready
3. HPC Running
4. Local Analysis (.ipynb)
5. Review/Stakeholder
6. Done
