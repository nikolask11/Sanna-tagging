# Final Official-Test Report

- Selected budget: 800
- Winning arm: slovakbert-czech-lapt-slovak-upos
- Ensemble: unweighted-arithmetic-mean-probabilities

## Frozen checkpoints

- Seed 0: `a9b36a34f6ead22852b776f541e639e70da9be0a16360993db361debc01d5388`
- Seed 1: `0b644e2c8d422828b5e91783d6fa5bd1bdb46c732d3bb1140f904b5e191bcdcc`
- Seed 2: `14c6f345c4db6ea6f2b4bdcdcb91fae314e903e5049da95ab3d381c130fd6202`

## Test metrics

| System | Token accuracy | Sentence ≥98% | Exact match |
|---|---:|---:|---:|
| Seed 0 | 0.977175 | 0.735771 | 0.733145 |
| Seed 1 | 0.977447 | 0.738396 | 0.736365 |
| Seed 2 | 0.976937 | 0.735077 | 0.732600 |
| Frozen 3-seed ensemble | 0.978944 | 0.754644 | 0.751771 |

## Ensemble calibration

```json
{
  "brier_score": 0.13878490193313373,
  "log_loss": 0.4289893703376134,
  "sentence_count": 20187,
  "fixed_risks": [
    {
      "accepted": 0,
      "accepted_high_quality": 0,
      "accepted_high_quality_yield": 0.0,
      "coverage": 0.0,
      "empirical_failure_risk": null,
      "target_failure_risk": 0.01
    },
    {
      "accepted": 0,
      "accepted_high_quality": 0,
      "accepted_high_quality_yield": 0.0,
      "coverage": 0.0,
      "empirical_failure_risk": null,
      "target_failure_risk": 0.02
    },
    {
      "accepted": 266,
      "accepted_high_quality": 261,
      "accepted_high_quality_yield": 0.012929112795363353,
      "coverage": 0.013176796948531234,
      "empirical_failure_risk": 0.018796992481203006,
      "target_failure_risk": 0.05
    }
  ],
  "risk_coverage": "20187 rows — see risk_coverage_summary.csv for the digest"
}
```

Selection, checkpoints, ensemble membership and weights, and calibration were not changed after selection.lock.json was frozen.

---

## Provenance

Reproduced verbatim from the `finalize-test` stage output of Kaggle notebook
`nikolaskallweit11/notebook55426794b3`, version 7 (`scriptVersionId=341764921`),
which ran successfully in 1114.8 s on GPU T4 x2.

- Run name: `cs-rerun-v2`
- Run fingerprint: `30a9fbaa373488ecfeec`
- Code commit: `abb04f9a9f3f0d6c44fab13a0eba8c1ce4613579`
- Config: `configs/cs_kaggle_v2.yaml`
- Official gold: `cs_pdtc-ud-test.conllu`, 39 742 361 bytes,
  SHA-256 `f5c1a7ee2575e7e42fbedf958f2d2f55af51d60a8c770643a514e627f04fa97a`
  (matches the pinned value in the config; verified before dispatch)
- Gold source: UD_Czech-PDTC at revision `d24709eac1f77cf5eef9e807080270955cba9e00`

The one and only deviation from the on-Kaggle `final/REPORT_FINAL.md` is that the
20 187-row `risk_coverage` array inside the calibration JSON has been replaced by a
pointer; every other line is byte-identical to what the stage printed. The complete
array remains in the Kaggle version-7 output under
`sanna-v2/runs/cs-rerun-v2/30a9fbaa373488ecfeec/final/`.
