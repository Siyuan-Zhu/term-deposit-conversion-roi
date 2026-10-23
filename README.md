# term-deposit-conversion-roi

A model that tells a bank's outbound call team which prospects to phone to sell term deposits, and shows the call list earns about 1.7 times the net value of calling everyone. It also shows where that result stops holding.

![Gain curve and net value by threshold](assets/figure1_gain_and_nvi_curve.png)

## The problem

A bank's outbound team phones prospects to sell term deposits. About 11% say yes. Every call costs agent time, and most calls end in a no. Whom should the team call?

Two traps make the answer look better than it is:

- The `duration` field. The dataset records how long each call lasted. Long calls usually end in a yes, so a model with `duration` looks excellent. But call length is only known after the call, so it cannot help you decide whom to dial.
- Picking a cutoff by feel. "Call everyone above 50%" sounds reasonable. It calls very few people and leaves most of the buyers uncalled.

## The solution

`python src/run_pipeline.py` does the following:

1. `src/data.py` downloads the UCI Bank Marketing file (41,188 calls, 2008 to 2010) and checks its SHA-256 and row count. A mismatch stops the run.
2. `src/model.py` blocks `duration` and `campaign` before fitting. It then trains a logistic regression on pre-call information only (age, job, past outcome, month, and macro-economic indicators) and calibrates its scores.
3. `src/policy.py` turns calibrated probabilities into call lists and prices each list.
4. `src/figure.py` draws the gain curve and net-value chart above.

The data is public: the UCI Bank Marketing dataset (id 222, `bank-additional-full.csv`), released under CC BY 4.0. Attribution is in [DATA.md](DATA.md). The file is downloaded at run time and is not committed.

## Method

- Leakage gate. `PreCallFeatureGuard` raises an error if `duration` reaches the model. It also blocks `campaign`, the number of contacts including the current call, which is not known before dialling. The gate stops the leak from coming back; the ROC-AUC rows below show how large it is.
- Logistic regression. Fitted by IRLS (a standard Newton-style solver), with L2 regularisation and extra weight on the rare "yes" class. It is simple, fast, and its coefficients can be read.
- Calibration (Platt scaling). Raw model scores are not probabilities: a score of 0.20 does not mean 20% of those people subscribe. Platt scaling fits a small correction on out-of-fold scores, so that a calibrated 0.20 does mean about 20%. The call rule needs real probabilities.
- Expected-value threshold. A call is worth making when the chance of a yes times the value of a yes exceeds the cost of the call. With cost 1 and value 12.5, that is p >= 1/12.5 = 0.08. The threshold follows from the ratio. It is not tuned on the test set.
- Comparison policies. Call everyone, a naive cutoff at 0.50, the break-even cutoff at 0.08, and a fixed capacity of the top 30% by score.

The 12.5:1 value-to-cost ratio is an assumption, not a measured figure: one conversion is worth 12.5 calls. Change `V_CONVERT` in `src/policy.py` and the threshold, call lists and ROI table all move with it.

## Findings

Random 70/30 split: 28,832 training rows, 12,356 test rows, 11.64% subscribe.

| Check | Result |
| :--- | :--- |
| Test ROC-AUC, pre-call features | 0.7981 (PR-AUC 0.4699) |
| Test ROC-AUC with `duration` added | 0.9362 |
| Expected calibration error, before and after Platt | 8.42% to 1.07% |
| Brier score, before and after Platt | 0.0886 to 0.0797 |

The gap between 0.7981 and 0.9362 is what the leakage gate removes. The lower score is the usable one.

| Policy | Calls | Capture | Precision | Net value | ROI % | Net-value index |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Call everyone | 12,356 | 100.0% | 11.64% | 5,619 | 45.5 | 100 |
| Naive cutoff, p >= 0.50 | 478 | 22.8% | 68.62% | 3,622 | 757.7 | 64 |
| Break-even, p >= 0.08 | 3,659 | 72.3% | 28.42% | 9,341 | 255.3 | 166 |
| Top 30% by p | 3,707 | 72.5% | 28.11% | 9,318 | 251.4 | 166 |

Net value is conversions times 12.5 minus calls. ROI % is net value divided by calls. The net-value index sets calling everyone to 100. The two measures rank the policies differently. The naive cutoff has the highest ROI because it calls few people, and the lowest net value because it leaves 77% of the buyers uncalled. The bank should maximise net value, which the break-even list does: it calls 30% of prospects and reaches 72% of buyers.

### The time-ordered split

The file is in date order, so training on the first 70% of rows and testing on the last 30% mimics deploying the model on future calls.

| | Random split | Time-ordered split |
| :--- | ---: | ---: |
| Test base rate | 11.64% | 24.55% |
| ROC-AUC, pre-call | 0.7981 | 0.6607 |
| ECE after Platt | 1.07% | 12.33% |
| Break-even net-value index | 166 | 79 |

On later data the model ranks worse, the calibration fails, and the break-even list does worse than calling everyone. The later period converts at more than twice the earlier rate, so calling everyone is already profitable there and the 0.08 cutoff drops 30% of the buyers. The macro columns (`euribor3m`, `nr.employed`) tie each row to a date, which is why a random split flatters the model. Read the random-split numbers as the best case.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest
python src/run_pipeline.py   # downloads and verifies the data, writes data/processed/ and the figure
pytest -q                    # 9 tests; most need the download
```

## Scope and limits

- One public dataset from a Portuguese bank, 2008 to 2010. The model has not been tested on other banks or periods.
- Cost and value are relative units (one call = 1, one conversion = 12.5), not currency. The ROI table scales with that ratio.
- The model is a plain logistic regression, chosen so its coefficients can be read.
- The model needs recalibrating on recent data before any live use. The time-ordered split shows why.
- `data/` is gitignored. Only the figure is committed. CI runs ruff and pytest on Python 3.11 to 3.13.

MIT licensed.
