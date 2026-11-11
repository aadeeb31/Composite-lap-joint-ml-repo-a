# Machine Learning for Adhesively Bonded Composite Lap Joints

Data and code for the study of 74 published studies (1,516 records) on adhesively bonded composite lap joints: 22 models in 8 families, six tasks (maximum load, lap-shear strength and failure mode, each for CFRP–CFRP and for composite–composite joints).

## Structure

```
data/raw          public workbook and source list (74 studies)
data/processed    six model-ready tables
data/data_dictionary.csv
src/zoo.py        model zoo, pipelines, grouped folds
src/paper_figures figure code used for the paper
notebooks/        run in numerical order
results/          tables, predictions, SHAP, PDP, nested validation
results/paper     stored inputs of the paper figures
figures/          all paper figures (png and pdf)
```

## Notebooks

| Notebook | Content |
|---|---|
| 00_Data_Preparation | workbook to six model-ready tables |
| 01–06 | one notebook per task: tuning, test scores, parity or confusion plots, permutation importance, SHAP, interactions, partial dependence |
| 07_Nested_Grouped_Validation | whole studies held out, repeated |
| 08_Statistics_on_Unseen_Studies | Friedman, Nemenyi, paired bootstrap, heavy-tail metrics, importance stability |
| 09_Prediction_Intervals_and_Calibration | conformal and Gaussian-process intervals, classifier reliability |
| 10_Sensitivity_to_Row_Choice | row subsets |
| 11_Paper_Figures | all figures of the paper |
| 12_Robustness_Checks | joint types and calculated targets, removal of study 64, one-hot encoding of the categorical inputs, baselines and measured-only scores |

## Run

```
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 00_Data_Preparation.ipynb
```

Run 01–06 next (they write `results/models/*.joblib`, needed by 09), then 07–11. Notebook 07 takes hours; the stored `results/nested/*.pkl` let 08 run without it.

## Reference

M. A. Chowdhury, J. Iftekhar, G. M. T. Mahmud, A. Baral, M. A. Al Bari. A diverse and explainable machine learning framework for load, strength and failure-mode prediction of adhesively bonded composite lap joints.
