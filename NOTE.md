11th October, 2026


“Imputer = mean; the six leading imputers were statistically indistinguishable (Lesson 1, RF and HGB); mean chosen as the simplest option that works with every later step (SMOTE needs no blanks) and is PyCaret’s default.” Median is equally defensible. Mean won for HGB, and the rule picked “leave blanks” for RF, but that can’t be used with SMOTE later.

Report hot deck and zero fill in your paper as “tested, slightly worse, not significant after correction”.








(env_main) sabbir@sabbir-IdeaPad-Slim-3-15IRH8:~/research/v3$ cd ~/research/v3
python compare_tools.py
splits compared: 10
AutoGluon (v2) test AUC: 0.6942 +/- 0.0192
PyCaret   (v3) test AUC: 0.6959 +/- 0.0188
mean difference (PyCaret - AutoGluon): +0.0017   95% CI [-0.0040, +0.0069]
corrected resampled t-test p-value: 0.760
Reading: if the CI includes 0 and p > 0.05, the two tools are statistically tied.


Assalamu Alaikum Ma'am, 

From the imputers I tested, mean imputation did well and I implemented this. I tested 9 imbalance technique out of which none performed well in AUC and accuracy.
SMOTE helped F1, but generally reduced ordinary accuracy and did not improve AUC
 PyCaret results show:
- Without scaling: AUC = 0.6959
- With MinMax scaling: AUC = 0.6949
AutoGluon showed a similar pattern: AUC was 0.6814 without scaling and 0.6770 with scaling.

I compared and executed the Autogluon and Pycaret results after verified preprocessing and analysis. Their mean AUC difference was only 0.0017, with p = 0.760. Which means not such huge difference. And the prediction performance was moderate but far from perfect because Mean AUC was around 0.69–0.70. 

My other findings from this experiment are: 

PyCaret selected CatBoost 31 times and Quadratic Discriminant Analysis 9 times.
AutoGluon selected NeuralNetTorch 25 times, NeuralNetFastAI 5 times, and several other models in the remaining runs.

CatBoost is a reasonable candidate for hyperparameter tuning experiment and XAI because PyCaret selected it frequently and due to the followings:
1. A standalone model with at least four tunable hyperparameters
2. CatBoost won most PyCaret runs
3. QDA can't be selected as it supports one hyperparameter only
4. Autogluon's Neural Network isn't standalone.


Ma'am I would like to proceed with the hyperparameter tuning and XAI part using CatBoost.

Ma'am for Smotified GAV and for Ganfied SMOTE, I went through few papers and found that these are good for extremely imbalanced data, whereas ours is mild(61/39). Another 2025 study abstract shows that SMOTE outperformed the GAN-based approaches across classifiers and datasets.

To verify in our Datasets, i executed and compared other imbalancing technique and found that Smotified GAN and GANified SMOTE performed low compared to SMOTE as a paper suggests and also low compared to the not implemented imbalance technique data.

