This is a machine learning project to predict an individual's obesity level from demographic, dietary, and lifestyle factors, built in Python.

Dataset
-2,111 records covering demographic info (Gender, Age, Height, Weight), family history of overweight, dietary habits (frequency of high-calorie food consumption, vegetable consumption, number of main meals, snacking), lifestyle factors (smoking, water intake, calorie monitoring, physical activity, tech usage, alcohol intake, transportation mode), and the target variable.
-The target, Obesity_Level, has 7 classes: Insufficient_Weight, Normal_Weight, Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II, and Obesity_Type_III.

Exploratory Data Analysis
- Explored the target class distribution, the relationship between Height and Weight (colored by obesity level), how Age and Weight vary across obesity levels, and how categorical factors like family history, high-calorie food consumption, and transportation mode relate to obesity level. Also examined a correlation heatmap of numeric features and how physical activity and tech usage vary across classes.
- Outliers: checked using the IQR method. Only Age and Weight flagged a small number of values, both explainable as genuine variation rather than data errors: a wider weight range is expected and meaningful in an obesity dataset. No_of_mainsPerDay, Physicals, and Tech_Usage are ordinal/discrete scales, so IQR-based flags on those columns weren't meaningful and were disregarded. No outliers were removed.

Feature Engineering
- Created BMI (Weight / Height), an Unhealthy_Score combining high-calorie food consumption, smoking, family history, and low physical activity into one composite score, an Age_Group bucket (Teen / Young Adult / Adult / Older), a Hydration_per_Meal ratio, an Active_Transport flag for walking/biking, and a Sedentary_Ratio comparing tech usage to physical activity.

Preprocessing
- Categorical columns (including the engineered Age_Group) were one-hot encoded, and the target was encoded with pd.factorize(). Data was split 80/20 with stratification, and numeric features were scaled with StandardScaler, fit on the training set only.
- An ablation test showed Gender_Male wasn't actually predictive, removing it maintained or slightly improved accuracy across models, so it was dropped from the final feature set.

Models
- Trained Logistic Regression, Decision Tree, Random Forest, XGBoost, and SVM, then combined the top performers (Random Forest, XGBoost, Decision Tree) into a Stacking Classifier with a Logistic Regression meta-learner.
- Logistic Regression came in weakest (91% accuracy),it confirmed the relationship between features and obesity level isn't purely linear. Decision Tree, Random Forest, and XGBoost all performed similarly well, in the 97-98% range. The Stacking Classifier came out on top, at 98.8% test accuracy and 98.2% cross-validation accuracy, with the most balanced performance across all seven classes.
- 5-fold cross-validation was run for every model to confirm results were stable and not just a lucky train/test split.

Best Model
- The Stacking Classifier is the strongest overall- best test accuracy, best macro F1, and consistent performance across all classes. Random Forest came very close on cross-validation accuracy alone, and at a fraction of the computational cost, so it was used for interpretability work instead of the heavier stacked ensemble.
- The hardest classes to separate were consistently Obesity_Type_I and Obesity_Type_II, since these are adjacent severity tiers with naturally overlapping feature profiles.

Feature Importance
- BMI was by far the strongest predictor, followed by Weight. Frequency of vegetable consumption was a genuinely interesting finding, it barely showed any linear correlation with the target, but ranked as a top driver in feature importance, suggesting a non-linear or interaction effect that tree-based models pick up on but simple correlation misses. Height and Age followed, with the engineered features (Unhealthy_Score, Hydration_per_Meal, Sedentary_Ratio) contributing more modestly.

Limitations
- BMI and Weight are closely tied to how obesity categories are typically defined, so their dominance as predictors is partly definitional rather than a novel discovery. Adjacent obesity classes remain harder to distinguish, which fits with them sitting close together on a continuous underlying scale. Cross-validation was run on pre-scaled features rather than within a fully leakage-free pipeline — given how well-separated the classes already are, this is expected to have negligible impact on the reported scores.

Tech Stack
- Python,numpy, pandas, scikit-learn, XGBoost, seaborn, matplotlib
