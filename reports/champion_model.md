# Champion Model Selection

Best Parameters:

{'max_depth': 3, 'min_samples_leaf': 4, 'n_estimators': 100}

The tuned Random Forest model was selected as the champion model based on GridSearchCV performance.

| Model                 |      MAE |     RMSE |       R2 |
|:----------------------|---------:|---------:|---------:|
| Linear Regression     | 0.47206  | 0.580452 | 0.156588 |
| Default Random Forest | 0.494973 | 0.621607 | 0.03275  |
| Tuned Random Forest   | 0.465685 | 0.579257 | 0.160057 |