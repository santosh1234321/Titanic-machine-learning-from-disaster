import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load data
train_data = pd.read_csv('titanic.csv')
test_data = pd.read_csv('test.csv')

# Stratified train-validation split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in split.split(train_data, train_data['Survived']):
    train_set = train_data.loc[train_idx]
    val_set = train_data.loc[val_idx]

# Custom transformers
class AgeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        imputer = SimpleImputer(strategy='mean')
        X['Age'] = imputer.fit_transform(X[['Age']])
        return X

class FeatureEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Fit on combined data to capture all categories
        combined_data = pd.concat([X, test_data], ignore_index=True)
        self.embarked_encoder = OneHotEncoder(handle_unknown='ignore')
        self.embarked_encoder.fit(combined_data[['Embarked']])
        self.sex_encoder = OneHotEncoder(handle_unknown='ignore')
        self.sex_encoder.fit(combined_data[['Sex']])
        return self

    def transform(self, X):
        # Transform separately using the fitted encoders
        embarked_enc = self.embarked_encoder.transform(X[['Embarked']]).toarray()
        embarked_cols = [f'Embarked_{cat}' for cat in self.embarked_encoder.categories_[0]]
        X[embarked_cols] = embarked_enc

        sex_enc = self.sex_encoder.transform(X[['Sex']]).toarray()
        sex_cols = [f'Sex_{cat}' for cat in self.sex_encoder.categories_[0]]
        X[sex_cols] = sex_enc

        return X

class FeatureDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(['Embarked', 'Name', 'Ticket', 'Cabin', 'Sex'], axis=1, errors='ignore')

# Pipeline setup
pipeline = Pipeline([
    ('age_imputer', AgeImputer()),
    ('feature_encoder', FeatureEncoder()),
    ('feature_dropper', FeatureDropper())
])

# Prepare training data
train_processed = pipeline.fit_transform(train_set)
X_train = train_processed.drop('Survived', axis=1)
y_train = train_processed['Survived']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Hyperparameter tuning for RandomForest
param_grid = {
    'n_estimators': [10, 100, 200, 500],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 3, 4]
}
clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy', return_train_score=True)
grid_search.fit(X_train_scaled, y_train)
best_clf = grid_search.best_estimator_

# Prepare and scale test data
test_processed = pipeline.transform(test_data)
# Fill missing values using ffill method
test_processed = test_processed.ffill()

# Ensure the test data has the same columns as the training data
test_cols = set(X_train.columns)
train_cols = set(test_processed.columns)

missing_in_test = list(test_cols - train_cols)
for col in missing_in_test:
    test_processed[col] = 0 # Or another appropriate default value

X_test_scaled = scaler.transform(test_processed[X_train.columns])

# Predict and save results
predictions = best_clf.predict(X_test_scaled)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('predictions.csv', index=False)
