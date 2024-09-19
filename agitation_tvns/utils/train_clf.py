#%%
import numpy as np
import matplotlib.pyplot as plt

# load the data
X = np.load('training_data_t.npy')
y = np.load('training_labels_t.npy')

plt.plot(X[2,:,:])


# %%
import numpy as np
from scipy.stats import skew
from scipy.signal import welch
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for trial in X:
            # Compute mean, variance, and skewness per channel
            mean_features = np.mean(trial, axis=0)
            var_features = np.var(trial, axis=0)
            skewness_features = skew(trial, axis=0)

            # Compute band power for alpha (8-13 Hz) as an example
            freqs, psd = welch(trial.T, fs=256, nperseg=256)  # fs is the sampling frequency
            alpha_band = np.logical_and(freqs >= 2, freqs <= 20)
            alpha_power = np.mean(psd[:, alpha_band], axis=0)

            # Concatenate all features
            trial_features = np.concatenate([ alpha_power])
            features.append(trial_features)
        return np.array(features)


# Create a pipeline
pipeline = Pipeline([
    ('feature_extractor', FeatureExtractor()),  # Extract features
    #('scaler', StandardScaler()),  # Normalize the data
    ('classifier', SVC(kernel='linear'))  # Use a linear SVM as the classifier
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Fit the pipeline on the training data
pipeline.fit(X, y)

# Predict on the test data
y_pred = pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Optionally, perform cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=3)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")

# Save the model
import joblib

# Assuming `pipeline` is your trained model
model_filename = 'eeg_classifier_model.pkl'

# Save the model to a file
joblib.dump(pipeline, model_filename)

print(f"Model saved to {model_filename}")
# %%
