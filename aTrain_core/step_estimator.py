import numpy as np

class QuadraticRegressionModel:
    def __init__(self):
        self.coefficients_segmentation = None
        self.coefficients_embedding = None

    def train(self, audio_lengths, segmentation_steps, embedding_steps):
        A_segmentation = np.vstack([audio_lengths**2, audio_lengths, np.ones(len(audio_lengths))]).T
        A_embedding = np.vstack([audio_lengths**2, audio_lengths, np.ones(len(audio_lengths))]).T

        self.coefficients_segmentation, _, _, _ = np.linalg.lstsq(A_segmentation, segmentation_steps, rcond=None)
        self.coefficients_embedding, _, _, _ = np.linalg.lstsq(A_embedding, embedding_steps, rcond=None)

    def predict_segmentation(self, length):
        if self.coefficients_segmentation is None:
            raise ValueError("Model has not been trained yet.")
        return self._predict(length, self.coefficients_segmentation)

    def predict_embedding(self, length):
        if self.coefficients_embedding is None:
            raise ValueError("Model has not been trained yet.")
        return self._predict(length, self.coefficients_embedding)

    def _predict(self, length, coefficients):
        a, b, c = coefficients
        return a * length**2 + b * length + c



