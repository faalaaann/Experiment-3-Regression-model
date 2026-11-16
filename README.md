# House Price Regression Model

A machine learning experiment that predicts house prices from basic property characteristics using a neural-network regression model built with TensorFlow and Keras.

## Project contents

- `EX3 Reggression model.ipynb` — Jupyter Notebook containing the complete workflow.
- `house_prices.csv` — Dataset with 300 house records.
- `README.md` — Project documentation.

## Dataset

The dataset contains the following columns:

| Column | Description |
| --- | --- |
| `area` | Property area |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `stories` | Number of stories |
| `age` | Property age |
| `price` | House price and prediction target |

## Methodology

1. Load the data with pandas.
2. Separate the input features from the target price.
3. Split the data into training and test sets using an 80/20 split.
4. Standardize the input features with `StandardScaler`.
5. Train a feed-forward neural network with two hidden layers:
   - 64 neurons with ReLU activation
   - 32 neurons with ReLU activation
   - 1 output neuron for the predicted price
6. Optimize the model with Adam using mean squared error loss and mean absolute error as a metric.
7. Train for 100 epochs and evaluate performance on the test set.
8. Compare actual and predicted prices using printed results and plots.

## Requirements

- Python 3.9+
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- scikit-learn
- TensorFlow

Install the dependencies with:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow jupyter
```

## Running the project

1. Clone the repository and enter the project directory:

   ```bash
   git clone https://github.com/faalaaann/Experiment-3-Regression-model.git
   cd Experiment-3-Regression-model
   ```

2. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

3. Open `EX3 Reggression model.ipynb` and run the cells in order.

The notebook displays the training and validation loss, test mean squared error, test mean absolute error, a sample of actual versus predicted prices, and an actual-versus-predicted scatter plot.

## Notes

Model results may vary slightly between runs because the neural network training process is stochastic. The dataset is intended for experimentation and educational use rather than production valuation.
