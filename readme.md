# Diabetes Classification with K-Nearest Neighbors

## The Project
### Overview
This project implements a K-Nearest Neighbors (KNN) classifier from scratch to predict whether an individual has diabetes based on health indicators. It uses the public domain ["Diabetes Health Indicators Dataset" from Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset).

### Goal

The goal of this project is to determine if we can use KNN to predict whether an individual has diabetes based on health indicators. Specifically, the aim is to produce a high confidence predications based on the test data points.

### Reflection

#### Approachability
I was surprised by the simplicity of KNN in both concept and implementation. I'm not sure I had any specific expectations but, generally I was intimidated by this going into it. After reading through the beginner guide, I was able to implement the algorithm with relative ease. That was really encounraging.

#### Data Sources
Learning about kaggle and UCI, not to mention the other federal data sources was eye opening. I knew there was lots of data available. But, I'd never ventured into it before.

### Results
Using a test sample size of 23,682, the model achieved the following:

#### Test #1
- Distance metric: Euclidean
- N Nearest Neighbors: 10
- Accuracy: 84.98%

#### Test #2
- Distance metric: Manhattan
- N Nearest Neighbors: 10
- Accuracy: 85.05%

## Installation

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:delano-mic/ds5010-final.git
    cd ds5010-final
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the classifier:

```bash
python run.py
```

This will:
1.  Load the training and test data.
2.  Normalize the features using Min-Max scaling.
3.  "Train" the KNN model.
4.  Predict the classification for the test data points.
5.  Output the accuracy of the model.

### Run with a specific distance function

To change the distance function, use the `--distance_metric` flag:

```bash
python run.py --distance_metric euclidean
```

### Run with a specific number of neighbors

To change the number of neighbors, use the `--n_neighbors` flag:

```bash
python run.py --n_neighbors 5
```

## Running Tests

To run the unit tests for the distance metrics and classifier:

```bash
pytest tests
```
