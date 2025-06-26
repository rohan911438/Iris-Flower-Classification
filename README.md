# 🌸 Iris Flower Classification

This project uses supervised machine learning to classify iris flowers into three species — *Setosa*, *Versicolor*, and *Virginica* — based on the dimensions of their sepals and petals. The model is built and evaluated using scikit-learn and visualized using popular Python libraries.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Model Workflow](#-model-workflow)
- [Visualizations](#-visualizations)
- [Results](#-results)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 🔍 Overview

The goal of this project is to predict the species of an iris flower using four features:
- Sepal length
- Sepal width
- Petal length
- Petal width

Various classification models are trained and evaluated, with visual insights and performance metrics.

---

## 📊 Dataset

- **Source**: Built-in dataset from `sklearn.datasets.load_iris()`
- **Samples**: 150
- **Classes**: Setosa, Versicolor, Virginica
- **Features**:
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)

---

## 🛠 Technologies Used

- Python 3.x
- Jupyter Notebook
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

---

## 🗂 Project Structure

```bash
Iris_Flower_Classification/
│
├── Iris_Flower_Classification.ipynb   # Main notebook
├── Dataset URL.txt                    # Dataset reference
├── README.md                          # Project documentation (this file)
└── requirements.txt                   # Dependencies (to be created)

```

## 💾 Installation

Clone the repository:

```bash
git clone https://github.com/rohan911438/Iris-Flower-Classification.git
cd Iris-Flower-Classification

```

Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```
Install dependencies:
```bash
pip install -r requirements.txt
```
## ▶️ How to Run

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the `Iris_Flower_Classification.ipynb` file.

Run the notebook cell by cell to follow the full analysis, model training, and evaluation.

---

## 🔄 Model Workflow

1. Load Dataset  
2. Explore Data: shape, head, info, statistical summary  
3. Visualize: scatter plots, histograms, heatmaps  
4. Preprocess: feature/label split, train-test split  
5. Train Models:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)  
6. Evaluate: accuracy score, confusion matrix

---

## 📈 Visualizations

The notebook includes multiple visualizations:
- Pairplots of the iris dataset
- Correlation heatmap
- Histograms of features
- Confusion matrix plots

---

## 🧪 Results

All classifiers achieved high accuracy. Below is a sample result (values may vary):

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 96.7%    |
| KNN Classifier      | 97.3%    |
| SVM                 | 98.0%    |

✅ **Best Performing Model**: *Support Vector Machine (SVM)*

---

## 🚀 Future Improvements

- Add more classifiers and tune hyperparameters (e.g. with GridSearchCV)
- Create an interactive web app using Streamlit or Flask
- Add cross-validation and classification reports
- Deploy the model online

---

## 📄 License

This project is open-source and free to use under the [MIT License](https://opensource.org/licenses/MIT).

---

## 🙌 Acknowledgements

- Dataset: UCI Machine Learning Repository via `scikit-learn`
- Author: [@rohan911438](https://github.com/rohan911438)




