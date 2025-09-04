
# Simple Linear Regression from Scratch

This project implements **Simple Linear Regression** in Python **without using any machine learning libraries**.  
Everything is done using pure Python to understand the math behind linear regression.

---

## 📂 Project Overview
- Implemented **Linear Regression** using the **Closed-Form Equation** (No Gradient Descent).  
- Built a custom `SimpleLinearRegression` class with methods:
  - `fit(X, y)` → Train model on dataset.
  - `predict(X)` → Predict target values for new data.
  - `score(X, y)` → R² score for model evaluation.
  - `mse(X, y)` → Mean Squared Error (MSE).
  - `rmse(X, y)` → Root Mean Squared Error (RMSE).
- Implemented a **train/test split** for realistic evaluation.

---

## 🧠 Key Learnings

- **How to derive and compute:**
- **Slope (m):**  
  $m = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$

- **Intercept (b):**  
  $b = \bar{y} - m \bar{x}$


- **Evaluation Metrics:**
  - $R^2 = 1 - \frac{SSR}{SST}$
  - $MSE = \frac{1}{n}\sum(y_i - \hat{y_i})^2$
  - $RMSE = \sqrt{MSE}$


---

## 📊 Results on Sample Dataset
| Metric       | Training Data | Validation Data |
|--------------|---------------|------------------|
| R² Score      | 0.99          | 0.70            |
| MSE           | 22.09         | 17.18           |
| RMSE          | 4.70          | 4.15            |

The drop on validation metrics shows **real-world performance** on unseen data.

---

## ⚙️ How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/engrhaseebsagheer/Linear-Regression-From-Scratch
   cd simple-linear-regression
   ```
2. Run the Python script:
   ```bash
   python LinearRegression.py
   ```
3. Add your dataset in CSV format:
   ```
   X,y
   1,12
   2,15
   3,20
   ...
   ```

---

## 📦 Dependencies
- Python 3.x  

No external libraries required.

---

## 🔍 Next Steps
- Add **Mean Absolute Error (MAE)** metric.  
- Implement **Multiple Linear Regression** (multiple features).  
- Add **Gradient Descent** for optimization.  
- Create **visualizations** for regression line and errors.

---

## ✍️ Author
- **Haseeb Sagheer** — Learning and building ML concepts from scratch.  
