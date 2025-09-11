import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for server environments

def find_y(m,x,b):
    return (m*x)+b

def find_mean(data):
    return sum(data) / len(data)
    
def find_slope(x_mean,y_mean,x_data,y_data):
    numerator = sum([(x-x_mean)*(y-y_mean) for x,y in zip(x_data,y_data)])
    denumerator = sum([(x-x_mean)**2 for x in x_data])
    return numerator/denumerator

def find_intercept(m,x_mean,y_mean):
    return(y_mean - (m*x_mean))

def read_csv(file_name,header = True):
    with open(file_name,"r") as file:
        lines = file.readlines()
        x_feature = []
        y_target = []
        first_header = int(header)
        for line in lines[first_header:]:
            x,y = line.split(",")
            x = x.strip()
            y = y.strip()
            try:
                x_feature.append(float(x))
                y_target.append(float(y))
            except:
                continue
        return x_feature,y_target
def predict(m, X, b):
    return [(m * val) + b for val in X]


def data_split(X,Y,split):
    train_part = int(len(X) * split)
    return X[:train_part],Y[:train_part],X[train_part:],Y[train_part:]


class SimpleLinearRegression:
    def __init__(self):
        self.m =None
        self.b = None
        self.yp = None
    def fit(self,X,y):
        x_mean  = find_mean(X)
        y_mean = find_mean(y)
        self.m = find_slope(x_mean,y_mean,X,y)
        self.b = find_intercept(self.m,x_mean,y_mean)
    def predict(self,X):
        self.yp =  [((self.m* x) + self.b) for x in X ]
        return self.yp
    def validate(self, x):
    # ✅ Return numeric prediction, not a string
        return (float(self.m) * float(x)) + float(self.b)

    def equation(self):
    # ✅ Keep equation for display as a string
        return f"y = {round(self.m, 2)}x + {round(self.b, 2)}"

    def score(self, X, y):
        # If predictions not available, generate them
        if self.yp is None or len(self.yp) != len(X):
            self.predict(X)
        ssr = sum((y_i - y_p) ** 2 for y_i, y_p in zip(y, self.yp))
        y_mean = find_mean(y)
        sst = sum((y_i - y_mean) ** 2 for y_i in y)
        return 0 if sst == 0 else 1 - (ssr / sst)

    def mse(self, X, y):
        # If predictions not available, generate them
        if self.yp is None or len(self.yp) != len(X):
            self.predict(X)
        sum_m = sum((y_i - y_p) ** 2 for y_i, y_p in zip(y, self.yp))
        return sum_m / len(X)
    def rmse(self,x,y):
        mse=  self.mse(x,y)
        return mse **0.5
    def mae(self, X, y):
    # If predictions not available, generate them
        if self.yp is None or len(self.yp) != len(X):
            self.predict(X)
        sum_of = sum(abs(y1 - y2) for y1, y2 in zip(y, self.yp))
        return sum_of / len(y)
    def mape(self, X, y):
    # If predictions not available, generate them
        if self.yp is None or len(self.yp) != len(X):
            self.predict(X)
    
    # Avoid division by zero by skipping zero y-values
        percentage_errors = [
        abs((y1 - y2) / y1) * 100
        for y1, y2 in zip(y, self.yp) if y1 != 0]
    
        return sum(percentage_errors) / len(percentage_errors)

    def evaluate(self, X, y, dataset_name="Dataset"):
        if self.yp is None or len(self.yp) != len(X):
            self.predict(X)

        r2 = self.score(X, y)
        mse = self.mse(X, y)
        rmse = self.rmse(X, y)
        mae = self.mae(X, y)
        mape = self.mape(X, y)
        return {                        "r_score":round(r2,2),
                       "mse":round(mse,2),
                       "rmse":round(rmse,2),
                       "mae":round(mae,2),
                       "mape":round(mape,2)


        }


   #     print(f"\nEvaluation on {dataset_name}:")
#        print("-" * 40)
 #       print(f"R² Score      : {r2:.2f}")
  #      print(f"MSE           : {mse:.2f}")
   #     print(f"RMSE          : {rmse:.2f}")
    #    print(f"MAE           : {mae:.2f}")
     #   print(f"MAPE (%)      : {mape:.2f}")
      #  print("-" * 40)

    def visualize(self, X, y, filename="regression_plot.png"):
    # Ensure predictions are available
        y_pred = self.predict(X)

    # Plot actual data points
        plt.scatter(X, y, color="blue", label="Actual Data", alpha=0.6)

    # Plot regression line
        plt.plot(X, y_pred, color="red", label="Regression Line")

    # Add labels and title
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Simple Linear Regression: Actual vs Predicted")
        plt.legend()
        plt.grid(True)

    # Save figure
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved as {filename}")

if __name__ == "__main__":
    X,y = read_csv("/Users/haseebsagheer/Documents/Python Learning/Linear-Regression-From-Scratch/Dataset/simple_linear_dataset.csv",header = False)

    x_train,y_train, x_val,y_val = data_split(X,y,split=0.9)
    model = SimpleLinearRegression()
    model.fit(x_train,y_train)
    model.predict(x_val)

# Evaluate on training set
    model.evaluate(x_train, y_train, "Training Set")

# Evaluate on validation set
    model.evaluate(x_val, y_val, "Validation Set")
    model.visualize(x_val, y_val, "Outputs/regression_plot.png")




