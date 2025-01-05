import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np

class BaseballMLPipeline:
    def __init__(self, organzed_baseball_data: pd.DataFrame, model) -> None:
        self.data = organzed_baseball_data 
        self.model = model
        self.data.sort_values(by = 'Season', ascending=False, inplace=True)

        train_years = self.data['Season'].unique()[1:]
        test_years = self.data['Season'].unique()[0]

        train_data = self.data[self.data['Season'].isin(train_years)]
        self.X_train = train_data.drop(columns=['wRC+_next', 'Season']).to_numpy()
        self.y_train = train_data['wRC+_next'].to_numpy()

        test_data = self.data[self.data['Season'] == test_years]
        self.X_test = test_data.drop(columns=['wRC+_next', 'Season']).to_numpy()
        self.y_test = test_data['wRC+_next'].to_numpy()

    def get_train_data(self):
        return self.X_train, self.y_train
    
    def get_test_data(self):
        return self.X_test, self.y_test
    
    def get_test_data_type(self):
        return f"self.X_test type: {type(self.X_test)}, self.y_test type: {type(self.y_test)}"
    
    def get_model(self):
        return self.model
    
    def __str__(self):
        return f"Training data: {self.X_train.shape}, {self.y_train.shape} \n" + \
        f"Testing data: {self.X_test.shape}, {self.y_test.shape} \n" + \
        f"Model {self.model}"

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

    def get_prediction(self):
        return self.y_pred

    def evaluate_model(self):  
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, self.y_pred)

        print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

    def plot_model(self):
        regr = LinearRegression()
        y_test_reshaped = self.y_test.reshape(-1, 1)
        regr.fit(y_test_reshaped, self.y_pred)
        best_fit_y = regr.predict(y_test_reshaped)

        test_pred = pd.DataFrame({"wRC+ Predicted": self.y_pred, "wRC+ Actual": self.y_test, })
        test_pred_plot = px.scatter(test_pred, x = "wRC+ Predicted", y = "wRC+ Actual")

        test_pred_plot.add_trace(go.Scatter(x=self.y_test, 
                                            y=best_fit_y, 
                                            mode='lines', 
                                            name='Line of Best Fit', 
                                            line=dict(dash = 'dot', width=2, color='red')))

        test_pred_plot.show()
