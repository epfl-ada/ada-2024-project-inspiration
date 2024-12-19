from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm



def regression(df, features, target):
    # Split features and target
    X = df[features]
    y = df[target]
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

    # we add constant column
    X_train_scaled = sm.add_constant(X_train_scaled)
        
    # Create and train the model
    model = sm.OLS(y_train, X_train_scaled).fit()
    
    # Scale and prepare test data
    X_test_scaled = scaler.transform(X_test.values)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    X_test_scaled = sm.add_constant(X_test_scaled)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    print(type(X_test))
    # Create results dataframe
    results_df = pd.DataFrame({
        #'X_test': X_test, # .values.tolist(),
        'y_test': y_test,
        'y_pred': y_pred
    })
    results_df = pd.concat([results_df, X_test], axis=1)
    if model.rsquared < 0.1:
        print('WARNING: R-squared is below 0.1')
        print('R-squared:', model.rsquared)
    
    return results_df, model


def plot_results(results_df,test_feature, feature_label, target_label):
    """
    Create a bar plot comparing y_test and y_pred
    """
    # for i in range(len(features)):
    #     plt.figure(figsize=(10, 6))
    #     plt.scatter(results_df.X_test[i], results_df['y_test'], color='blue', alpha=0.5, label='Actual')
    #     plt.scatter(results_df.X_test[i], results_df['y_pred'], color='orange', alpha=0.5, label='Predicted')
    #     plt.xlabel(feature)
    #     plt.ylabel(target)
    #     plt.title('Actual vs Predicted Values')
    #     plt.legend()
    #     plt.show()
    
    
    plt.figure(figsize=(10, 6))
    plt.bar(test_feature, results_df['y_test'], color='blue', alpha=0.5, label='Actual')
    plt.bar(test_feature, results_df['y_pred'], color='orange', alpha=0.5, label='Predicted')
    plt.xlabel(feature_label)
    plt.ylabel(target_label)
    plt.title('Actual vs Predicted target')
    plt.legend()
    plt.show()