import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Check for TensorFlow availability
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    tensorflow_available = True
except ImportError:
    print("Warning: TensorFlow not found. Skipping RNN (LSTM) model. Install TensorFlow with 'pip install tensorflow' to include RNN.")
    tensorflow_available = False

# Verify required libraries
required_libraries = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn']
missing_libraries = []
for lib in required_libraries:
    try:
        __import__(lib)
    except ImportError:
        missing_libraries.append(lib)

if missing_libraries:
    raise ImportError(f"Missing required libraries: {', '.join(missing_libraries)}. Install with 'pip install {' '.join(missing_libraries)}'")

# Configuration
input_file = r'C:\Users\Kompiuteris\Desktop\IM AI WITH THE BRAIDS\4 LABORAS HEHE\cleaned_dataset.csv'  # From clean_missing_data.py
output_folder = r'C:\Users\Kompiuteris\Desktop\IM AI WITH THE BRAIDS\4 LABORAS HEHE\4LaboroRezultatai2'
os.makedirs(output_folder, exist_ok=True)

# Define output file paths for preprocessing
output_encoded = os.path.join(output_folder, 'encoded_dataset.csv')
output_scaled = os.path.join(output_folder, 'scaled_dataset.csv')
output_outliers_removed = os.path.join(output_folder, 'outliers_removed_dataset.csv')
output_distribution_plot = os.path.join(output_folder, 'data_distribution.png')
output_correlation_plot = os.path.join(output_folder, 'correlation_matrix.png')

# Define output file paths for regression modeling
output_comparison_table = os.path.join(output_folder, 'model_comparison.csv')
output_plots = {
    'linear': {
        'pred_actual': os.path.join(output_folder, 'linear_pred_actual.png'),
        'residuals': os.path.join(output_folder, 'linear_residuals.png'),
        'learning_curve': os.path.join(output_folder, 'linear_learning_curve.png')
    },
    'poly': {
        'pred_actual': os.path.join(output_folder, 'poly_pred_actual.png'),
        'residuals': os.path.join(output_folder, 'poly_residuals.png'),
        'learning_curve': os.path.join(output_folder, 'poly_learning_curve.png')
    },
    'dt': {
        'pred_actual': os.path.join(output_folder, 'dt_pred_actual.png'),
        'residuals': os.path.join(output_folder, 'dt_residuals.png'),
        'learning_curve': os.path.join(output_folder, 'dt_learning_curve.png'),
        'feature_importance': os.path.join(output_folder, 'dt_feature_importance.png')
    },
    'rf': {
        'pred_actual': os.path.join(output_folder, 'rf_pred_actual.png'),
        'residuals': os.path.join(output_folder, 'rf_residuals.png'),
        'learning_curve': os.path.join(output_folder, 'rf_learning_curve.png'),
        'feature_importance': os.path.join(output_folder, 'rf_feature_importance.png')
    }
}

if tensorflow_available:
    output_plots['rnn'] = {
        'pred_actual': os.path.join(output_folder, 'rnn_pred_actual.png'),
        'residuals': os.path.join(output_folder, 'rnn_residuals.png'),
        'learning_curve': os.path.join(output_folder, 'rnn_learning_curve.png')
    }

try:
    # Print current working directory for debugging
    print(f"Current working directory: {os.getcwd()}")
    print(f"All output files will be saved to: {os.path.abspath(output_folder)}")

    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_csv(input_file, parse_dates=['last_review'], dayfirst=True)

    # Step 2: Convert Categorical Variables to Numerical
    print("\nStep 2: Encoding categorical variables...")
    categorical_columns = ['neighbourhood_group', 'neighbourhood', 'room_type']  # Extensible list
    label_encoders = {}

    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"Encoded {col}. Classes: {le.classes_}")
        else:
            print(f"Column {col} not found in dataset.")

    df.to_csv(output_encoded, index=False, date_format='%Y-%m-%d')
    print(f"Encoded dataset saved to {os.path.abspath(output_encoded)}")

    # Step 3: Scale Numeric Attributes
    print("\nStep 3: Scaling numeric attributes...")
    exclude_columns = ['id', 'host_id', 'neighbourhood_group', 'neighbourhood', 'room_type', 'last_review']
    numeric_columns = [col for col in df.columns if col not in exclude_columns and df[col].dtype in ['int64', 'float64']]
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    df.to_csv(output_scaled, index=False, date_format='%Y-%m-%d')
    print(f"Scaled dataset saved to {os.path.abspath(output_scaled)}")
    print(f"Scaled columns: {numeric_columns}")

    # Step 4: Visualize Data Distribution
    print("\nStep 4: Visualizing data distribution...")
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_columns, 1):
        plt.subplot((len(numeric_columns) // 3) + 1, 3, i)
        plt.hist(df[col], bins=30, edgecolor='black')
        plt.title(col)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_distribution_plot)
    print(f"Data distribution histograms saved to {os.path.abspath(output_distribution_plot)}")
    plt.close()

    # Step 5: Remove Outliers
    print("\nStep 5: Removing outliers...")
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    original_rows = len(df)
    for col in numeric_columns:
        df = remove_outliers(df, col)

    df.to_csv(output_outliers_removed, index=False, date_format='%Y-%m-%d')
    print(f"Dataset with outliers removed saved to {os.path.abspath(output_outliers_removed)}")
    print(f"Original rows: {original_rows}, Rows after outlier removal: {len(df)}")

    # Step 6: Correlation Matrix and Analysis
    print("\nStep 6: Analyzing correlations...")
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = df[numeric_columns].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Matrix')
    plt.savefig(output_correlation_plot)
    print(f"Correlation matrix saved to {os.path.abspath(output_correlation_plot)}")
    plt.close()

    threshold = 0.7
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

    if high_corr_pairs:
        print("\nHighly correlated pairs (|correlation| > 0.7):")
        for pair in high_corr_pairs:
            print(f"{pair[0]} and {pair[1]}: {pair[2]:.2f}")
    else:
        print("\nNo pairs with |correlation| > 0.7 found.")

    # Regression Modeling
    print("\nStarting regression modeling...")
    # Validate data for modeling
    if df.empty:
        raise ValueError("Dataset is empty after preprocessing. Please check preprocessing steps.")
    
    target = 'price'
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset. Available columns: {df.columns.tolist()}")
    
    exclude_columns = ['price', 'id', 'host_id', 'last_review']
    features = [col for col in df.columns if col not in exclude_columns and df[col].dtype in ['int64', 'float64']]
    
    if not features:
        raise ValueError("No numeric features found for modeling. Please check the dataset preprocessing.")

    print(f"Features used for modeling: {features}")
    print(f"Target: {target}")
    print(f"Dataset size for modeling: {len(df)} rows")

    X = df[features]
    y = df[target]

    if X.isna().any().any() or y.isna().any():
        raise ValueError("Dataset contains NaN values after preprocessing. Please ensure data is clean.")

    # Split data
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # Initialize results dictionary
    results = {
        'Model': [],
        'MAE': [],
        'MSE': []
    }

    # Mapping of model names to output_plots keys
    model_key_mapping = {
        'Linear Regression': 'linear',
        'Polynomial Regression': 'poly',
        'Decision Tree': 'dt',
        'Random Forest': 'rf',
        'RNN (LSTM)': 'rnn'
    }

    # Function to plot predictions vs actual
    def plot_pred_actual(y_true, y_pred, model_name, output_path):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(f'{model_name}: Predicted vs Actual Price')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    # Function to plot residuals
    def plot_residuals(y_true, y_pred, model_name, output_path):
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, bins=30, kde=True)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title(f'{model_name}: Residuals Distribution')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    # Function to plot learning curves
    def plot_learning_curve(estimator, X, y, model_name, output_path):
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error'
        )
        train_scores_mean = -train_scores.mean(axis=1)
        test_scores_mean = -test_scores.mean(axis=1)
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, label='Training MSE')
        plt.plot(train_sizes, test_scores_mean, label='Validation MSE')
        plt.xlabel('Training Examples')
        plt.ylabel('Mean Squared Error')
        plt.title(f'{model_name}: Learning Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    # Function to evaluate and store results
    def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            results['Model'].append(model_name)
            results['MAE'].append(mae)
            results['MSE'].append(mse)
            plot_key = model_key_mapping.get(model_name)
            if plot_key not in output_plots:
                raise KeyError(f"Plot key '{plot_key}' not found in output_plots dictionary.")
            plot_pred_actual(y_test, y_pred, model_name, output_plots[plot_key]['pred_actual'])
            plot_residuals(y_test, y_pred, model_name, output_plots[plot_key]['residuals'])
            plot_learning_curve(model, X_train, y_train, model_name, output_plots[plot_key]['learning_curve'])
            return model
        except Exception as e:
            raise Exception(f"Error in {model_name}: {str(e)}")

    # 1. Linear Regression
    print("\nTraining Linear Regression...")
    lr = LinearRegression()
    lr = evaluate_model(lr, 'Linear Regression', X_train, X_test, y_train, y_test)

    # 2. Polynomial Regression
    print("\nTraining Polynomial Regression...")
    degrees = [2, 3]
    best_degree = 2
    best_mse = float('inf')
    best_polyreg = None
    for degree in degrees:
        polyreg = make_pipeline(PolynomialFeatures(degree), StandardScaler(), LinearRegression())
        polyreg.fit(X_train, y_train)
        y_pred = polyreg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        if mse < best_mse:
            best_mse = mse
            best_degree = degree
            best_polyreg = polyreg
    print(f"Best polynomial degree: {best_degree}")
    polyreg = evaluate_model(best_polyreg, 'Polynomial Regression', X_train, X_test, y_train, y_test)

    # 3. Decision Tree
    print("\nTraining Decision Tree...")
    dt_param_grid = {
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
    dt = DecisionTreeRegressor(random_state=42)
    dt_grid = GridSearchCV(dt, dt_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    dt_grid = evaluate_model(dt_grid, 'Decision Tree', X_train, X_test, y_train, y_test)
    print(f"Best Decision Tree params: {dt_grid.best_params_}")

    # Feature importance for Decision Tree
    plt.figure(figsize=(10, 6))
    feature_importance = dt_grid.best_estimator_.feature_importances_
    sns.barplot(x=feature_importance, y=features)
    plt.title('Decision Tree: Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(output_plots['dt']['feature_importance'])
    plt.close()

    # 4. Random Forest
    print("\nTraining Random Forest...")
    rf_param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20]
    }
    rf = RandomForestRegressor(random_state=42)
    rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    rf_grid = evaluate_model(rf_grid, 'Random Forest', X_train, X_test, y_train, y_test)
    print(f"Best Random Forest params: {rf_grid.best_params_}")

    # Feature importance for Random Forest
    plt.figure(figsize=(10, 6))
    feature_importance = rf_grid.best_estimator_.feature_importances_
    sns.barplot(x=feature_importance, y=features)
    plt.title('Random Forest: Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(output_plots['rf']['feature_importance'])
    plt.close()

    # 5. Recurrent Neural Network (LSTM) - Only if TensorFlow is available
    if tensorflow_available:
        print("\nTraining RNN (LSTM)...")
        X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

        def build_lstm(units, input_shape):
            model = Sequential()
            model.add(LSTM(units, input_shape=input_shape))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            return model

        best_units = 50
        best_mse = float('inf')
        best_lstm = None
        for units in [50, 100]:
            lstm = build_lstm(units, (1, X_train.shape[1]))
            lstm.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=0)
            y_pred = lstm.predict(X_test_lstm, verbose=0).flatten()
            mse = mean_squared_error(y_test, y_pred)
            if mse < best_mse:
                best_mse = mse
                best_units = units
                best_lstm = lstm
        print(f"Best LSTM units: {best_units}")

        y_pred = best_lstm.predict(X_test_lstm, verbose=0).flatten()
        results['Model'].append('RNN (LSTM)')
        results['MAE'].append(mean_absolute_error(y_test, y_pred))
        results['MSE'].append(mean_squared_error(y_test, y_pred))
        plot_pred_actual(y_test, y_pred, 'RNN (LSTM)', output_plots['rnn']['pred_actual'])
        plot_residuals(y_test, y_pred, 'RNN (LSTM)', output_plots['rnn']['residuals'])

        train_sizes = np.linspace(0.1, 1.0, 10)
        train_scores = []
        test_scores = []
        for frac in train_sizes:
            n_samples = int(len(X_train) * frac)
            X_subset = X_train_lstm[:n_samples]
            y_subset = y_train[:n_samples]
            lstm = build_lstm(best_units, (1, X_train.shape[1]))
            lstm.fit(X_subset, y_subset, epochs=20, batch_size=32, verbose=0)
            train_pred = lstm.predict(X_subset, verbose=0).flatten()
            test_pred = lstm.predict(X_test_lstm, verbose=0).flatten()
            train_scores.append(mean_squared_error(y_subset, train_pred))
            test_scores.append(mean_squared_error(y_test, test_pred))
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes * len(X_train), train_scores, label='Training MSE')
        plt.plot(train_sizes * len(X_train), test_scores, label='Validation MSE')
        plt.xlabel('Training Examples')
        plt.ylabel('Mean Squared Error')
        plt.title('RNN (LSTM): Learning Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_plots['rnn']['learning_curve'])
        plt.close()

    # Save comparison table
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_comparison_table, index=False)
    print("\nModel Performance Comparison:")
    print(results_df)

except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {str(e)}")