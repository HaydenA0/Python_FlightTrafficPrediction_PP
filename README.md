


# Flight Traffic Analysis and Prediction Dashboard

### 0. Demonstration

![Main Page](./images/main_interface.png)
![plot](./images/plot_lr_scatter_prepandemic_traffic.png)

### 1. General information

-   **Name of the project**: Flight Traffic Analysis and Prediction Dashboard
-   **Type of the project**: Internship
-   **Main Language(s) of the project**: Python
-   **Goal of this project**: The goal was to analyze a dataset of historical flight operations, train machine learning models to predict future flight traffic, and present these insights and tools in an interactive web dashboard. A key focus was understanding the impact of the COVID-19 pandemic on flight volumes.
-   **Scope of this project**: The project encompasses the entire pipeline from data cleaning and feature engineering to model training and deployment as a local web application. It includes training and evaluating multiple regression models (Ridge and LightGBM), and building a multi-tab, bilingual (EN/FR) dashboard using Plotly Dash for data visualization and live predictions.
-   **Compatibility**: This project is built with Python and its standard data science libraries (Pandas, Scikit-learn, etc.). It can be run on any operating system that supports Python (Windows, macOS, Linux). All required packages are listed in the `requirements.txt` file.

### 2. Project

This project is split into two main components: a model training pipeline and a web dashboard for interaction.

First, I developed a comprehensive training script, `train_model.py`. This script is responsible for loading the raw `flights.csv` data and performing extensive feature engineering. I created features from dates (day of week, day of year), cyclical features using sine/cosine transformations to represent seasonality, and categorical features to capture different periods of the COVID-19 pandemic. It also calculates historical pre-pandemic traffic averages for each airport to serve as a baseline. The script then trains two different models: a regularized Ridge linear regression and a more complex Tuned LightGBM gradient boosting model. The entire process—including the fitted data preprocessor, the trained models, and necessary metadata like feature names and category labels—is saved into a single `flight_prediction_models.joblib` file.

The second component is the interactive dashboard, `app.py`. I built this using the Dash framework. A key design choice was to separate the application logic from the model training. The Dash app simply loads the `flight_prediction_models.joblib` file at startup, making it fast and lightweight. The dashboard is organized into several tabs: an operational summary for exploring data within a specific date range, a model performance tab to compare the metrics of the trained models, a prediction tab to get a flight traffic forecast for a specific airport and future date, and a visualizations tab for broader exploratory analysis of the training data. For the prediction feature, I wrote a helper function within the app that carefully reconstructs the feature vector for a new input, ensuring it matches the format the model was trained on. All visualizations are generated using Plotly for interactivity.

### 3. How to run the project :

To run this project, you will need Python 3 installed. It is highly recommended to use a virtual environment.

1.  **Set up the environment**
    Clone the repository and navigate into the project directory. Create and activate a Python virtual environment:
    ```sh
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

2.  **Install dependencies**
    Install all the required Python packages using the `requirements.txt` file:
    ```sh
    pip install -r requirements.txt
    ```

3.  **Train the models**
    Before you can run the dashboard, you must first run the training script. This will process the data and create the `flight_prediction_models.joblib` file that the app depends on.
    ```sh
    python train_model.py
    ```
    This script may take a few moments to run. Upon completion, you will see a `models` directory containing the `.joblib` file.

4.  **Run the dashboard application**
    Once the models are trained and saved, you can launch the Dash web application:
    ```sh
    python app.py
    ```

5.  **View the dashboard**
    Open your web browser and navigate to the address shown in the terminal, which is typically: `http://127.0.0.1:8050/`


