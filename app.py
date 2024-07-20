import gradio as gr
import pickle
import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Load the model and preprocessor
model = pickle.load(open('model.pkl', 'rb'))
preprocessor = pickle.load(open('preprocess_pipeline.pkl', 'rb'))



def call_gradio(Sex, Equipment, Age, BodyweightKg, BestSquatKg, Bestbenchkg):
    # Convert inputs to appropriate data types
    Age = float(Age)
    BodyweightKg = float(BodyweightKg)
    BestSquatKg = float(BestSquatKg)
    Bestbenchkg = float(Bestbenchkg)
    
    # Create a DataFrame with the input data
    df_x = pd.DataFrame({
        'Sex': [Sex],
        'Equipment': [Equipment],
        'Age': [Age],
        'BodyweightKg': [BodyweightKg],
        'BestSquatKg': [BestSquatKg],
        'Bestbench(kg)': [Bestbenchkg],
    })
    
    # Define the categorical and numerical feature lists
    categorical_features = ['Sex', 'Equipment']
    numerical_features = df_x.drop(categorical_features, axis=1).columns.tolist()
    
    # Ensure that the preprocessor is fitted
    if not hasattr(preprocessor, 'transformers_'):
        raise RuntimeError("The preprocessor is not fitted yet. Fit the preprocessor before calling this function.")
    
    
    # Transform the data using the preprocessor
    X_processed = preprocessor.transform(df_x)
    
    # Access the OneHotEncoder directly to get feature names
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = ohe.get_feature_names_out(categorical_features)
    
    # Combine numerical features and one-hot encoded feature names
    all_feature_names = numerical_features + list(cat_feature_names)
    
    # Create a DataFrame with processed features
    X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)
    
    # Predict using the model
    y_pred = model.predict(X_processed_df)
    
    max_kg = int(y_pred[0])
    return max_kg

# Define Gradio inputs and outputs
sex_dropdown = gr.Dropdown(choices=['M', 'F'], label="Sex", info="Select Male or Female")
equipment_dropdown = gr.Dropdown(choices=['Raw', 'Wraps', 'Single-ply', 'Multi-ply'], label="Equipment", info="Select the equipment used for the competition.")
age_textbox = gr.Textbox(lines=1, label="Age", info="Enter your Age")
bodyweight_kg_textbox = gr.Textbox(lines=1, label="BodyweightKg", info="Enter your Bodyweight in Kg")
best_squat_kg_textbox = gr.Textbox(lines=1, label="BestSquatKg", info="Enter your Best Squat in Kg")
best_bench_kg_textbox = gr.Textbox(lines=1, label="BestbenchKg", info="Enter your Best Bench in Kg")

# Custom description with image and footer
description = """
<div style='text-align: center;'>
    <h1 style='font-size: 50px;'>PowerLift Muscle Map</h1>
    <p>Use this model to estimate your best Deadlift (kg) based on your selected features. Input your details and see the predicted weight (kg) you could lift.</p>
    <p><strong>Output:</strong> Estimated Best Deadlift (kg)</p>
    <br>
    <p style='font-size: 10px; color: #555;'>❤️ PDS</p>
    
</div>

"""

# Create and launch Gradio interface
iface = gr.Interface(
    fn=call_gradio,
    inputs=[sex_dropdown, equipment_dropdown, age_textbox, bodyweight_kg_textbox, best_squat_kg_textbox, best_bench_kg_textbox],
    outputs="number",
    description=description,
)

iface.launch()
