# Powerlift_Muscle_Map

**Author:** Prince Deepak Siddharth

## Project Overview

PowerLift Muscle Map is a machine learning project designed to predict an individual's best deadlift weight based on various features. This project utilizes the Gradient Boosting Regressor algorithm and is built with scikit-learn, EDA, pandas, numpy, matplotlib, seaborn, and the pickle library. The aim is to provide insights and motivation for individuals participating in strength training, particularly powerlifting.

## Dataset Description

The dataset used for this project is sourced from Kaggle: [Powerlifting Benchpress Weight Predict Dataset](https://www.kaggle.com/datasets/kukuroo3/powerlifting-benchpress-weight-predict/data).

### Features:

1. **playerId**: A unique identifier for each player.
2. **Name**: The name of the player.
3. **Sex**: The gender of the player ('Male' or 'Female').
4. **Equipment**: The type of lifting equipment used, which can be one of the following:
    - **Raw**: Lifting without any extra equipment.
    - **Wraps**: Using supportive wraps for knees or wrists.
    - **Single-ply**: Wearing special lifting gear made from one layer of fabric.
    - **Multi-ply**: Using lifting gear made from several layers of fabric.
5. **Age**: The age of the player.
6. **BodyweightKg**: The body weight of the player in kilograms.
7. **BestSquatKg**: The maximum weight the player can squat, measured in kilograms.
8. **BestDeadliftKg**: The maximum weight the player can deadlift, measured in kilograms. This is the target variable for prediction.
9. **Bestbench(kg)**: The maximum weight the player can bench press, measured in kilograms.

### Aim:

The goal of the PowerLift Muscle Map project is to build and evaluate predictive models to estimate the best deadlift weight based on the available features. The project explores different models, analyzes feature importance, and visualizes the impact of various factors on lifting performance. It aims to help individuals in the gym predict their maximum deadlift weight based on some information, motivating them to achieve their predicted goals.

**Caution for Beginners**:
- When starting with deadlift training, it's important to begin with half of your predicted maximum weight. Gradually increase the weight in small steps, focusing on improving your form and boosting your confidence with each lift. This approach helps you build strength safely and effectively.

*Precaution is better than injury.*  
**Rule for Gym || Rule for Life**  
...Time to Level up guys...  
(2 AM thoughts - PDS)

## Project Deployment

The project has been deployed using Gradio on Hugging Face: [PowerLift Muscle Map](https://huggingface.co/spaces/PrinceDeepakSiddharth12/PowerLift_Muscle_Map).

### Files Included:

- **Jupyter Notebook**: `powerlifting_muscle_map.ipynb`
- **Python Script**: `app.py`
- **Dataset CSV Files**:
  - `gym_powerlifting_correct.csv`
  - `gym_powerlifting_cleaned_data.csv`
- **Pickle Files**:
  - `preprocess_pipeline.pkl`
  - `model.pkl`
- **Requirements File**: `requirements.txt`

## Usage

To run the project locally:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/PowerLift_Muscle_Map.git
    ```
2. Navigate to the project directory:
    ```sh
    cd PowerLift_Muscle_Map
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
4. Run the application:
    ```sh
    python app.py
    ```

## Conclusion

The PowerLift Muscle Map project provides a useful tool for individuals to predict their best deadlift weight based on various features. It serves as a motivational aid and offers insights into the impact of different factors on lifting performance.

**Let's get stronger together!**

---

❤️ PDS
