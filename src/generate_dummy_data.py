import pandas as pd
import numpy as np
import os

def generate_dummy_data(output_path='data/dummy_dataset.csv', n_samples=560, seed=42):
    """
    Generates a synthetic dataset mimicking the T2DM cohort from the paper.
    
    Paper Stats:
    - N=560
    - CVD Prevalence: ~7.3% (41 cases)
    - Features: Age, Sex, BMI, HbA1c, Cholesterol, etc.
    """
    np.random.seed(seed)
    
    print(f"Generating synthetic dataset with {n_samples} patients...")
    
    # --- 1. Generate Continuous Variables (Normal Distribution) ---
    # format: (mean, std)
    cont_specs = {
        'Age': (58.56, 10.70),
        'Diabetes_Duration': (7.67, 7.37),
        'BMI': (29.49, 5.54),
        'Pulse_Pressure': (56.75, 15.80),
        'HbA1c': (7.43, 1.81),
        'Fasting_Glucose': (165.15, 56.15),
        'Total_Cholesterol': (226.64, 50.04),
        'Triglycerides': (167.39, 110.81),
        'HDL_Cholesterol': (48.35, 16.46)
    }
    
    data = {}
    for col, (mu, sigma) in cont_specs.items():
        # Generate and clip to realistic ranges (no negative values)
        values = np.random.normal(mu, sigma, n_samples)
        values = np.maximum(values, 0) 
        data[col] = values

    # --- 2. Generate Categorical Variables (Probabilities) ---
    # Sex: Male ~47%, Female ~53%
    data['Sex'] = np.random.choice(['Male', 'Female'], n_samples, p=[0.47, 0.53])
    
    # Smoking: Non 51.6%, Current 26%, Ex 22.4%
    data['Smoking_Habit'] = np.random.choice(
        ['Non Smoker', 'Current Smoker', 'Ex-Smoker'], 
        n_samples, 
        p=[0.516, 0.260, 0.224]
    )
    
    # Parental History of Diabetes: No 54%, Yes 46%
    data['Parental_History_Diabetes'] = np.random.choice(
        ['No', 'Yes'], n_samples, p=[0.54, 0.46]
    )
    
    # Lipid Lowering Therapy: No 83.7%, Statins 13.2%, Fibrates 3%
    data['Lipid_Lowering_Therapy'] = np.random.choice(
        ['No', 'Statins', 'Fibrates'], 
        n_samples, 
        p=[0.8375, 0.1321, 0.0304]
    )
    
    # Aspirin: No 90.9%, 100mg 7.8%, 325mg 3%
    # Adjust sums to exactly 1.0 due to rounding
    probs_asp = [0.9089, 0.0785, 0.0126] # 325mg reduced slightly to sum to 1
    data['Aspirin'] = np.random.choice(
        ['No', '100 mg', '325 mg'], 
        n_samples, 
        p=probs_asp
    )

    df = pd.DataFrame(data)

    # --- 3. Synthesize Target Variable (CVD_Outcome) ---
    # To ensure the model can actually "learn" something, we shouldn't assign CVD randomly.
    # We create a "Risk Score" based on established medical knowledge (and paper findings).
    # Higher Risk = Higher Chance of CVD.
    
    # Normalize for score calculation
    norm = lambda x: (x - x.mean()) / x.std()
    
    risk_score = (
        0.4 * norm(df['Age']) + 
        0.3 * norm(df['Total_Cholesterol']) + 
        0.3 * norm(df['HbA1c']) + 
        0.2 * norm(df['Diabetes_Duration']) - 
        0.3 * norm(df['HDL_Cholesterol']) +  # Low HDL is bad
        0.2 * (df['Smoking_Habit'] == 'Current Smoker').astype(int) +
        0.2 * (df['Lipid_Lowering_Therapy'] != 'No').astype(int) # Proxy for bad lipids
    )
    
    # Add some noise
    risk_score += np.random.normal(0, 0.5, n_samples)
    
    # Select top ~7.3% as CVD cases (approx 41 patients)
    target_count = int(0.0732 * n_samples)
    threshold = risk_score.nlargest(target_count).min()
    
    df['CVD_Outcome'] = (risk_score >= threshold).astype(int)
    
    # --- 4. Save ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Dataset saved to {output_path}")
    print(f"CVD Prevalence: {df['CVD_Outcome'].mean()*100:.2f}% ({df['CVD_Outcome'].sum()} cases)")
    print(df.head())

if __name__ == "__main__":
    generate_dummy_data()
