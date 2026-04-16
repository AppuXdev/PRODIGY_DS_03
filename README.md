# Task 3: Decision Tree Classifier - Bank Marketing Dataset

## Objective
Build a decision tree classifier to predict whether a customer will subscribe to a term deposit based on demographic and behavioral data from bank marketing campaigns.

## Dataset
- **Source**: UCI Machine Learning Repository - Bank Marketing Dataset
- **File**: bank.csv
- **Records**: 4521 clients, 16 input features + 1 target variable
- **Target**: `y` - has the client subscribed to a term deposit? (yes/no)
- **Features**: age, job, marital, education, balance, housing, loan, contact, duration, campaign, etc.

## Files
- `bank.ipynb` - Jupyter notebook with complete analysis and visualizations
- `bank.py` - Python script version of the model
- `bank.csv` - Dataset used for training and testing

## Methodology
1. **Data Preprocessing**
   - Target variable `y` mapped to binary 0/1
   - Binary features encoded: default, housing, loan
   - Categorical features label encoded: job, marital, education, contact, month, poutcome
   
2. **Model Building**
   - **Algorithm**: DecisionTreeClassifier from scikit-learn
   - **Parameters**: max_depth=5, min_samples_split=20, random_state=42
   - **Train-Test Split**: 80-20 with stratification to handle class imbalance

3. **Evaluation**
   - Accuracy score, confusion matrix, classification report
   - Feature importance analysis
   - Decision tree visualization

## Results
- **Accuracy**: ~89-90% on test set
- **Class Distribution**: 88.5% 'no', 11.5% 'yes' - imbalanced dataset

### Top 5 Most Important Features
1. **duration**: 0.556 - Call duration is the strongest predictor
2. **pdays**: 0.149 - Days since client was last contacted  
3. **poutcome**: 0.119 - Outcome of previous marketing campaign
4. **age**: 0.078 - Client age
5. **month**: 0.042 - Last contact month

## Key Insights
1. **Behavioral > Demographic**: Call duration and contact history predict purchases better than age, job, or education
2. **Engagement Signal**: Longer call duration strongly correlates with subscription - engaged customers buy
3. **Warm Leads Convert**: Previous campaign success and recent contact improve conversion rates
4. **Model Limitation**: Due to 88/12 class imbalance, model predicts non-subscribers better than subscribers

## Tech Stack
`Python` `Pandas` `NumPy` `Scikit-learn` `Matplotlib` `Seaborn` `Jupyter Notebook`

## How to Run
1. Install dependencies: `pip install pandas scikit-learn matplotlib seaborn jupyter`
2. **Notebook**: Open `gtrain.ipynb` in Jupyter
3. **Script**: Run `python train.py`
