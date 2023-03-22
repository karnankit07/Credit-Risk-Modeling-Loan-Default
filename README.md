![What-is-Credit-Risk-Modeling](https://user-images.githubusercontent.com/57935250/226844021-ec7c5b2d-ca13-49e3-8a03-e4449c91cf43.jpg)

# Credit Risk Modeling

Credit risk modeling is the process of using statistical analysis and mathematical models to assess the creditworthiness of borrowers and to estimate the likelihood of default on loans or other credit products. The models are typically based on historical data and use a range of factors, such as income, employment history, credit history, and other financial information, to predict the probability of default.

Credit risk modeling is used by financial institutions, such as banks, to manage their credit risk exposure, set appropriate interest rates, and make informed decisions about lending. It can also be used by investors to assess the creditworthiness of bonds and other debt securities.

There are several types of credit risk models, including credit scoring models, which assign a numerical score to each borrower based on their credit history and other factors; credit migration models, which track changes in credit quality over time; and default probability models, which estimate the probability of default for a given borrower or portfolio.

### Default Probability Models

Default probability models, also known as default prediction models, are a type of credit risk model that estimate the likelihood of a borrower defaulting on a loan or other credit product. These models are widely used by financial institutions to assess credit risk, set appropriate interest rates, and make informed lending decisions.

Default probability models typically use a range of borrower-specific and macroeconomic factors to predict the probability of default. Some common borrower-specific factors that may be included in these models include credit history, income, employment history, debt-to-income ratio, and other financial information. Macroeconomic factors, such as interest rates, inflation, and economic growth, may also be incorporated into the models to account for broader economic trends.

There are several different types of default probability models, including logistic regression models, decision tree models, and neural network models. These models use statistical techniques to analyze historical data and identify patterns and relationships that can be used to predict default risk.

## Code and Resources Used

Python Version: 3.11

Packages: pandas, NumPy, sklearn, scipy, matplotlib, seaborn, pickle, xgboost, imblearn

Algorithms: Linear Regression, Logistic Regression, Decision Tree, Random Forest Classifier

Dataset: Applicant.csv(https://drive.google.com/file/d/1O1SbkngGlJP18R7I7YWN2zQ_mJJ9REHR/view?usp=share_link)

Loan.csv(https://drive.google.com/file/d/1aIs_gBAexWwmELNztOT8DxIooF7YMpaz/view?usp=share_link)
## Dataset Information
The dataset has two files:

1. `applicant.csv`: This file contains personal data about the (primary) applicant
- Unique ID: `applicant_id` (string)
- Other fields:
    - Primary_applicant_age_in_years (numeric)
    - Gender (string)
    - Marital_status (string)
    - Number_of_dependents (numeric)
    - Housing (string)
    - Years_at_current_residence (numeric)
    - Employment_status (string)
    - Has_been_employed_for_at_least (string)
    - Has_been_employed_for_at_most (string)
    - Telephone (string)
    - Foreign_worker (numeric)
    - Savings_account_balance (string)
    - Balance_in_existing_bank_account_(lower_limit_of_bucket) (string)
    - Balance_in_existing_bank_account_(upper_limit_of_bucket) (string)

1. `loan.csv`: This file contains data more specific to the loan application
- Target: `high_risk_application` (numeric)
- Other fields:
    - applicant_id (string)
    - Months_loan_taken_for (numeric)
    - Purpose (string)
    - Principal_loan_amount (numeric)
    - EMI_rate_in_percentage_of_disposable_income (numeric)
    - Property (string)
    - Has_coapplicant (numeric)
    - Has_guarantor (numeric)
    - Other_EMI_plans (string)
    - Number_of_existing_loans_at_this_bank (numeric)
    - Loan_history (string)
## Data Preparation

We have two dataset applicant dataset is related to personal detail of the applicant and second dataset is related to specific loan details of applicants.

In both dataset we have one common column (applicant_id) to connect two dataframe in order to have all the data of the applicant in one dataframe to analyze.

In applicant dataset we have 1000 rows and 15 columns.
In 6 columns we have null values we have to find a way to handle null values.

In loan datset we have 1000 rows and 13 columns.
Here we have null values in 3 columns.

I have merged both the dataframe. We have two unique id columns(applicant_id, loan_application_id) in our dataset which is only for identify each row uniquely and not important for finding any insight, I have dropped them.

There are 9 columns, Where we have null values.
In columns we have null values near 50% or more than 50 %.
This much null values can affect our prediction drastically either we can drop them directly or find a better way of replacing them.
I had handled null values in each column one by one.


## Exploratory Data analysis

Inference after EDA

 Male gender have more data in our datset.

 More single people in our dataset than married/divorces/seperated and have least for divorces seperated.

 People own house has more in our dataset than who live in rented house.

 We have skilled employee/ Officer maximun in our dataset and very low for unemployed/ unskilled & non resident.

 We have maximum number of people who is employed for at least 1 year and maximum number of people who is employed for maximum for 7 years.

 Maximum number of people have saving balance account at low category and very low amount of people have either medium, high or very high saving balance
account.

 Purpose of taking a loan in most of the cases is for elctronic equipment, new vehicle or FF&E (Furnitures fixtures and equipment).

 Maximum number of people who owns some kind of property are for car or other than real estate.

 In more than 50% of the cases are for existing loans paid back duly till now and less than 5% of the cases are for no loan taken/ all loans paid back duly.

 High risk applicant is comparatively higher for female than male.

 Separated/divorced people have comparatively higher risk applicant.

 People who have housing status free or for rent have comparatively higher risk applicant.

 People who are employed for 0 or 1 year are comparatively high risk applicants.

 People who have saving balance account low or medium are comparatively higher risk applicant.

 Purpose of taking a loan for education or vehicle are comparatively higher risk applicant.

 People who own car or other property are at little higher risk.
## Model Building 

I have used Logistic Regression model Logistic regression is a common statistical technique used in credit risk modeling to predict the probability of default. In logistic regression, the goal is to estimate the probability of an event occurring, in this case, default on a loan, based on a set of predictor variables.

The logistic regression model produces a predicted probability of default, which can be used to classify borrowers into different risk categories. For example, a lender may use the predicted probability of default to assign borrowers to different risk tiers and set appropriate interest rates based on the level of risk.


Hyperparameters -: GridSearchCV on Logistic Regression

I used a decision tree because it’s a flowchart-like structure in which each internal node represents a "test" on an attribute (e.g. whether a coin toss comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after computing all attributes). It is useful in cases where data set is very scattered in both X and Y axis in graph and in which multiple
attributes are responsible for prediction.

Hyperparameters -: GridSearchCV on Decision Tree

I have used Random Forest classifier because Random Forest is suitable for situations when we have a large dataset, and interpretability is not a major concern.
Decision trees are much easier to interpret and understand. Since a random forest combines multiple decision trees, it becomes more difficult to interpret. A random forest is a meta estimator that fits a number of decision tree classifiers on various
sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

Hyperparameters -: GridSearchCV on Random Forest Classifier


## 🚀 About Me
I'm a passionate Data Science Enthusiast

- 🌱 I’m currently learning **Data Science, Machine Learning, Statistical Analysis, Predictive Modelling, Deep Learning**

- 👨‍💻 All of my projects are available at https://sites.google.com/view/ankit-kumar-portfolio/home


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://sites.google.com/view/ankit-kumar-portfolio/home)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ankit-kumar-07123/)



## Acknowledgements & Reference

 - [A Machine Learning Approach To Credit Risk Assessment](https://towardsdatascience.com/a-machine-learning-approach-to-credit-risk-assessment-ba8eda1cd11f)
 - [Credit Risk Modeling with Machine Learning](https://towardsdatascience.com/credit-risk-modeling-with-machine-learning-8c8a2657b4c4)
