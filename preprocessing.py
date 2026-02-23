import pandas as pd

def preprocess_data():
    # loading the data
    print("Loading datasets...")
    try:
        #app_train = pd.read_csv('application_train.csv', nrows=10000)
        #bureau = pd.read_csv('bureau.csv', nrows=30000)

        app_train = pd.read_csv('application_train.csv')
        bureau = pd.read_csv('bureau.csv')

    except FileNotFoundError:
        print("Error:'application_train.csv' and 'bureau.csv' not found")
        return

    # bureau.csv has multiple rows per client We need 1 row per client.
    print("Aggregating bureau data...")

    # group by client id (SK_ID_CURR) and calculating statistics
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({
        'DAYS_CREDIT': ['mean', 'min', 'max'],  # how long ago was their credit
        'AMT_CREDIT_SUM': ['mean', 'sum'],  # debt amount
        'CREDIT_ACTIVE': ['count']  # previous loans
    })

    # flatten the multi level column names
    bureau_agg.columns = ['_'.join(col).strip() for col in bureau_agg.columns.values]
    bureau_agg.reset_index(inplace=True)

    # merge new features into main table
    print("Merging datasets...")
    df = app_train.merge(bureau_agg, on='SK_ID_CURR', how='left')

    # droping the id column
    df = df.drop(columns=['SK_ID_CURR'])

    print("Cleaning missing values and encoding...")

    # seperating target y and features X
    # label encoding for handling categorical data
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        # fill missing strings
        df[col] = df[col].fillna("Unknown")
        # converting to integers
        df[col] = pd.factorize(df[col])[0]

    # handling numerical data
    # filling missing numbers with median
    numerical_cols = df.select_dtypes(exclude=['object']).columns

    for col in numerical_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # making sure all data is numeric
    print(f"Final Data Shape: {df.shape}")
    print(f"Missing values remaining: {df.isnull().sum().sum()}")

    # saving csv
    output_filename = 'clean_data.csv'
    df.to_csv(output_filename, index=False)
    print(f"Success! Clean data saved to '{output_filename}'")


if __name__ == "__main__":
    preprocess_data()