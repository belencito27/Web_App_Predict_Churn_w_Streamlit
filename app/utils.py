import numpy as np
import pandas as pd

def transform_data(test_df, col_order, mean_eve_mins, onehot):
    
    # Copy dataframe
    X = test_df.copy()
    
    # Data preprocessing - handle missing values
    X['area_code'] = X['area_code'].fillna('missing')
    X['voice_mail_plan'] = X['voice_mail_plan'].fillna('missing')
    X['total_eve_minutes_missing'] = X['total_eve_minutes'].isnull().astype(int)
    X['total_eve_minutes'] = X['total_eve_minutes'].fillna(mean_eve_mins)
    
    # Feature Engineering
    # FE Rations
    X['day_ratio'] = X['total_day_charge'] / X['total_day_minutes']
    X['eve_ratio'] = X['total_eve_charge'] / X['total_eve_minutes']
    X['night_ratio'] = X['total_night_charge'] / X['total_night_minutes']
    X['intl_ratio'] = X['total_intl_charge'] / X['total_intl_minutes']
    X = X.drop(['total_day_charge', 'total_day_minutes', 'total_eve_charge', 'total_eve_minutes','total_night_charge', 'total_night_minutes', 'total_intl_charge', 'total_intl_minutes' ], axis=1)
    
    # FE Log Transform
    X['number_customer_service_calls'] = np.log(X['number_customer_service_calls']+1)
    
    # FE Unhappy Customers
    X['promotions_offered'] = X['promotions_offered'].replace(['NO', np.NaN], 'No')
    X['unhappy_customers'] = ((X.remaining_term < 5) & (X.last_nps_rating <= 7) & (X.promotions_offered == 'No')).astype(int)
    
    # One hot encoding
    encoded_columns = onehot.transform(X.select_dtypes(include='object')).toarray()
    X = X.select_dtypes(exclude='object')
    X[onehot.get_feature_names_out()] = encoded_columns
    
    return X[col_order]