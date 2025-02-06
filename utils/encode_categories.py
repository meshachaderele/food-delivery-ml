
def categories_encoder(train_data):
    import pandas as pd
    # Encoding categorical variables
    traffic_mapping = {'Very Low': 0, 'Low': 1, 'Moderate': 2, 'High': 3, 'Very High': 4}
    train_data['Traffic_Level'] = train_data['Traffic_Level'].map(traffic_mapping)

    train_data = pd.get_dummies(train_data, columns=['Type_of_order', 'Type_of_vehicle', 'weather_description'], drop_first=True).astype(int)

    return train_data
