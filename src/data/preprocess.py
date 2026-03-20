def prepare_data(df):
    
    X = df[["HorasEstudio"]]
    y = df["Resultado"].map({"No":0, "Si":1}) # convertir a 0/1
    
    return X, y