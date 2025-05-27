import pandas as pd

def run():

    #Loading Data
    data = pd.read_csv('testData.csv')

    #Preprocessing Data
    data = preprocess_data(data)   

    #Training Model
    model = train_model(data)

    #Evaluating Model
    evaluate_model(model, data)


if __name__ == "__main__":
    run()
