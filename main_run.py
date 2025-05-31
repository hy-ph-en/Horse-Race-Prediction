import pandas as pd
from Data_Processing.preprocessing import Preprocessing
from Model.stack import model_run
from Evaluation.evaluation import evaluate_model


def run():

    #Preprocessing Data
    data = Preprocessing().check_and_preprocess()

    #Training Model
    model_components, train_data = model_run(data)

    #Evaluating Model
    metrics = evaluate_model(model_components, train_data)
    
    return model_components, metrics


if __name__ == "__main__":
    model_components, metrics = run()
