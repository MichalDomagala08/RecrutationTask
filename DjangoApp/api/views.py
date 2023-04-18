from django.shortcuts import render

from django.shortcuts import render

#Import necessary libraries
import pickle
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import render
import numpy as np
import sys

# Create your views here.
@api_view(['GET'])
def index_page(request):
    return_data = {
        "error" : "0",
        "message" : "Successful",
    }
    return Response(return_data)

@api_view(["POST"])
def predict_cover(request):
    try:
        Elevation = request.data.get('Elevation',None)
        Aspect = request.data.get('Aspect',None)
        Slope = request.data.get('Slope',None)
        Horizontal_Distance_To_Hydrology = request.data.get('Horizontal_Distance_To_Hydrology',None)
        Vertical_Distance_To_Hydrology = request.data.get('Vertical_Distance_To_Hydrology',None)
        Horizontal_Distance_To_Roadways = request.data.get('Horizontal_Distance_To_Roadways',None)
        Hillshade_9am = request.data.get('Hillshade_9am',None)
        Hillshade_Noon = request.data.get('Hillshade_Noon',None)
        Hillshade_3pm = request.data.get('Hillshade_3pm',None)
        Horizontal_Distance_To_Fire_Points = request.data.get('Horizontal_Distance_To_Fire_Points',None)
        Wilderness_Area1 = request.data.get('Wilderness_Area1',None)
        Wilderness_Area2 = request.data.get('Wilderness_Area2',None)
        Wilderness_Area3 = request.data.get('Wilderness_Area3',None)
        Wilderness_Area4 = request.data.get('Wilderness_Area4',None)
        Soil_Type1 = request.data.get('Soil_Type1',None)
        Soil_Type2 = request.data.get('Soil_Type2',None)
        Soil_Type3 = request.data.get('Soil_Type3',None)
        Soil_Type4 = request.data.get('Soil_Type4',None)
        Soil_Type5 = request.data.get('Soil_Type5',None)
        Soil_Type6 = request.data.get('Soil_Type6',None)
        Soil_Type7 = request.data.get('Soil_Type7',None)
        Soil_Type8 = request.data.get('Soil_Type8',None)
        Soil_Type9 = request.data.get('Soil_Type9',None)
        Soil_Type10 = request.data.get('Soil_Type10',None)
        Soil_Type11 = request.data.get('Soil_Type11',None)
        Soil_Type12 = request.data.get('Soil_Type12',None)
        Soil_Type13 = request.data.get('Soil_Type13',None)
        Soil_Type14 = request.data.get('Soil_Type14',None)
        Soil_Type15 = request.data.get('Soil_Type15',None)
        Soil_Type16 = request.data.get('Soil_Type16',None)
        Soil_Type17 = request.data.get('Soil_Type17',None)
        Soil_Type18 = request.data.get('Soil_Type18',None)
        Soil_Type19 = request.data.get('Soil_Type19',None)
        Soil_Type20 = request.data.get('Soil_Type20',None)
        Soil_Type21 = request.data.get('Soil_Type21',None)
        Soil_Type22 = request.data.get('Soil_Type22',None)
        Soil_Type23 = request.data.get('Soil_Type23',None)
        Soil_Type24 = request.data.get('Soil_Type24',None)
        Soil_Type25 = request.data.get('Soil_Type25',None)
        Soil_Type26 = request.data.get('Soil_Type26',None)
        Soil_Type27 = request.data.get('Soil_Type27',None)
        Soil_Type28 = request.data.get('Soil_Type28',None)
        Soil_Type29 = request.data.get('Soil_Type29',None)
        Soil_Type30 = request.data.get('Soil_Type30',None)
        Soil_Type31 = request.data.get('Soil_Type31',None)
        Soil_Type32 = request.data.get('Soil_Type32',None)
        Soil_Type33 = request.data.get('Soil_Type33',None)
        Soil_Type34 = request.data.get('Soil_Type34',None)
        Soil_Type35 = request.data.get('Soil_Type35',None)
        Soil_Type36 = request.data.get('Soil_Type36',None)
        Soil_Type37 = request.data.get('Soil_Type37',None)
        Soil_Type38 = request.data.get('Soil_Type38',None)
        Soil_Type39 = request.data.get('Soil_Type39',None)
        Soil_Type40 = request.data.get('Soil_Type40',None)
        modelType = request.data.get('modelType',None)

        fields = [Elevation,Aspect,Slope,Horizontal_Distance_To_Hydrology,
                  Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,
                  Hillshade_9am,Hillshade_Noon,Hillshade_3pm,Horizontal_Distance_To_Fire_Points,
                  Wilderness_Area1,Wilderness_Area2,Wilderness_Area3,Wilderness_Area4,
                  Soil_Type1,Soil_Type2,Soil_Type3,Soil_Type4,Soil_Type5,Soil_Type6,
                  Soil_Type7,Soil_Type8,Soil_Type9,Soil_Type10,Soil_Type11,Soil_Type12,
                  Soil_Type13,Soil_Type14,Soil_Type15,Soil_Type16,Soil_Type17,Soil_Type18,
                  Soil_Type19,Soil_Type20,Soil_Type21,Soil_Type22,Soil_Type23,Soil_Type24,
                  Soil_Type25,Soil_Type26,Soil_Type27,Soil_Type28,Soil_Type29,Soil_Type30,
                  Soil_Type31,Soil_Type32,Soil_Type33,Soil_Type34,Soil_Type35,Soil_Type36,
                  Soil_Type37,Soil_Type38,Soil_Type39,Soil_Type40]
        fieldsExp =  [Elevation,Aspect,Slope,Horizontal_Distance_To_Hydrology,
                  Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,
                  Hillshade_9am,Hillshade_Noon,Hillshade_3pm,Horizontal_Distance_To_Fire_Points,
                  Wilderness_Area1,Wilderness_Area2,Wilderness_Area3,Wilderness_Area4,
                  Soil_Type1,Soil_Type2,Soil_Type3,Soil_Type4,Soil_Type5,Soil_Type6,
                  Soil_Type7,Soil_Type8,Soil_Type9,Soil_Type10,Soil_Type11,Soil_Type12,
                  Soil_Type13,Soil_Type14,Soil_Type15,Soil_Type16,Soil_Type17,Soil_Type18,
                  Soil_Type19,Soil_Type20,Soil_Type21,Soil_Type22,Soil_Type23,Soil_Type24,
                  Soil_Type25,Soil_Type26,Soil_Type27,Soil_Type28,Soil_Type29,Soil_Type30,
                  Soil_Type31,Soil_Type32,Soil_Type33,Soil_Type34,Soil_Type35,Soil_Type36,
                  Soil_Type37,Soil_Type38,Soil_Type39,Soil_Type40,modelType]
        if not None in fieldsExp:
            #Datapreprocessing Convert the values to float
            for i in range(len(fields)):
                fields[i] = float(fields[i])

            #Passing data to model & loading the model from Models directory
            if  (type(modelType) ==str) and (modelType in ['LogReg','SVC','DNN','Naive']):
                if modelType == 'LogReg': #Logistic Regression
                    model_path = '.\\Models\\LinearModel.sav'
                    classifier = pickle.load(open(model_path, 'rb'))

                elif  modelType == 'SVC': #Support Vector Classifier
                    model_path = '.\\Models\\SVC.sav'
                    classifier = pickle.load(open(model_path, 'rb'))

                elif  modelType == 'SVC':
                    model_path = '.\\Models\\DNN.sav'
                    classifier = pickle.load(open(model_path, 'rb'))

                elif modelType == 'Naive':

                    # path to NaiveClassifier 
                    sys.path.insert(1,'.\\Models')
                    from  NaiveClassificator import  NaiveCoverClassifer
                    classifier = NaiveCoverClassifer()

                    
                prediction = classifier.predict([fields])[0]
                predictions = {
                    'error' : '0',
                    'message' : 'Successfull',
                    'prediction' : prediction,
                }
            else:
                predictions = {
                    'error' : '1',
                    'message': 'model not chosedn correctly!'                
                }
        else:
            predictions = {
                'error' : '2',
                'message': 'Invalid Parameters'                
            }
    except Exception as e:
        predictions = {
            'error' : '3',
            "message": str(e)
        }
    
    return Response(predictions)
