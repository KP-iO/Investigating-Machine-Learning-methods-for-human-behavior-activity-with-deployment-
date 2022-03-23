# [1] This was helped to be achieved by using the following  source https://www.geeksforgeeks.org/deploying-ml-models-as-api-using-fastapi/

import warnings

import pandas as pd
# Importing Necessary modules
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')
# Declaring our FastAPI instance
app = FastAPI()
#  Load Data
filename = 'TrainingCleaned.csv'
df = pd.read_csv(filename)


# Get our X and Y
X = df.drop(['subject', 'Activity'], axis=1)
Y = df['Activity']

# Build and train our model
model = LogisticRegression(C=1, solver='sag')
model.fit(X, Y)




# Validatuon with BaseModel
class BehaviourActivity(BaseModel):
    tBodyAccmeanX: float
    tBodyAccmeanY: float
    tBodyAccmeanZ: float
    tBodyAccstdX: float
    tBodyAccstdY: float
    tBodyAccstdZ: float
    tBodyAccmadX: float
    tBodyAccmadY: float
    tBodyAccmadZ: float
    tBodyAccmaxX: float
    tBodyAccmaxY: float
    tBodyAccmaxZ: float
    tBodyAccminX: float
    tBodyAccminY: float
    tBodyAccminZ: float
    tBodyAccsma: float
    tBodyAccenergyX: float
    tBodyAccenergyY: float
    tBodyAccenergyZ: float
    tBodyAcciqrX: float
    tBodyAcciqrY: float
    tBodyAcciqrZ: float
    tBodyAccentropyX: float
    tBodyAccentropyY: float
    tBodyAccentropyZ: float
    tBodyAccarCoeffX1: float
    tBodyAccarCoeffX2: float
    tBodyAccarCoeffX3: float
    tBodyAccarCoeffX4: float
    tBodyAccarCoeffY1: float
    tBodyAccarCoeffY2: float
    tBodyAccarCoeffY3: float
    tBodyAccarCoeffY4: float
    tBodyAccarCoeffZ1: float
    tBodyAccarCoeffZ2: float
    tBodyAccarCoeffZ3: float
    tBodyAccarCoeffZ4: float
    tBodyAcccorrelationXY: float
    tBodyAcccorrelationXZ: float
    tBodyAcccorrelationYZ: float
    tGravityAccmeanX: float
    tGravityAccmeanY: float
    tGravityAccmeanZ: float
    tGravityAccstdX: float
    tGravityAccstdY: float
    tGravityAccstdZ: float
    tGravityAccmadX: float
    tGravityAccmadY: float
    tGravityAccmadZ: float
    tGravityAccmaxX: float
    tGravityAccmaxY: float
    tGravityAccmaxZ: float
    tGravityAccminX: float
    tGravityAccminY: float
    tGravityAccminZ: float
    tGravityAccsma: float
    tGravityAccenergyX: float
    tGravityAccenergyY: float
    tGravityAccenergyZ: float
    tGravityAcciqrX: float
    tGravityAcciqrY: float
    tGravityAcciqrZ: float
    tGravityAccentropyX: float
    tGravityAccentropyY: float
    tGravityAccentropyZ: float
    tGravityAccarCoeffX1: float
    tGravityAccarCoeffX2: float
    tGravityAccarCoeffX3: float
    tGravityAccarCoeffX4: float
    tGravityAccarCoeffY1: float
    tGravityAccarCoeffY2: float
    tGravityAccarCoeffY3: float
    tGravityAccarCoeffY4: float
    tGravityAccarCoeffZ1: float
    tGravityAccarCoeffZ2: float
    tGravityAccarCoeffZ3: float
    tGravityAccarCoeffZ4: float
    tGravityAcccorrelationXY: float
    tGravityAcccorrelationXZ: float
    tGravityAcccorrelationYZ: float
    tBodyAccJerkmeanX: float
    tBodyAccJerkmeanY: float
    tBodyAccJerkmeanZ: float
    tBodyAccJerkstdX: float
    tBodyAccJerkstdY: float
    tBodyAccJerkstdZ: float
    tBodyAccJerkmadX: float
    tBodyAccJerkmadY: float
    tBodyAccJerkmadZ: float
    tBodyAccJerkmaxX: float
    tBodyAccJerkmaxY: float
    tBodyAccJerkmaxZ: float
    tBodyAccJerkminX: float
    tBodyAccJerkminY: float
    tBodyAccJerkminZ: float
    tBodyAccJerksma: float
    tBodyAccJerkenergyX: float
    tBodyAccJerkenergyY: float
    tBodyAccJerkenergyZ: float
    tBodyAccJerkiqrX: float
    tBodyAccJerkiqrY: float
    tBodyAccJerkiqrZ: float
    tBodyAccJerkentropyX: float
    tBodyAccJerkentropyY: float
    tBodyAccJerkentropyZ: float
    tBodyAccJerkarCoeffX1: float
    tBodyAccJerkarCoeffX2: float
    tBodyAccJerkarCoeffX3: float
    tBodyAccJerkarCoeffX4: float
    tBodyAccJerkarCoeffY1: float
    tBodyAccJerkarCoeffY2: float
    tBodyAccJerkarCoeffY3: float
    tBodyAccJerkarCoeffY4: float
    tBodyAccJerkarCoeffZ1: float
    tBodyAccJerkarCoeffZ2: float
    tBodyAccJerkarCoeffZ3: float
    tBodyAccJerkarCoeffZ4: float
    tBodyAccJerkcorrelationXY: float
    tBodyAccJerkcorrelationXZ: float
    tBodyAccJerkcorrelationYZ: float
    tBodyGyromeanX: float
    tBodyGyromeanY: float
    tBodyGyromeanZ: float
    tBodyGyrostdX: float
    tBodyGyrostdY: float
    tBodyGyrostdZ: float
    tBodyGyromadX: float
    tBodyGyromadY: float
    tBodyGyromadZ: float
    tBodyGyromaxX: float
    tBodyGyromaxY: float
    tBodyGyromaxZ: float
    tBodyGyrominX: float
    tBodyGyrominY: float
    tBodyGyrominZ: float
    tBodyGyrosma: float
    tBodyGyroenergyX: float
    tBodyGyroenergyY: float
    tBodyGyroenergyZ: float
    tBodyGyroiqrX: float
    tBodyGyroiqrY: float
    tBodyGyroiqrZ: float
    tBodyGyroentropyX: float
    tBodyGyroentropyY: float
    tBodyGyroentropyZ: float
    tBodyGyroarCoeffX1: float
    tBodyGyroarCoeffX2: float
    tBodyGyroarCoeffX3: float
    tBodyGyroarCoeffX4: float
    tBodyGyroarCoeffY1: float
    tBodyGyroarCoeffY2: float
    tBodyGyroarCoeffY3: float
    tBodyGyroarCoeffY4: float
    tBodyGyroarCoeffZ1: float
    tBodyGyroarCoeffZ2: float
    tBodyGyroarCoeffZ3: float
    tBodyGyroarCoeffZ4: float
    tBodyGyrocorrelationXY: float
    tBodyGyrocorrelationXZ: float
    tBodyGyrocorrelationYZ: float
    tBodyGyroJerkmeanX: float
    tBodyGyroJerkmeanY: float
    tBodyGyroJerkmeanZ: float
    tBodyGyroJerkstdX: float
    tBodyGyroJerkstdY: float
    tBodyGyroJerkstdZ: float
    tBodyGyroJerkmadX: float
    tBodyGyroJerkmadY: float
    tBodyGyroJerkmadZ: float
    tBodyGyroJerkmaxX: float
    tBodyGyroJerkmaxY: float
    tBodyGyroJerkmaxZ: float
    tBodyGyroJerkminX: float
    tBodyGyroJerkminY: float
    tBodyGyroJerkminZ: float
    tBodyGyroJerksma: float
    tBodyGyroJerkenergyX: float
    tBodyGyroJerkenergyY: float
    tBodyGyroJerkenergyZ: float
    tBodyGyroJerkiqrX: float
    tBodyGyroJerkiqrY: float
    tBodyGyroJerkiqrZ: float
    tBodyGyroJerkentropyX: float
    tBodyGyroJerkentropyY: float
    tBodyGyroJerkentropyZ: float
    tBodyGyroJerkarCoeffX1: float
    tBodyGyroJerkarCoeffX2: float
    tBodyGyroJerkarCoeffX3: float
    tBodyGyroJerkarCoeffX4: float
    tBodyGyroJerkarCoeffY1: float
    tBodyGyroJerkarCoeffY2: float
    tBodyGyroJerkarCoeffY3: float
    tBodyGyroJerkarCoeffY4: float
    tBodyGyroJerkarCoeffZ1: float
    tBodyGyroJerkarCoeffZ2: float
    tBodyGyroJerkarCoeffZ3: float
    tBodyGyroJerkarCoeffZ4: float
    tBodyGyroJerkcorrelationXY: float
    tBodyGyroJerkcorrelationXZ: float
    tBodyGyroJerkcorrelationYZ: float
    tBodyAccMagmean: float
    tBodyAccMagstd: float
    tBodyAccMagmad: float
    tBodyAccMagmax: float
    tBodyAccMagmin: float
    tBodyAccMagsma: float
    tBodyAccMagenergy: float
    tBodyAccMagiqr: float
    tBodyAccMagentropy: float
    tBodyAccMagarCoeff1: float
    tBodyAccMagarCoeff2: float
    tBodyAccMagarCoeff3: float
    tBodyAccMagarCoeff4: float
    tGravityAccMagmean: float
    tGravityAccMagstd: float
    tGravityAccMagmad: float
    tGravityAccMagmax: float
    tGravityAccMagmin: float
    tGravityAccMagsma: float
    tGravityAccMagenergy: float
    tGravityAccMagiqr: float
    tGravityAccMagentropy: float
    tGravityAccMagarCoeff1: float
    tGravityAccMagarCoeff2: float
    tGravityAccMagarCoeff3: float
    tGravityAccMagarCoeff4: float
    tBodyAccJerkMagmean: float
    tBodyAccJerkMagstd: float
    tBodyAccJerkMagmad: float
    tBodyAccJerkMagmax: float
    tBodyAccJerkMagmin: float
    tBodyAccJerkMagsma: float
    tBodyAccJerkMagenergy: float
    tBodyAccJerkMagiqr: float
    tBodyAccJerkMagentropy: float
    tBodyAccJerkMagarCoeff1: float
    tBodyAccJerkMagarCoeff2: float
    tBodyAccJerkMagarCoeff3: float
    tBodyAccJerkMagarCoeff4: float
    tBodyGyroMagmean: float
    tBodyGyroMagstd: float
    tBodyGyroMagmad: float
    tBodyGyroMagmax: float
    tBodyGyroMagmin: float
    tBodyGyroMagsma: float
    tBodyGyroMagenergy: float
    tBodyGyroMagiqr: float
    tBodyGyroMagentropy: float
    tBodyGyroMagarCoeff1: float
    tBodyGyroMagarCoeff2: float
    tBodyGyroMagarCoeff3: float
    tBodyGyroMagarCoeff4: float
    tBodyGyroJerkMagmean: float
    tBodyGyroJerkMagstd: float
    tBodyGyroJerkMagmad: float
    tBodyGyroJerkMagmax: float
    tBodyGyroJerkMagmin: float
    tBodyGyroJerkMagsma: float
    tBodyGyroJerkMagenergy: float
    tBodyGyroJerkMagiqr: float
    tBodyGyroJerkMagentropy: float
    tBodyGyroJerkMagarCoeff1: float
    tBodyGyroJerkMagarCoeff2: float
    tBodyGyroJerkMagarCoeff3: float
    tBodyGyroJerkMagarCoeff4: float
    fBodyAccmeanX: float
    fBodyAccmeanY: float
    fBodyAccmeanZ: float
    fBodyAccstdX: float
    fBodyAccstdY: float
    fBodyAccstdZ: float
    fBodyAccmadX: float
    fBodyAccmadY: float
    fBodyAccmadZ: float
    fBodyAccmaxX: float
    fBodyAccmaxY: float
    fBodyAccmaxZ: float
    fBodyAccminX: float
    fBodyAccminY: float
    fBodyAccminZ: float
    fBodyAccsma: float
    fBodyAccenergyX: float
    fBodyAccenergyY: float
    fBodyAccenergyZ: float
    fBodyAcciqrX: float
    fBodyAcciqrY: float
    fBodyAcciqrZ: float
    fBodyAccentropyX: float
    fBodyAccentropyY: float
    fBodyAccentropyZ: float
    fBodyAccmaxIndsX: float
    fBodyAccmaxIndsY: float
    fBodyAccmaxIndsZ: float
    fBodyAccmeanFreqX: float
    fBodyAccmeanFreqY: float
    fBodyAccmeanFreqZ: float
    fBodyAccskewnessX: float
    fBodyAcckurtosisX: float
    fBodyAccskewnessY: float
    fBodyAcckurtosisY: float
    fBodyAccskewnessZ: float
    fBodyAcckurtosisZ: float
    fBodyAccbandsEnergy18: float
    fBodyAccbandsEnergy916: float
    fBodyAccbandsEnergy1724: float
    fBodyAccbandsEnergy2532: float
    fBodyAccbandsEnergy3340: float
    fBodyAccbandsEnergy4148: float
    fBodyAccbandsEnergy4956: float
    fBodyAccbandsEnergy5764: float
    fBodyAccbandsEnergy116: float
    fBodyAccbandsEnergy1732: float
    fBodyAccbandsEnergy3348: float
    fBodyAccbandsEnergy4964: float
    fBodyAccbandsEnergy124: float
    fBodyAccbandsEnergy2548: float
    fBodyAccbandsEnergy181: float
    fBodyAccbandsEnergy9161: float
    fBodyAccbandsEnergy17241: float
    fBodyAccbandsEnergy25321: float
    fBodyAccbandsEnergy33401: float
    fBodyAccbandsEnergy41481: float
    fBodyAccbandsEnergy49561: float
    fBodyAccbandsEnergy57641: float
    fBodyAccbandsEnergy1161: float
    fBodyAccbandsEnergy17321: float
    fBodyAccbandsEnergy33481: float
    fBodyAccbandsEnergy49641: float
    fBodyAccbandsEnergy1241: float
    fBodyAccbandsEnergy25481: float
    fBodyAccbandsEnergy182: float
    fBodyAccbandsEnergy9162: float
    fBodyAccbandsEnergy17242: float
    fBodyAccbandsEnergy25322: float
    fBodyAccbandsEnergy33402: float
    fBodyAccbandsEnergy41482: float
    fBodyAccbandsEnergy49562: float
    fBodyAccbandsEnergy57642: float
    fBodyAccbandsEnergy1162: float
    fBodyAccbandsEnergy17322: float
    fBodyAccbandsEnergy33482: float
    fBodyAccbandsEnergy49642: float
    fBodyAccbandsEnergy1242: float
    fBodyAccbandsEnergy25482: float
    fBodyAccJerkmeanX: float
    fBodyAccJerkmeanY: float
    fBodyAccJerkmeanZ: float
    fBodyAccJerkstdX: float
    fBodyAccJerkstdY: float
    fBodyAccJerkstdZ: float
    fBodyAccJerkmadX: float
    fBodyAccJerkmadY: float
    fBodyAccJerkmadZ: float
    fBodyAccJerkmaxX: float
    fBodyAccJerkmaxY: float
    fBodyAccJerkmaxZ: float
    fBodyAccJerkminX: float
    fBodyAccJerkminY: float
    fBodyAccJerkminZ: float
    fBodyAccJerksma: float
    fBodyAccJerkenergyX: float
    fBodyAccJerkenergyY: float
    fBodyAccJerkenergyZ: float
    fBodyAccJerkiqrX: float
    fBodyAccJerkiqrY: float
    fBodyAccJerkiqrZ: float
    fBodyAccJerkentropyX: float
    fBodyAccJerkentropyY: float
    fBodyAccJerkentropyZ: float
    fBodyAccJerkmaxIndsX: float
    fBodyAccJerkmaxIndsY: float
    fBodyAccJerkmaxIndsZ: float
    fBodyAccJerkmeanFreqX: float
    fBodyAccJerkmeanFreqY: float
    fBodyAccJerkmeanFreqZ: float
    fBodyAccJerkskewnessX: float
    fBodyAccJerkkurtosisX: float
    fBodyAccJerkskewnessY: float
    fBodyAccJerkkurtosisY: float
    fBodyAccJerkskewnessZ: float
    fBodyAccJerkkurtosisZ: float
    fBodyAccJerkbandsEnergy18: float
    fBodyAccJerkbandsEnergy916: float
    fBodyAccJerkbandsEnergy1724: float
    fBodyAccJerkbandsEnergy2532: float
    fBodyAccJerkbandsEnergy3340: float
    fBodyAccJerkbandsEnergy4148: float
    fBodyAccJerkbandsEnergy4956: float
    fBodyAccJerkbandsEnergy5764: float
    fBodyAccJerkbandsEnergy116: float
    fBodyAccJerkbandsEnergy1732: float
    fBodyAccJerkbandsEnergy3348: float
    fBodyAccJerkbandsEnergy4964: float
    fBodyAccJerkbandsEnergy124: float
    fBodyAccJerkbandsEnergy2548: float
    fBodyAccJerkbandsEnergy181: float
    fBodyAccJerkbandsEnergy9161: float
    fBodyAccJerkbandsEnergy17241: float
    fBodyAccJerkbandsEnergy25321: float
    fBodyAccJerkbandsEnergy33401: float
    fBodyAccJerkbandsEnergy41481: float
    fBodyAccJerkbandsEnergy49561: float
    fBodyAccJerkbandsEnergy57641: float
    fBodyAccJerkbandsEnergy1161: float
    fBodyAccJerkbandsEnergy17321: float
    fBodyAccJerkbandsEnergy33481: float
    fBodyAccJerkbandsEnergy49641: float
    fBodyAccJerkbandsEnergy1241: float
    fBodyAccJerkbandsEnergy25481: float
    fBodyAccJerkbandsEnergy182: float
    fBodyAccJerkbandsEnergy9162: float
    fBodyAccJerkbandsEnergy17242: float
    fBodyAccJerkbandsEnergy25322: float
    fBodyAccJerkbandsEnergy33402: float
    fBodyAccJerkbandsEnergy41482: float
    fBodyAccJerkbandsEnergy49562: float
    fBodyAccJerkbandsEnergy57642: float
    fBodyAccJerkbandsEnergy1162: float
    fBodyAccJerkbandsEnergy17322: float
    fBodyAccJerkbandsEnergy33482: float
    fBodyAccJerkbandsEnergy49642: float
    fBodyAccJerkbandsEnergy1242: float
    fBodyAccJerkbandsEnergy25482: float
    fBodyGyromeanX: float
    fBodyGyromeanY: float
    fBodyGyromeanZ: float
    fBodyGyrostdX: float
    fBodyGyrostdY: float
    fBodyGyrostdZ: float
    fBodyGyromadX: float
    fBodyGyromadY: float
    fBodyGyromadZ: float
    fBodyGyromaxX: float
    fBodyGyromaxY: float
    fBodyGyromaxZ: float
    fBodyGyrominX: float
    fBodyGyrominY: float
    fBodyGyrominZ: float
    fBodyGyrosma: float
    fBodyGyroenergyX: float
    fBodyGyroenergyY: float
    fBodyGyroenergyZ: float
    fBodyGyroiqrX: float
    fBodyGyroiqrY: float
    fBodyGyroiqrZ: float
    fBodyGyroentropyX: float
    fBodyGyroentropyY: float
    fBodyGyroentropyZ: float
    fBodyGyromaxIndsX: float
    fBodyGyromaxIndsY: float
    fBodyGyromaxIndsZ: float
    fBodyGyromeanFreqX: float
    fBodyGyromeanFreqY: float
    fBodyGyromeanFreqZ: float
    fBodyGyroskewnessX: float
    fBodyGyrokurtosisX: float
    fBodyGyroskewnessY: float
    fBodyGyrokurtosisY: float
    fBodyGyroskewnessZ: float
    fBodyGyrokurtosisZ: float
    fBodyGyrobandsEnergy18: float
    fBodyGyrobandsEnergy916: float
    fBodyGyrobandsEnergy1724: float
    fBodyGyrobandsEnergy2532: float
    fBodyGyrobandsEnergy3340: float
    fBodyGyrobandsEnergy4148: float
    fBodyGyrobandsEnergy4956: float
    fBodyGyrobandsEnergy5764: float
    fBodyGyrobandsEnergy116: float
    fBodyGyrobandsEnergy1732: float
    fBodyGyrobandsEnergy3348: float
    fBodyGyrobandsEnergy4964: float
    fBodyGyrobandsEnergy124: float
    fBodyGyrobandsEnergy2548: float
    fBodyGyrobandsEnergy181: float
    fBodyGyrobandsEnergy9161: float
    fBodyGyrobandsEnergy17241: float
    fBodyGyrobandsEnergy25321: float
    fBodyGyrobandsEnergy33401: float
    fBodyGyrobandsEnergy41481: float
    fBodyGyrobandsEnergy49561: float
    fBodyGyrobandsEnergy57641: float
    fBodyGyrobandsEnergy1161: float
    fBodyGyrobandsEnergy17321: float
    fBodyGyrobandsEnergy33481: float
    fBodyGyrobandsEnergy49641: float
    fBodyGyrobandsEnergy1241: float
    fBodyGyrobandsEnergy25481: float
    fBodyGyrobandsEnergy182: float
    fBodyGyrobandsEnergy9162: float
    fBodyGyrobandsEnergy17242: float
    fBodyGyrobandsEnergy25322: float
    fBodyGyrobandsEnergy33402: float
    fBodyGyrobandsEnergy41482: float
    fBodyGyrobandsEnergy49562: float
    fBodyGyrobandsEnergy57642: float
    fBodyGyrobandsEnergy1162: float
    fBodyGyrobandsEnergy17322: float
    fBodyGyrobandsEnergy33482: float
    fBodyGyrobandsEnergy49642: float
    fBodyGyrobandsEnergy1242: float
    fBodyGyrobandsEnergy25482: float
    fBodyAccMagmean: float
    fBodyAccMagstd: float
    fBodyAccMagmad: float
    fBodyAccMagmax: float
    fBodyAccMagmin: float
    fBodyAccMagsma: float
    fBodyAccMagenergy: float
    fBodyAccMagiqr: float
    fBodyAccMagentropy: float
    fBodyAccMagmaxInds: float
    fBodyAccMagmeanFreq: float
    fBodyAccMagskewness: float
    fBodyAccMagkurtosis: float
    fBodyBodyAccJerkMagmean: float
    fBodyBodyAccJerkMagstd: float
    fBodyBodyAccJerkMagmad: float
    fBodyBodyAccJerkMagmax: float
    fBodyBodyAccJerkMagmin: float
    fBodyBodyAccJerkMagsma: float
    fBodyBodyAccJerkMagenergy: float
    fBodyBodyAccJerkMagiqr: float
    fBodyBodyAccJerkMagentropy: float
    fBodyBodyAccJerkMagmaxInds: float
    fBodyBodyAccJerkMagmeanFreq: float
    fBodyBodyAccJerkMagskewness: float
    fBodyBodyAccJerkMagkurtosis: float
    fBodyBodyGyroMagmean: float
    fBodyBodyGyroMagstd: float
    fBodyBodyGyroMagmad: float
    fBodyBodyGyroMagmax: float
    fBodyBodyGyroMagmin: float
    fBodyBodyGyroMagsma: float
    fBodyBodyGyroMagenergy: float
    fBodyBodyGyroMagiqr: float
    fBodyBodyGyroMagentropy: float
    fBodyBodyGyroMagmaxInds: float
    fBodyBodyGyroMagmeanFreq: float
    fBodyBodyGyroMagskewness: float
    fBodyBodyGyroMagkurtosis: float
    fBodyBodyGyroJerkMagmean: float
    fBodyBodyGyroJerkMagstd: float
    fBodyBodyGyroJerkMagmad: float
    fBodyBodyGyroJerkMagmax: float
    fBodyBodyGyroJerkMagmin: float
    fBodyBodyGyroJerkMagsma: float
    fBodyBodyGyroJerkMagenergy: float
    fBodyBodyGyroJerkMagiqr: float
    fBodyBodyGyroJerkMagentropy: float
    fBodyBodyGyroJerkMagmaxInds: float
    fBodyBodyGyroJerkMagmeanFreq: float
    fBodyBodyGyroJerkMagskewness: float
    fBodyBodyGyroJerkMagkurtosis: float
    angletBodyAccMeangravity: float
    angletBodyAccJerkMeangravityMean: float
    angletBodyGyroMeangravityMean: float
    angletBodyGyroJerkMeangravityMean: float
    angleXgravityMean: float
    angleYgravityMean: float
    angleZgravityMean: float

# # Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'Welcome to Khrishawns ML Deployment!'}


# # Defining path operation for /name endpoint
@app.post('/predict')
def predictor(data: BehaviourActivity):
    global prediction
    test_data = [[
        data.tBodyAccmeanX,
        data.tBodyAccmeanY,
        data.tBodyAccmeanZ,
        data.tBodyAccstdX,
        data.tBodyAccstdY,
        data.tBodyAccstdZ,
        data.tBodyAccmadX,
        data.tBodyAccmadY,
        data.tBodyAccmadZ,
        data.tBodyAccmaxX,
        data.tBodyAccmaxY,
        data.tBodyAccmaxZ,
        data.tBodyAccminX,
        data.tBodyAccminY,
        data.tBodyAccminZ,
        data.tBodyAccsma,
        data.tBodyAccenergyX,
        data.tBodyAccenergyY,
        data.tBodyAccenergyZ,
        data.tBodyAcciqrX,
        data.tBodyAcciqrY,
        data.tBodyAcciqrZ,
        data.tBodyAccentropyX,
        data.tBodyAccentropyY,
        data.tBodyAccentropyZ,
        data.tBodyAccarCoeffX1,
        data.tBodyAccarCoeffX2,
        data.tBodyAccarCoeffX3,
        data.tBodyAccarCoeffX4,
        data.tBodyAccarCoeffY1,
        data.tBodyAccarCoeffY2,
        data.tBodyAccarCoeffY3,
        data.tBodyAccarCoeffY4,
        data.tBodyAccarCoeffZ1,
        data.tBodyAccarCoeffZ2,
        data.tBodyAccarCoeffZ3,
        data.tBodyAccarCoeffZ4,
        data.tBodyAcccorrelationXY,
        data.tBodyAcccorrelationXZ,
        data.tBodyAcccorrelationYZ,
        data.tGravityAccmeanX,
        data.tGravityAccmeanY,
        data.tGravityAccmeanZ,
        data.tGravityAccstdX,
        data.tGravityAccstdY,
        data.tGravityAccstdZ,
        data.tGravityAccmadX,
        data.tGravityAccmadY,
        data.tGravityAccmadZ,
        data.tGravityAccmaxX,
        data.tGravityAccmaxY,
        data.tGravityAccmaxZ,
        data.tGravityAccminX,
        data.tGravityAccminY,
        data.tGravityAccminZ,
        data.tGravityAccsma,
        data.tGravityAccenergyX,
        data.tGravityAccenergyY,
        data.tGravityAccenergyZ,
        data.tGravityAcciqrX,
        data.tGravityAcciqrY,
        data.tGravityAcciqrZ,
        data.tGravityAccentropyX,
        data.tGravityAccentropyY,
        data.tGravityAccentropyZ,
        data.tGravityAccarCoeffX1,
        data.tGravityAccarCoeffX2,
        data.tGravityAccarCoeffX3,
        data.tGravityAccarCoeffX4,
        data.tGravityAccarCoeffY1,
        data.tGravityAccarCoeffY2,
        data.tGravityAccarCoeffY3,
        data.tGravityAccarCoeffY4,
        data.tGravityAccarCoeffZ1,
        data.tGravityAccarCoeffZ2,
        data.tGravityAccarCoeffZ3,
        data.tGravityAccarCoeffZ4,
        data.tGravityAcccorrelationXY,
        data.tGravityAcccorrelationXZ,
        data.tGravityAcccorrelationYZ,
        data.tBodyAccJerkmeanX,
        data.tBodyAccJerkmeanY,
        data.tBodyAccJerkmeanZ,
        data.tBodyAccJerkstdX,
        data.tBodyAccJerkstdY,
        data.tBodyAccJerkstdZ,
        data.tBodyAccJerkmadX,
        data.tBodyAccJerkmadY,
        data.tBodyAccJerkmadZ,
        data.tBodyAccJerkmaxX,
        data.tBodyAccJerkmaxY,
        data.tBodyAccJerkmaxZ,
        data.tBodyAccJerkminX,
        data.tBodyAccJerkminY,
        data.tBodyAccJerkminZ,
        data.tBodyAccJerksma,
        data.tBodyAccJerkenergyX,
        data.tBodyAccJerkenergyY,
        data.tBodyAccJerkenergyZ,
        data.tBodyAccJerkiqrX,
        data.tBodyAccJerkiqrY,
        data.tBodyAccJerkiqrZ,
        data.tBodyAccJerkentropyX,
        data.tBodyAccJerkentropyY,
        data.tBodyAccJerkentropyZ,
        data.tBodyAccJerkarCoeffX1,
        data.tBodyAccJerkarCoeffX2,
        data.tBodyAccJerkarCoeffX3,
        data.tBodyAccJerkarCoeffX4,
        data.tBodyAccJerkarCoeffY1,
        data.tBodyAccJerkarCoeffY2,
        data.tBodyAccJerkarCoeffY3,
        data.tBodyAccJerkarCoeffY4,
        data.tBodyAccJerkarCoeffZ1,
        data.tBodyAccJerkarCoeffZ2,
        data.tBodyAccJerkarCoeffZ3,
        data.tBodyAccJerkarCoeffZ4,
        data.tBodyAccJerkcorrelationXY,
        data.tBodyAccJerkcorrelationXZ,
        data.tBodyAccJerkcorrelationYZ,
        data.tBodyGyromeanX,
        data.tBodyGyromeanY,
        data.tBodyGyromeanZ,
        data.tBodyGyrostdX,
        data.tBodyGyrostdY,
        data.tBodyGyrostdZ,
        data.tBodyGyromadX,
        data.tBodyGyromadY,
        data.tBodyGyromadZ,
        data.tBodyGyromaxX,
        data.tBodyGyromaxY,
        data.tBodyGyromaxZ,
        data.tBodyGyrominX,
        data.tBodyGyrominY,
        data.tBodyGyrominZ,
        data.tBodyGyrosma,
        data.tBodyGyroenergyX,
        data.tBodyGyroenergyY,
        data.tBodyGyroenergyZ,
        data.tBodyGyroiqrX,
        data.tBodyGyroiqrY,
        data.tBodyGyroiqrZ,
        data.tBodyGyroentropyX,
        data.tBodyGyroentropyY,
        data.tBodyGyroentropyZ,
        data.tBodyGyroarCoeffX1,
        data.tBodyGyroarCoeffX2,
        data.tBodyGyroarCoeffX3,
        data.tBodyGyroarCoeffX4,
        data.tBodyGyroarCoeffY1,
        data.tBodyGyroarCoeffY2,
        data.tBodyGyroarCoeffY3,
        data.tBodyGyroarCoeffY4,
        data.tBodyGyroarCoeffZ1,
        data.tBodyGyroarCoeffZ2,
        data.tBodyGyroarCoeffZ3,
        data.tBodyGyroarCoeffZ4,
        data.tBodyGyrocorrelationXY,
        data.tBodyGyrocorrelationXZ,
        data.tBodyGyrocorrelationYZ,
        data.tBodyGyroJerkmeanX,
        data.tBodyGyroJerkmeanY,
        data.tBodyGyroJerkmeanZ,
        data.tBodyGyroJerkstdX,
        data.tBodyGyroJerkstdY,
        data.tBodyGyroJerkstdZ,
        data.tBodyGyroJerkmadX,
        data.tBodyGyroJerkmadY,
        data.tBodyGyroJerkmadZ,
        data.tBodyGyroJerkmaxX,
        data.tBodyGyroJerkmaxY,
        data.tBodyGyroJerkmaxZ,
        data.tBodyGyroJerkminX,
        data.tBodyGyroJerkminY,
        data.tBodyGyroJerkminZ,
        data.tBodyGyroJerksma,
        data.tBodyGyroJerkenergyX,
        data.tBodyGyroJerkenergyY,
        data.tBodyGyroJerkenergyZ,
        data.tBodyGyroJerkiqrX,
        data.tBodyGyroJerkiqrY,
        data.tBodyGyroJerkiqrZ,
        data.tBodyGyroJerkentropyX,
        data.tBodyGyroJerkentropyY,
        data.tBodyGyroJerkentropyZ,
        data.tBodyGyroJerkarCoeffX1,
        data.tBodyGyroJerkarCoeffX2,
        data.tBodyGyroJerkarCoeffX3,
        data.tBodyGyroJerkarCoeffX4,
        data.tBodyGyroJerkarCoeffY1,
        data.tBodyGyroJerkarCoeffY2,
        data.tBodyGyroJerkarCoeffY3,
        data.tBodyGyroJerkarCoeffY4,
        data.tBodyGyroJerkarCoeffZ1,
        data.tBodyGyroJerkarCoeffZ2,
        data.tBodyGyroJerkarCoeffZ3,
        data.tBodyGyroJerkarCoeffZ4,
        data.tBodyGyroJerkcorrelationXY,
        data.tBodyGyroJerkcorrelationXZ,
        data.tBodyGyroJerkcorrelationYZ,
        data.tBodyAccMagmean,
        data.tBodyAccMagstd,
        data.tBodyAccMagmad,
        data.tBodyAccMagmax,
        data.tBodyAccMagmin,
        data.tBodyAccMagsma,
        data.tBodyAccMagenergy,
        data.tBodyAccMagiqr,
        data.tBodyAccMagentropy,
        data.tBodyAccMagarCoeff1,
        data.tBodyAccMagarCoeff2,
        data.tBodyAccMagarCoeff3,
        data.tBodyAccMagarCoeff4,
        data.tGravityAccMagmean,
        data.tGravityAccMagstd,
        data.tGravityAccMagmad,
        data.tGravityAccMagmax,
        data.tGravityAccMagmin,
        data.tGravityAccMagsma,
        data.tGravityAccMagenergy,
        data.tGravityAccMagiqr,
        data.tGravityAccMagentropy,
        data.tGravityAccMagarCoeff1,
        data.tGravityAccMagarCoeff2,
        data.tGravityAccMagarCoeff3,
        data.tGravityAccMagarCoeff4,
        data.tBodyAccJerkMagmean,
        data.tBodyAccJerkMagstd,
        data.tBodyAccJerkMagmad,
        data.tBodyAccJerkMagmax,
        data.tBodyAccJerkMagmin,
        data.tBodyAccJerkMagsma,
        data.tBodyAccJerkMagenergy,
        data.tBodyAccJerkMagiqr,
        data.tBodyAccJerkMagentropy,
        data.tBodyAccJerkMagarCoeff1,
        data.tBodyAccJerkMagarCoeff2,
        data.tBodyAccJerkMagarCoeff3,
        data.tBodyAccJerkMagarCoeff4,
        data.tBodyGyroMagmean,
        data.tBodyGyroMagstd,
        data.tBodyGyroMagmad,
        data.tBodyGyroMagmax,
        data.tBodyGyroMagmin,
        data.tBodyGyroMagsma,
        data.tBodyGyroMagenergy,
        data.tBodyGyroMagiqr,
        data.tBodyGyroMagentropy,
        data.tBodyGyroMagarCoeff1,
        data.tBodyGyroMagarCoeff2,
        data.tBodyGyroMagarCoeff3,
        data.tBodyGyroMagarCoeff4,
        data.tBodyGyroJerkMagmean,
        data.tBodyGyroJerkMagstd,
        data.tBodyGyroJerkMagmad,
        data.tBodyGyroJerkMagmax,
        data.tBodyGyroJerkMagmin,
        data.tBodyGyroJerkMagsma,
        data.tBodyGyroJerkMagenergy,
        data.tBodyGyroJerkMagiqr,
        data.tBodyGyroJerkMagentropy,
        data.tBodyGyroJerkMagarCoeff1,
        data.tBodyGyroJerkMagarCoeff2,
        data.tBodyGyroJerkMagarCoeff3,
        data.tBodyGyroJerkMagarCoeff4,
        data.fBodyAccmeanX,
        data.fBodyAccmeanY,
        data.fBodyAccmeanZ,
        data.fBodyAccstdX,
        data.fBodyAccstdY,
        data.fBodyAccstdZ,
        data.fBodyAccmadX,
        data.fBodyAccmadY,
        data.fBodyAccmadZ,
        data.fBodyAccmaxX,
        data.fBodyAccmaxY,
        data.fBodyAccmaxZ,
        data.fBodyAccminX,
        data.fBodyAccminY,
        data.fBodyAccminZ,
        data.fBodyAccsma,
        data.fBodyAccenergyX,
        data.fBodyAccenergyY,
        data.fBodyAccenergyZ,
        data.fBodyAcciqrX,
        data.fBodyAcciqrY,
        data.fBodyAcciqrZ,
        data.fBodyAccentropyX,
        data.fBodyAccentropyY,
        data.fBodyAccentropyZ,
        data.fBodyAccmaxIndsX,
        data.fBodyAccmaxIndsY,
        data.fBodyAccmaxIndsZ,
        data.fBodyAccmeanFreqX,
        data.fBodyAccmeanFreqY,
        data.fBodyAccmeanFreqZ,
        data.fBodyAccskewnessX,
        data.fBodyAcckurtosisX,
        data.fBodyAccskewnessY,
        data.fBodyAcckurtosisY,
        data.fBodyAccskewnessZ,
        data.fBodyAcckurtosisZ,
        data.fBodyAccbandsEnergy18,
        data.fBodyAccbandsEnergy916,
        data.fBodyAccbandsEnergy1724,
        data.fBodyAccbandsEnergy2532,
        data.fBodyAccbandsEnergy3340,
        data.fBodyAccbandsEnergy4148,
        data.fBodyAccbandsEnergy4956,
        data.fBodyAccbandsEnergy5764,
        data.fBodyAccbandsEnergy116,
        data.fBodyAccbandsEnergy1732,
        data.fBodyAccbandsEnergy3348,
        data.fBodyAccbandsEnergy4964,
        data.fBodyAccbandsEnergy124,
        data.fBodyAccbandsEnergy2548,
        data.fBodyAccbandsEnergy181,
        data.fBodyAccbandsEnergy9161,
        data.fBodyAccbandsEnergy17241,
        data.fBodyAccbandsEnergy25321,
        data.fBodyAccbandsEnergy33401,
        data.fBodyAccbandsEnergy41481,
        data.fBodyAccbandsEnergy49561,
        data.fBodyAccbandsEnergy57641,
        data.fBodyAccbandsEnergy1161,
        data.fBodyAccbandsEnergy17321,
        data.fBodyAccbandsEnergy33481,
        data.fBodyAccbandsEnergy49641,
        data.fBodyAccbandsEnergy1241,
        data.fBodyAccbandsEnergy25481,
        data.fBodyAccbandsEnergy182,
        data.fBodyAccbandsEnergy9162,
        data.fBodyAccbandsEnergy17242,
        data.fBodyAccbandsEnergy25322,
        data.fBodyAccbandsEnergy33402,
        data.fBodyAccbandsEnergy41482,
        data.fBodyAccbandsEnergy49562,
        data.fBodyAccbandsEnergy57642,
        data.fBodyAccbandsEnergy1162,
        data.fBodyAccbandsEnergy17322,
        data.fBodyAccbandsEnergy33482,
        data.fBodyAccbandsEnergy49642,
        data.fBodyAccbandsEnergy1242,
        data.fBodyAccbandsEnergy25482,
        data.fBodyAccJerkmeanX,
        data.fBodyAccJerkmeanY,
        data.fBodyAccJerkmeanZ,
        data.fBodyAccJerkstdX,
        data.fBodyAccJerkstdY,
        data.fBodyAccJerkstdZ,
        data.fBodyAccJerkmadX,
        data.fBodyAccJerkmadY,
        data.fBodyAccJerkmadZ,
        data.fBodyAccJerkmaxX,
        data.fBodyAccJerkmaxY,
        data.fBodyAccJerkmaxZ,
        data.fBodyAccJerkminX,
        data.fBodyAccJerkminY,
        data.fBodyAccJerkminZ,
        data.fBodyAccJerksma,
        data.fBodyAccJerkenergyX,
        data.fBodyAccJerkenergyY,
        data.fBodyAccJerkenergyZ,
        data.fBodyAccJerkiqrX,
        data.fBodyAccJerkiqrY,
        data.fBodyAccJerkiqrZ,
        data.fBodyAccJerkentropyX,
        data.fBodyAccJerkentropyY,
        data.fBodyAccJerkentropyZ,
        data.fBodyAccJerkmaxIndsX,
        data.fBodyAccJerkmaxIndsY,
        data.fBodyAccJerkmaxIndsZ,
        data.fBodyAccJerkmeanFreqX,
        data.fBodyAccJerkmeanFreqY,
        data.fBodyAccJerkmeanFreqZ,
        data.fBodyAccJerkskewnessX,
        data.fBodyAccJerkkurtosisX,
        data.fBodyAccJerkskewnessY,
        data.fBodyAccJerkkurtosisY,
        data.fBodyAccJerkskewnessZ,
        data.fBodyAccJerkkurtosisZ,
        data.fBodyAccJerkbandsEnergy18,
        data.fBodyAccJerkbandsEnergy916,
        data.fBodyAccJerkbandsEnergy1724,
        data.fBodyAccJerkbandsEnergy2532,
        data.fBodyAccJerkbandsEnergy3340,
        data.fBodyAccJerkbandsEnergy4148,
        data.fBodyAccJerkbandsEnergy4956,
        data.fBodyAccJerkbandsEnergy5764,
        data.fBodyAccJerkbandsEnergy116,
        data.fBodyAccJerkbandsEnergy1732,
        data.fBodyAccJerkbandsEnergy3348,
        data.fBodyAccJerkbandsEnergy4964,
        data.fBodyAccJerkbandsEnergy124,
        data.fBodyAccJerkbandsEnergy2548,
        data.fBodyAccJerkbandsEnergy181,
        data.fBodyAccJerkbandsEnergy9161,
        data.fBodyAccJerkbandsEnergy17241,
        data.fBodyAccJerkbandsEnergy25321,
        data.fBodyAccJerkbandsEnergy33401,
        data.fBodyAccJerkbandsEnergy41481,
        data.fBodyAccJerkbandsEnergy49561,
        data.fBodyAccJerkbandsEnergy57641,
        data.fBodyAccJerkbandsEnergy1161,
        data.fBodyAccJerkbandsEnergy17321,
        data.fBodyAccJerkbandsEnergy33481,
        data.fBodyAccJerkbandsEnergy49641,
        data.fBodyAccJerkbandsEnergy1241,
        data.fBodyAccJerkbandsEnergy25481,
        data.fBodyAccJerkbandsEnergy182,
        data.fBodyAccJerkbandsEnergy9162,
        data.fBodyAccJerkbandsEnergy17242,
        data.fBodyAccJerkbandsEnergy25322,
        data.fBodyAccJerkbandsEnergy33402,
        data.fBodyAccJerkbandsEnergy41482,
        data.fBodyAccJerkbandsEnergy49562,
        data.fBodyAccJerkbandsEnergy57642,
        data.fBodyAccJerkbandsEnergy1162,
        data.fBodyAccJerkbandsEnergy17322,
        data.fBodyAccJerkbandsEnergy33482,
        data.fBodyAccJerkbandsEnergy49642,
        data.fBodyAccJerkbandsEnergy1242,
        data.fBodyAccJerkbandsEnergy25482,
        data.fBodyGyromeanX,
        data.fBodyGyromeanY,
        data.fBodyGyromeanZ,
        data.fBodyGyrostdX,
        data.fBodyGyrostdY,
        data.fBodyGyrostdZ,
        data.fBodyGyromadX,
        data.fBodyGyromadY,
        data.fBodyGyromadZ,
        data.fBodyGyromaxX,
        data.fBodyGyromaxY,
        data.fBodyGyromaxZ,
        data.fBodyGyrominX,
        data.fBodyGyrominY,
        data.fBodyGyrominZ,
        data.fBodyGyrosma,
        data.fBodyGyroenergyX,
        data.fBodyGyroenergyY,
        data.fBodyGyroenergyZ,
        data.fBodyGyroiqrX,
        data.fBodyGyroiqrY,
        data.fBodyGyroiqrZ,
        data.fBodyGyroentropyX,
        data.fBodyGyroentropyY,
        data.fBodyGyroentropyZ,
        data.fBodyGyromaxIndsX,
        data.fBodyGyromaxIndsY,
        data.fBodyGyromaxIndsZ,
        data.fBodyGyromeanFreqX,
        data.fBodyGyromeanFreqY,
        data.fBodyGyromeanFreqZ,
        data.fBodyGyroskewnessX,
        data.fBodyGyrokurtosisX,
        data.fBodyGyroskewnessY,
        data.fBodyGyrokurtosisY,
        data.fBodyGyroskewnessZ,
        data.fBodyGyrokurtosisZ,
        data.fBodyGyrobandsEnergy18,
        data.fBodyGyrobandsEnergy916,
        data.fBodyGyrobandsEnergy1724,
        data.fBodyGyrobandsEnergy2532,
        data.fBodyGyrobandsEnergy3340,
        data.fBodyGyrobandsEnergy4148,
        data.fBodyGyrobandsEnergy4956,
        data.fBodyGyrobandsEnergy5764,
        data.fBodyGyrobandsEnergy116,
        data.fBodyGyrobandsEnergy1732,
        data.fBodyGyrobandsEnergy3348,
        data.fBodyGyrobandsEnergy4964,
        data.fBodyGyrobandsEnergy124,
        data.fBodyGyrobandsEnergy2548,
        data.fBodyGyrobandsEnergy181,
        data.fBodyGyrobandsEnergy9161,
        data.fBodyGyrobandsEnergy17241,
        data.fBodyGyrobandsEnergy25321,
        data.fBodyGyrobandsEnergy33401,
        data.fBodyGyrobandsEnergy41481,
        data.fBodyGyrobandsEnergy49561,
        data.fBodyGyrobandsEnergy57641,
        data.fBodyGyrobandsEnergy1161,
        data.fBodyGyrobandsEnergy17321,
        data.fBodyGyrobandsEnergy33481,
        data.fBodyGyrobandsEnergy49641,
        data.fBodyGyrobandsEnergy1241,
        data.fBodyGyrobandsEnergy25481,
        data.fBodyGyrobandsEnergy182,
        data.fBodyGyrobandsEnergy9162,
        data.fBodyGyrobandsEnergy17242,
        data.fBodyGyrobandsEnergy25322,
        data.fBodyGyrobandsEnergy33402,
        data.fBodyGyrobandsEnergy41482,
        data.fBodyGyrobandsEnergy49562,
        data.fBodyGyrobandsEnergy57642,
        data.fBodyGyrobandsEnergy1162,
        data.fBodyGyrobandsEnergy17322,
        data.fBodyGyrobandsEnergy33482,
        data.fBodyGyrobandsEnergy49642,
        data.fBodyGyrobandsEnergy1242,
        data.fBodyGyrobandsEnergy25482,
        data.fBodyAccMagmean,
        data.fBodyAccMagstd,
        data.fBodyAccMagmad,
        data.fBodyAccMagmax,
        data.fBodyAccMagmin,
        data.fBodyAccMagsma,
        data.fBodyAccMagenergy,
        data.fBodyAccMagiqr,
        data.fBodyAccMagentropy,
        data.fBodyAccMagmaxInds,
        data.fBodyAccMagmeanFreq,
        data.fBodyAccMagskewness,
        data.fBodyAccMagkurtosis,
        data.fBodyBodyAccJerkMagmean,
        data.fBodyBodyAccJerkMagstd,
        data.fBodyBodyAccJerkMagmad,
        data.fBodyBodyAccJerkMagmax,
        data.fBodyBodyAccJerkMagmin,
        data.fBodyBodyAccJerkMagsma,
        data.fBodyBodyAccJerkMagenergy,
        data.fBodyBodyAccJerkMagiqr,
        data.fBodyBodyAccJerkMagentropy,
        data.fBodyBodyAccJerkMagmaxInds,
        data.fBodyBodyAccJerkMagmeanFreq,
        data.fBodyBodyAccJerkMagskewness,
        data.fBodyBodyAccJerkMagkurtosis,
        data.fBodyBodyGyroMagmean,
        data.fBodyBodyGyroMagstd,
        data.fBodyBodyGyroMagmad,
        data.fBodyBodyGyroMagmax,
        data.fBodyBodyGyroMagmin,
        data.fBodyBodyGyroMagsma,
        data.fBodyBodyGyroMagenergy,
        data.fBodyBodyGyroMagiqr,
        data.fBodyBodyGyroMagentropy,
        data.fBodyBodyGyroMagmaxInds,
        data.fBodyBodyGyroMagmeanFreq,
        data.fBodyBodyGyroMagskewness,
        data.fBodyBodyGyroMagkurtosis,
        data.fBodyBodyGyroJerkMagmean,
        data.fBodyBodyGyroJerkMagstd,
        data.fBodyBodyGyroJerkMagmad,
        data.fBodyBodyGyroJerkMagmax,
        data.fBodyBodyGyroJerkMagmin,
        data.fBodyBodyGyroJerkMagsma,
        data.fBodyBodyGyroJerkMagenergy,
        data.fBodyBodyGyroJerkMagiqr,
        data.fBodyBodyGyroJerkMagentropy,
        data.fBodyBodyGyroJerkMagmaxInds,
        data.fBodyBodyGyroJerkMagmeanFreq,
        data.fBodyBodyGyroJerkMagskewness,
        data.fBodyBodyGyroJerkMagkurtosis,
        data.angletBodyAccMeangravity,
        data.angletBodyAccJerkMeangravityMean,
        data.angletBodyGyroMeangravityMean,
        data.angletBodyGyroJerkMeangravityMean,
        data.angleXgravityMean,
        data.angleYgravityMean,
        data.angleZgravityMean,
    ]]

    #  [1] End of source
    class_label = model.predict(test_data)[0]
 

    # If the encoded numbers come back what Activity is it?
    if(class_label == 0):
        prediction = 'Laying'
    if (class_label == 1):
        prediction = 'Sitting'
    if ( class_label == 2):
        prediction = 'Standing'
    if ( class_label == 3):
        prediction = 'Walking'
    if (class_label ==4):
        prediction = 'Walking Downstairs/ 4'
    if (class_label == 5):
        prediction = 'Standing/ 5'

    return {prediction}

