# 50.039 DL-Small-Project
This is a project to propose, train and evaluate a Deep Learning model, which attempts to classify the X-ray images of patients and help doctors with the diagnosis of COVID/non-COVID pneumonia. We have trained 2 models using the three-classes classifier and two binary classifiers architecture.



## Project Setup (The following libraries are needed)
1. Python 
    * version: 3.7
2. Pytorch 
    * version: 1.7.1
3. Matplotlib 
    * version: 3.2.2
4. Time
5. Numpy
# 
## To retrain the models from scratch
`Click Run all from mainCode.ipynb `


The newly trained models will be saved in the __model__ folder

# 

## To load the saved models
`Run the cells at the corresponding section of mainCode.ipynb`
* Three classes classifier
    * under the section named: `Load the trained model of the three classes classifier`
* 2 binary classifier
    * Under the section named: `Load the trained model of the 2 binary classifier`
The saved models are saved under the __final_saved_models__ folder