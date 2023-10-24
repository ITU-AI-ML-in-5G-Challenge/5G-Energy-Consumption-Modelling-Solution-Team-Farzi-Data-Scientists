# AI/ML for 5G-Energy Consumption Modelling by ITU AI/ML in 5G Challenge

## 5G Energy Consumption Modelling [RANK 1](https://zindi.africa/competitions/aiml-for-5g-energy-consumption-modelling/leaderboard) Solution Team [Farzi Data Scientists](https://zindi.africa/competitions/aiml-for-5g-energy-consumption-modelling/leaderboard/teams/farzi_data_scientists)

## [Problem Statement](https://zindi.africa/competitions/aiml-for-5g-energy-consumption-modelling)
5G, the fifth generation of radio technology, has brought about new services, technologies, and networking paradigms, with the corresponding social benefits. However, there is growing concern over the energy consumption of these new network deployments. While 5G networks are estimated to be about 4x more energy-efficient than 4G networks, their energy consumption is approximately 3x larger due to the need for a larger number of cells to provide the same coverage at higher frequencies and the increased processing required for wider bandwidths and more antennas.

Base station energy consumption depends on multiple factors, such as specific architecture (e.g. RRU or AAU), configuration parameters (e.g., number of operated carriers, bandwidth, transmit power), traffic conditions (e.g., number of allocated physical resource blocks), and the activation of energy-saving methods (e.g., symbol shutdown, RF shutdown). To reduce network energy consumption, it is crucial to optimize base station parameters and energy-saving methods. This requires a deep understanding of how these parameters and methods impact the energy consumption of different base stations. Therefore, accurate modelling of energy consumption is essential for achieving more energy-efficient network deployments.

![](https://zindi-public-release.s3.eu-west-2.amazonaws.com/uploads/image_attachment/image/1943/738561f9-bec5-423d-be50-154fc829046b.png)

## Objective
This ML challenge targets addressing the important questions mentioned above. In the challenge, the participants are asked to design a machine learning-based solution that can be trained on a dataset of few scenarios and then generalize successfully to data from scenarios not seen before. In particular, the designed machine learning model must be able to achieve the following objectives.

* **Objective A:** Develop a model able to estimate the energy consumed by different base station products. The participants are required to develop a model that estimates the energy consumed by different base station products, taking into consideration the impact of various engineering configurations, traffic conditions, and energy-saving methods.

* **Objective B:** Achieve generalization capabilities across different base station products. The model must estimate the energy consumption of a new base station product based on measurements collected from existing ones, such as Products A, B, and C. For example, if training data is available for these three products, the model must be able to provide an estimate of the energy consumed by Product D.

* **Objective C:** Achieve generalization capabilities across different base station configurations. The model must predict the energy consumption of newly configured parameters based on a small number of real network configuration parameters. For instance, if the training data contains samples collected from many base station products, when the transmit power is set to 30, 35, and 43 dBm, the model must estimate the energy consumed when the transmit power is set to 40 dBm.

## Metrics
To focus on the cross-equipment and cross-configuration generalization capability of the model, the test set estimation accuracy is evaluated by using a weighted relative error evaluation method. Specifically, the error weight, w_i, of the sample corresponding to the new device and/or new configuration in the test set is larger and is provided in the test set.

The final model performance is ranked according to the **minimum WMAPE error**

![](https://zindi-public-release.s3.eu-west-2.amazonaws.com/uploads/image_attachment/image/1944/584d2bb7-8d36-4a32-91b2-4ce603be6bbf.png)
 
### Things to Learn
```text
* Regression Problem Statement to Optimize (Predict) Energy Consumption
* Understanding Domin Knowledge in 5G Data and Energy Consumption
* ANN Architecture along with Activation Functions
* Optimizing MAE/ MAPE/ WMAPE
* Usage of NO FUTURE VALUES
```

## Solution Approach : [Report - 5G-Energy Consumption Modelling by ITU.pdf](https://github.com/ITU-AI-ML-in-5G-Challenge/5G-Energy-Consumption-Modelling-Solution-Team-Farzi-Data-Scientists/blob/main/Report%20-%205G-Energy%20Consumption%20Modelling%20by%20ITU.pdf)

## Steps to reproduce

#### Kaggle Environment:
1. Accelerator: **GPU P100**
2. Language: **Python**
3. Persistence: **No Persistence**
4. Environment: **Pin to original environment (2023-09-07)**
5. Internet: **on**
6. The above values are NOT all default so please make sure by checking once. Please TURN ON the GPU.

### Packages
```text
* pip install numpy==1.23.5
* pip install pandas==2.0.3
* pip install fastai==2.7.12
* pip install plotly==5.17.0
* pip install plotnine==0.12.3
* pip install fastinference==0.0.36
* pip install scipy==1.7.3
* pip install sklearn==1.2.2
* pip install torch==2.0.0
* pip install tensorflow==2.12.0
* pip install shap==0.42.1
```
**Note:** Make sure to run all the codes in the kaggle GPU environment for reproducibility and RAM out of memory error issue as I have used kaggle to code because of more RAM allotment.


### Steps
* Open a kaggle notebook
  * Add data (your data will remain private, so no worries) *Data folder is attached in the solution folder (you can directly upload the folder or the zipped version, also attached in the solution folder*
* **Dataset** - **ecm-itu-zindi-kp-data**  This folder contains training data downloaded from ITU website
  * *Imgs_202307101549519358.csv*
  * *Imgs_2023071012123392536.csv*
  * *Imgs_2023071012130978799.csv*
  * *imgs_2023071012133740345.csv*
Test data downloaded from zindi
  * *SampleSubmission,csv*
If you decide to download and create your own folder, make sure to place both training and test in same folder and change path in the notebook

#### Notebooks

* **ecm-zindi-kp-v4-training-and-prediction-notebook.ipynb** This file contains the training and evaluation code for the above competititon using Fast AI and Keras in which only past values are used for modelling along with raw and engineered features.
  
    ```text
    * Local CV & Score
        * FastAI ANN:
          * Local GroupKFold OOF MAPE: 0.029
          * Public Leaderboard: 0.0558
          * Private Leaderboard: 0.0549
        * Keras ANN:                            [Final Submission]
          * Local GroupKFold OOF MAPE: 0.026 
          * Public Leaderboard: 0.0434
          * Private Leaderboard: 0.0435
        * Mean Weighted Ensemble:
          * Local GroupKFold OOF MAPE: 0.0259
          * Public Leaderboard: 0.0474
          * Private Leaderboard: 0.0470
    * Solution Files
        * submission_fastai.csv
        * submission_keras.csv                  [Final Submission]
        * submission_ensemble.csv
        * Submission_ensemble_hm.csv
    ```
* **shap-analysis-model-explanation.ipynb**
  This notebook is for the organizers to understand the models trained. You NEEDNOT run it to match the score on LB. There is a separate notebook for this because shap analysis for ANN models take a lot of time to run. The     input to this notebook is again the input data and the output models of above training notebook. If you decide to run this. Please change the paths in the code accordingly.


## Contributors
1. [Krishna Priya](https://www.linkedin.com/in/krishnapriya18/)
2. [Rajat Ranjan](https://www.linkedin.com/in/rajat-ranjan24/)
