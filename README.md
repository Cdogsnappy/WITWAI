# Where in the World am I?
Approaching GeoLocation with Machine Learning

## Authors
- Nick Boddy
- Caleb Raschka
- Morgan Turville-Heitz

This project was made for CS 766 Computer Vision at UW-Madison.
[Project Presentation](https://docs.google.com/presentation/d/1akX8-ajRWpmPtD93Bkqu015D39DX9NZOcdgDnppzA8Q)
[GitHub](https://github.com/Cdogsnappy/WITWAI)

---

## Motivations
[GeoGuessr](https://www.geoguessr.com/) is a web-based game wherein the player is dropped into random places in the world and is given visual information via Google Street View. Using the information presented, the player must make a guess as to the coordinates of the image by placing a pin on the globe.

Due to the game's popularity, a competitive scene has arisen in which certain players have become more or less experts at GeoLocation. They have played many games and tend to be capable at:
1. Identifying language or script
2. Identifying biome have internalized distinguishing features 

Website:

The final project report must be submitted to Canvas as a website. The website must include:

- Project title and team members
- Link to presentation slides
- Link to github repository.
- Intro presenting and motivating the problem
- Your methodology (method, data, evaluation metrics). If applicable, highlight how it differs from prior work (new experiments, new methods, etc)

# Methodology 

### Overview 

We have implemented a multi-model pipeline in order to classify the country label of each image. This is utilizing an OCR text recognition model, a SotA LLM for language identification, an image segmentation model, and a deep feedforward network. All code is written in python, using either Jupyter notebooks or Python scripts. Implementation of the FFNN is handled in PyTorch. 

### Dataset(s)

The initial dataset that was used during our pipeline testing is a set of 10,000 images and scraped from google street view. These can be found at [Google Street View](https://www.kaggle.com/datasets/paulchambaz/google-street-view/data) - the script *kaggle_data_prep.ipynb* can be used to collect this dataset. Images are 640x640. We found a very low density of text, as well as a low density of segmentable features, in this dataset, so this was only used for testing the initial implementation of our models, and was not used during model training. 

For the model training, we attempted to collect two separate datasets:
1. Google Maps Street View (gmaps) Data
2. OpenStreetMap API - Mapillary (mapillary) Data

The gmaps dataset is collected by the code in the later half of the *mapillary_google_maps_data_prep.ipynb* script. The data collection is performed as follows:

1) We load a GeoPandas object (*world*) containing the polygon representation of countries. This maps a (longitude/latitude) pair to a specific country ID.
2) We join *world* with a new GeoPandas object containing the Urban Areas polygon database, *urban_areas*. This is the Global Human Settlement Layer R2019A dataset [GHS-FUA R2019A](https://human-settlement.emergency.copernicus.eu/ghs_fua.php). By joining the two objects, we are able to restrict sampling to urban areas, in order to target higher-information images when sampling.
3) We then sample 66% of images from "Urban areas" and 33% of images from "Rural areas," as defined above.
4) Sampling of coordinates is performed as follows:
- a. We randomly generate a (lat/lon) pair.
- b. We create a Shapely.geometry point from the pair.
- c. We join the Shapely point with either the *urban_areas* object or the *world* object, depending on if the sample is intended to be rural or urban.
- d. We check if the joined Shapely point is found within the selected object, and if so, we then check if the point is in Europe.
- e. If the object is in Europe, we return the (lat/lon) pair and the country as a label for the datapoint.
5) If the (lat/lon) sampling was succesful, we have to perform an important step - identifying an address that we can query the API for.
- a. We use a gmaps object with the function *reverse_geocode* to identify an address based on a (lat/lon) pair.
- b. If there is a nearby address within ~5 kilometers, the closest address is returned. 
- c. This address then needs to be parsed, in order to query the API for an image in step 6). 
6) If the address sampling was succesful, we query the Google Maps API, and query the API for an image. This functions as follows: 
- a. We use a standard HTTP query, with the address as an argument in the HTTP request.
- b. If we receive a 200 response code (i.e. succesful query), and if the image is not empty (image size > 10kb), we store the image. A non-negligible number of images result in empty returns, which are blank with an image filesize ~5kb.
7) Finally, we store the (lat/lon) and country ID in a .csv file, and store the image with the filename corresponding to the dataset's index. 

### Data Processing



### OCR Language Guessing

### Segmentation

### FFNN



### Quantitative Results
Our model exhibits fairly unstable behavior under our current data and parameters. For our evaluation metrics, we 
simply used 0-1 accuracy. When considering our options for metrics, we decided that a notion of proximity would be ignored, 
contrary to the rules of GeoGuessr. Thus, the model loss is the same when it guesses a country 50 km away that it would be if it
guessed a country 500 km away. This is reflected in our truth vectors mentioned in Methodology. 

The model does not train well under these conditions, and is very sensitive to initialization. We found that when training for too long 
the model quickly experienced mode collapse, and so a small amount of training epochs are used. Without mode collapse, we get varying levels of accuracy.
The saved model included reported an accuracy of ~6%. This is of course not a great result, but it is better than randomly guessing, 
as we have 23 classes which would have a random guess accuracy of ~4.5%.
- (Important) Demos of your approach.

### FFNN Training and Testing
Prerequisites: OCR Training + Segmentation Training

There are some auxiliary files that won't be necessary for use.

1. [ClassFrequency.py](Classifier/ClassFrequency.py) is used to calculate the frequency of each class in the data. This was used for data
normalization and is not necessary for training.

2. [TestSetBuilder.py](Classifier/TestSetBuilder.py) is used to build a test and train dataset using all of the data. This does not account
for class imbalance and thus will produce completely unstable training that will result in mode collapse.

Some files will be used automatically and won't need to be called by a user.

1. [CustomImageDataset.py](Classifier/CustomImageDataset.py) is an implementation of PyTorch's [Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) module which can
take the data for training and testing and feed it to the neural network for us.

2. [final_net.py](Classifier/final_net.py) has been wrapped inside two files, ModelTester and ModelTrainer, and thus doesn't need to 
be used for reproducing results. However, any model tuning would be done in final_net.

### How to Train
1. Run [BalancedSetBuilder.py](Classifier/BalancedSetBuilder.py)  
This script will build a dataset with balanced classes for the model to run on. The
data_size parameter determines how many data points for each country to use. Note that
countries with < data_size samples will be skipped, and so the used_countries.csv file must be updated
to reflect the countries that are present in the data.
2. Run [ModelTrainer.py](Classifier/ModelTrainer.py)  
Yes, it's that easy. If you are content with the model 
parameters of final_net.py, then go ahead and run ModelTrainer. This will run the model for the 
given number of epochs, using the given batch size, and will test the model on the test_data generated from
BalancedSetBuilder. If you do wish to alter training hyperparameters, they can be found here in [final_net.py](Classifier/final_net.py):
![](PageFiles/final_net_parameters.png)
Note that the training parameter learning rate cannot be decoupled for the two submodels in the FFNN; The weights must be updated simultaneously through
backpropagation as there are no ways to train the intermediate results of the FCN otherwise. You can also alter additional parameters such as model depth in the FFNN class definition, but note that performance will be impacted.

3. Run [ModelTester.py](Classifier/ModelTester.py)  
You can specify a saved model file (.pt) to use for testing on the 
current test dataset. This will reproduce the results that ModelTrainer outputs at the end of training
if you run it right after ModelTrainer, but if you run BalancedDataSetBuilder again
then the saved model will be tested on a new test set.





