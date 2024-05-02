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



- Discussion of quantitative results
- (Important) Demos of your approach.
