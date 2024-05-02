# Where in the World am I?
A multi-model approach to GeoLocation using machine learning.

A project made primarily as part of CS 766 Computer Vision at UW-Madison.

Authors:
- Nick Boddy
- Caleb Raschka
- Morgan Turville-Heitz

[Project Presentation](https://docs.google.com/presentation/d/1akX8-ajRWpmPtD93Bkqu015D39DX9NZOcdgDnppzA8Q)

[GitHub](https://github.com/Cdogsnappy/WITWAI)

---

## Motivations
[GeoGuessr](https://www.geoguessr.com/) is a web-based game wherein the player is dropped into random places in the world and is given visual information via Google Street View. Using the information presented, the player must make a guess as to the coordinates of the image by placing a pin on the globe.

Due to the game's popularity, a competitive scene has arisen in which certain players have become more or less experts at GeoLocation. These players have learned and harnessed certain skills that allow them to excel. These include but are not limited to:
1. Reading and understanding text
2. Identifying language/script of text
3. Recognizing different types/styles of:
    - Architecture
    - Vehicles
    - Biomes
    - Flora
    - Fauna
    - Geography
4. Understanding geographical distributions and correlations of the presences of these identifications

This last point is ultimately key. For a player to make the best prediction with the information available, they must rely on their understanding of how all of these pieces assemble to produce a geographical location that makes sense.

By now it is surely obvious that the task of GeoLocation is learnable, and our approach to this problem is designed to leverage how skilled GeoGuessr players approach the game.

## Prior Work

Before expanding on our approach 

Website:

The final project report must be submitted to Canvas as a website. The website must include:

- Project title and team members
- Link to presentation slides
- Link to github repository.
- Intro presenting and motivating the problem
- Your methodology (method, data, evaluation metrics). If applicable, highlight how it differs from prior work (new experiments, new methods, etc)

# Methodology 

## Overview 

We have implemented a multi-model pipeline in order to classify the country label of each image. This is utilizing an OCR text recognition model, a SotA LLM for language identification, an image segmentation model, and a deep feedforward network. All code is written in python, using either Jupyter notebooks or Python scripts. Implementation of the FFNN is handled in PyTorch. 

## Dataset(s)

The initial dataset that was used during our pipeline testing is a set of 10,000 images and scraped from google street view. These can be found at [Google Street View](https://www.kaggle.com/datasets/paulchambaz/google-street-view/data) - the script *kaggle_data_prep.ipynb* can be used to collect this dataset. Images are 640x640. We found a very low density of text, as well as a low density of segmentable features, in this dataset, so this was only used for testing the initial implementation of our models, and was not used during model training. These images are pruned to only contain European images, which is  ~20% of the dataset.

For the model training, we attempted to collect two separate datasets:
1. Google Maps Street View (gmaps) Data
2. OpenStreetMap API - Mapillary (mapillary) Data

The gmaps dataset is collected by the code in the later half of the *mapillary_google_maps_data_prep.ipynb* script. The data collection is performed as follows:

1) We load a GeoPandas object (*world*) containing the polygon representation of countries. This maps a (longitude/latitude) pair to a specific country ID.
2) We join *world* with a new GeoPandas object containing the Urban Areas polygon database, *urban_areas*. This is the Global Human Settlement Layer R2019A dataset [GHS-FUA R2019A](https://human-settlement.emergency.copernicus.eu/ghs_fua.php). By joining the two objects, we are able to restrict sampling to urban areas, in order to target higher-information images when sampling.
3) We then sample 66% of images from "Urban areas" and 33% of images from "Rural areas," as defined above.
4) Sampling of coordinates is performed as follows:
- We randomly generate a (lat/lon) pair.
- We create a Shapely.geometry point from the pair.
- We join the Shapely point with either the *urban_areas* object or the *world* object, depending on if the sample is intended to be rural or urban.
- We check if the joined Shapely point is found within the selected object, and if so, we then check if the point is in Europe.
- If the object is in Europe, we return the (lat/lon) pair and the country as a label for the datapoint.
5) If the (lat/lon) sampling was succesful, we have to perform an important step - identifying an address that we can query the API for.
- We use a gmaps object with the function *reverse_geocode* to identify an address based on a (lat/lon) pair.
- If there is a nearby address within ~5 kilometers, the closest address is returned. 
- This address then needs to be parsed, in order to query the API for an image in step 6). 
6) If the address sampling was succesful, we query the Google Maps API, and query the API for an image. This functions as follows: 
- We use a standard HTTP query, with the address as an argument in the HTTP request.
- If we receive a 200 response code (i.e. succesful query), and if the image is not empty (image size > 10kb), we store the image. A non-negligible number of images result in empty returns, which are blank with an image filesize ~5kb.
7) Finally, we store the (lat/lon) and country ID in a .csv file, and store the image with the filename corresponding to the dataset's index. 

It is important to note here that we were rate-limited by the gmaps querying - Google thought we were data scraping, and were unable to generate as large of a dataset as we would've liked. However, the gmaps dataset would be the optimal dataset in the future, due to data uniformity and consistency. This will be discussed further in the results section.

The mapillary dataset is collected similarly to the gmaps dataset.

1) We load a GeoPandas object (*world*) containing the polygon representation of countries. This maps a (longitude/latitude) pair to a specific country ID.
2) We join *world* with a new GeoPandas object containing the Urban Areas polygon database, *urban_areas*. This is the Global Human Settlement Layer R2019A dataset [GHS-FUA R2019A](https://human-settlement.emergency.copernicus.eu/ghs_fua.php). By joining the two objects, we are able to restrict sampling to urban areas, in order to target higher-information images when sampling.
3) We then filter out only European urban areas prior to beginning the dataset search. 
4) We then randomly sample urban areas in the resultant *urban_areas_in_europe* object.
5) This urban area object contains bounding boxes (west, south, east, north), i.e. two (lon/lat) pairs identifying the opposing corners of the box.
6) Each urban area contains multiple "tiles". We sweep over these tiles, and query the Mapillary API to identify if any images are present in a given tile.
7) If a given tile contains images, we request the API again for the URL of the images for a given tile. 
8) Then, we query the API for the image at the given URL. 
9) Finally, we store the (lat/lon) and country ID in a .csv file, and store the image with the filename corresponding to the dataset's index.

## OCR Language Guessing

There are two separate methods for the OCR model. The first method is what is used for the kaggle and gmaps dataset, and the second method is used for the mapillary dataset. However, the two methods are identical after a certain point, so I will discuss the data preparation that is needed for the kaggle and gmaps datasets first, before discussing how all three datasets are processed by the OCR model.

#### Kaggle/Gmaps Preparation

The primary issue is that the kaggle/gmaps datasets contain Google© watermarks. Therefore, OCR text recognition will 


## Segmentation

## FFNN



- Discussion of quantitative results
- (Important) Demos of your approach.
