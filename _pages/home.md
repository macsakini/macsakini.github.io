---
layout: single
author_profile: true
classes: wide
---

<style>
.card {
  background-color: black;
  color: white;
  padding: 1rem;
}

.cards {
  max-width: 1200px;
  margin: 50px auto;
  display: grid;
  grid-gap: 1rem;
}

/* Screen larger than 600px? 2 column */
@media (min-width: 600px) {
  .cards { grid-template-columns: repeat(2, 1fr); }
}

/* Screen larger than 900px? 3 columns */
@media (min-width: 900px) {
  .cards { grid-template-columns: repeat(3, 1fr); }
}
</style>

<h1 align="center">City Skylines</h1>
<br>

<p align = "center" > <img src = "/assets/images/cityskylines/Slide1.jpeg"/></p>

A project of this stature requires a lot of data mining. We need to gather from very many different sources to capture all the dynamics of a city. The city under demonstration is Nairobi but more shall be done on other cities such as London and Tokyo.

<p align = "center" > <img src = "/assets/images/cityskylines/Slide2.jpeg"/></p>

<h2 align = "center">Demographics</h2>

## Population 
-	Divide the sub counties into sublocations.
-	We have population data of sublocations in 2009.
-	We have population data of sub-counties in 2019.
-	Find the percentage increase between sub-counties and use this to expand on sublocations for 2019.

<p align = "center" > <img src = "/assets/images/cityskylines/Slide11.jpeg"/></p>

## Population Density
-	Having population data, for both 2009 and 2019 and the area in square kilometers. We can calculate the population density.

## Area of Point of interest.
-	We have the area of each sublocation and subcounty ready. Spatial data rarely changes.

<p align = "center" > <img src = "/assets/images/cityskylines/Slide3.jpeg"/></p>

<h2 align = "center">Spatial Analysis</h2>

## Points of Interest
-	Clip the points using the clip-layer function.

<p align = "center" > <img src = "/assets/images/cityskylines/Slide8.jpeg"/></p>

-	Points include churches, schools, malls, barber shops, restaurants, anything listed in the OSM data as point.

<p align = "center" > <img src = "/assets/images/cityskylines/Slide9.jpeg"/></p>

-	Find their total counts per sublocation i.e., Zimmerman – 100 points, Karen- 300 points.
-	Find their counts per type. i.e., Churches – 10, Schools -20.

<p align = "center" > <img src = "/assets/images/cityskylines/Slide7.jpeg"/></p>

## Area covered by buildings.
-	Total area size in a sublocation covered by buildings

## Road cover. 
<p align = "center" > <img src = "/assets/images/cityskylines/Slide4.jpeg"/></p>

<p align = "center" > <img src = "/assets/images/cityskylines/Slide5.jpeg"/></p>

-	Total area of sublocation covered by roads. 
## Land Cover
-	Can be calculated if available.
## Slope of the area.
-	What is the slope coefficient of the area?


<h2 align = "center">Economics</h2>

## Land Prices.
-	Price of land per acre in region.
-	Scraped data from Jiji, PigiaMe and The Star.

<p align = "center" > <img src = "/assets/images/cityskylines/Slide10.jpeg"/></p>

## Rental prices.
-	Average rental price for a two-bedroom house in the region.
## Type of area.
-	Is the area residential, suburb, mixed, occupational?

<p align = "center" > <img src = "/assets/images/cityskylines/Slide6.jpeg"/></p>

<h2 align = "center">Accessibility (Only for available OSM way ID)</h2>

## Travel Times.
-	Traffic experienced in the area.
## Roads
-	Road size. Is it a major highway? How many lanes is the major road in said sublocation? What is OSM ID.
-	Can uber data help better model this data. CBD is the point of reference. 
-	Travel in the morning at 6.00, 9.00, 12.00.
-	Travel in the evening at 3.00, 6.00, 9.00

<h2 align = "center">Imagery</h2>

## Image processing of area.
-	30 images from each area. Picked randomly. This can be done from Instagram, twitter and fakebook. TBD

<h2 align = "center">Social Thoughts</h2>

## LDA and PCA analysis of tweets.
-	Most common words related to a specific area.
-	If Twitter API allows, this can be done quite comfortably.


<h1 align="center">Coming Soon ...</h1>
<div class="cards">
  <div class="card">
    <h5>BERT RL</h5>
    <p>This is a natural language processing model built to make analysis.</p>
  </div>
  <div class="card">
    <h5>City Skylines</h5>
    <p>City skylines is a GIS project built with deep learning capabilities to analyse and make viable predictions on city trends.</p>
  </div>
  <div class="card">
    <h5>Option Pricing</h5>
  </div>
  <div class="card">
    <h5>Fractals Crack</h5>
  </div>
  <div class="card">
    <h5>Blue Locator</h5>
  </div>
  <div class="card">
    <h5>Sixers</h5>
    <p>This is a natural language processing model built to make analysis.</p>
  </div>
</div>