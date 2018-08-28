# Bubble-Chart-using-D3
D3 visualisation of mobile subscription and telephone subscription of 10 country

Visualisation for Mobile Cellular Subscription v/s Fixed Telephone
Subscription from 2000 to 2015 using D3.js

<b>1.Introduction</b>

Visualisation of a dataset is a sensible and highly persuasive way of communication. It helps to
understand the data in hand in a much better way. Even though how complex the data is; visualisation make
the interpretation of data much simpler. The key benefit of visualisation is better understandability,
readability, simplicity.
In this visualisation we are using D3.js to visualise the data. We are comparing Mobile cellular
subscription and Fixed Telephone subscription between ten countries over the years of 2000 to 2015. The
chart which is used to visualise the data over a span of 15 years is the time series motion chart [1].

<b>2. Dataset</b>

The data is collected from the http://data.worldbank.org/ [2] where the country wise data is
downloaded. We will be considering 11 datasets from the world bank website. The countries which are
considered when visualising the data is India, Ireland, United States, Australia, China, Japan, UK, France,
Brazil and South Africa. The dataset will be having a lot of unwanted data which will not be taken into
consideration when doing the visualisation, this data will be filtered out, and only the data which we need
for the visualisation will be retained in the csv file. As all the countries data are in different csv files, the
data will be merged into one file for visualisation. Dataset containing the population of the above said
counties are also downloaded into the system which will be used as a segment of the visualisation activity.
The population will determine the size of the dot which will be visualised in the motion chart. The bigger
the population of a country, bigger will be the size of the dot which is getting displayed in the visualisation.

<b>3 Initialising</b>

In this visualisation we are calling the data from a webserver rather than placing it in a json file
locally. For that we will copy the code which we got as an output from open refine and place it in the
webserver http://myjson.com/2njqe [3]. Upon clicking the save button, the website will give you a link
which can be used as an input to the visualisation code which is written for creating the motion chart.
The tool which is used over here to create the visualisation is D3.js which is one of the most
powerful JavaScript library. The chart which is taken into consideration to create this beautiful visualisation
is the time series motion chart. The characteristic of the motion chart is that it will show a moving
visualization above a particular time period and in this case it will be from 2000 to 2015 which shows the
animation of Mobile cellular subscription vs Fixed telephonic subscription over the 15 years.

