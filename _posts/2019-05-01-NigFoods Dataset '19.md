---
title: "NigFoods Dataset '19"
date: 2019-05-01
categories: dataset
---

![Nairobi Taxis](http://dooneyskitchen.com/wp-content/uploads/2014/11/Ofada-rice-2-1024x6821.jpg)

## NigFoods Dataset '19


Hi guys!

So I am following the fastai 2019 deep learning course. After the first lesson, Jeremy Howard the course instructor encouraged the students to gather a personal dataset and try to perform an image classification and see just how far we could drive up our accuracy level. I decided to do a classification of Nigerian Foods hence the first step was figuring out how to curate the dataset which I would use, after going through the fastai learning forums (*which I found really inspiring*), I found out that I could use a Library termed ***google_image_download*** which was easily installable by running a simple pip install command in the anaconda jupyter environment.
```python
!pip install google_image_download
```

The library required also that I download a chrome driver if I intended to download a large volume of images. 

Great library if you ask me!
Parsing in the keywords which you want to download- the script runs a google images search, grabs the image urls and starts downloading it to your local drive , saving each batch of pictures with a title of their search keyword.

The *NigFoods Dataset* contains 13 different types of food and 3 different types of snack, which are saved in folders according to their names.
```
1.	Spaghetti
2.	Noodles
3.	Rice and Beans
4.	Bread and Tea
5.	Rice and Stew
6.	Akara
7.	Egusi Soup
8.	Oha Soup
9.	Afang Soup
10.     Abacha 
11.     Beans
12.     Moi Moi
13.     Macaroni
14.     Meat Pie
15.     Garri and Groundnut
16.     Jollof Rice
17.     Okpa        



```
![Nairobi Taxis]({{site.baseurl}}/assets/img/Capture.png)

A Sample code for scraping one of the food datasets would be:
```python
from google_images_download import google_images_download   #importing the library

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"Oha Soup","limit":10000,"print_urls":True,'chromedriver':'C:/Users/HP/Downloads/chromedriver_win32/chromedriver.exe'}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images
```
looking forward to building an image classifier trained on these collected images.