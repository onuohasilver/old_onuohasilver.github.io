---
title: "Nairobi Traffic Prediction Challenge."
date: 2018-12-20
categories: competition
---

![Nairobi Taxis](http://3lq1ku40fh612q5lii5rfl0n.wpengine.netdna-cdn.com/wp-content/uploads/2016/02/phoenixtraffic.jpg)

## Competition Description


Nairobi is one of the most heavily congested cities in Africa. Each day thousands of Kenyans make the trip into Nairobi from towns such as Kisii, Keroka, and beyond for work, business, or to visit friends and family. The journey can be long, and the final approach into the city can impact the length of the trip significantly depending on traffic. How do traffic patterns influence people’s decisions to come into the city by bus and which bus to take? Does knowing the traffic patterns in Nairobi help anticipate the demand for particular routes at particular times?
<!--more-->


The aim of the competition is to create a predictive model using traffic data provided from Uber Movement and historic bus ticket sales data from Mobiticket to predict the number of tickets that will be sold for buses into Nairobi from cities in "up country" Kenya.

The data used to train the model will be historic hourly traffic patterns in Nairobi and historic ticket purchasing data for 14 bus routes into Nairobi from October 2017 to April 2018, and includes the place or origin, the scheduled time of departure, the channel used for the purchase, the type of vehicle, the capacity of the vehicle, and the assigned seat number. Zindi competitors will be allowed to create their own customized traffic datasets using the Uber Movement platform.

This resulting model can be used by Mobiticket and bus operators to anticipate customer demand for certain rides, to manage resources and vehicles more efficiently, to offer promotions and sell other services more effectively, such as micro-insurance, or even improve customer service by being able to send alerts and other useful information to customers.





The solutions to this challenge are the first step towards solving Nairobi's traffic problems. We look forward to taking this journey with you!

This competition is sponsored by Uber , Mobiticket, and insight2impact.


```python
#importing the necessary codes
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_absolute_error
import seaborn as sns
%matplotlib inline
```

### Importing the competition dataset


```python
train = pd.read_csv("train_revised.csv", parse_dates=["travel_date"],)
test = pd.read_csv("test_questions.csv", parse_dates=["travel_date"])
sub = pd.read_csv("sample_submission.csv")
train_idx = len(train) #getting the length of the to enable separation later

```

view a subset of the train set to get an idea of the kind of data we are working with.
```python
#view subset of train
train.head()
```

We are told to predict the number of tickets that would be sold at each point
however there is no column to indicate the our target variable.
So we will have to take an average of the number of tickets that was sold per
**ride_id**, since the **ride_id's** are unique identifiers for different taxi's on
the Nairobi road that we are analysing.


```python
ridecount=pd.DataFrame(train['ride_id'].value_counts()).reset_index()
ridecount.columns=['ride_id','number_of_tickets']
```


```python
#train[train['ride_id']==5778]
```




Examining different ride id's buttresses the idea that all **ride_id's** were vehicles that left at the same time and on the same day.
the above line of code can be uncommented to see an example!

Therefore we can now go ahead and drop the duplicate values


```python
train.drop_duplicates(subset='ride_id',inplace=True)
train.shape
```




    (6249, 10)



we can now go ahead and merge the two datasets (the train and the ridecounts)


```python
train=train.merge(ridecount,how='left',left_on='ride_id',right_on='ride_id')
```


```python
train.head()
```





    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]),
     <a list of 17 Text xticklabel objects>)




![Nairobi Taxis]({{site.baseurl}}/assets/img/output_20_1.png)



```python
sns.set(rc={'figure.figsize':(6,6)})
sns.barplot(x='car_type',y='number_of_tickets',data=train)
plt.xticks(rotation=70)
```




    (array([0, 1]), <a list of 2 Text xticklabel objects>)




![Nairobi Taxis]({{site.baseurl}}/assets/img/output_21_1.png)



```python
sns.set(rc={'figure.figsize':(6,6)})
sns.countplot(x='car_type',data=train)
plt.xticks(rotation=70)
```




    (array([0, 1]), <a list of 2 Text xticklabel objects>)




![Nairobi Taxis]({{site.baseurl}}/assets/img/output_22_1.png)



```python
sns.set(rc={'figure.figsize':(6,6)})
sns.countplot(x='number_of_tickets',data=train)
plt.xticks(rotation=0)
plt.title('Distribution of number of tickets purchased')
```




    Text(0.5,1,'Distribution of number of tickets purchased')




![Nairobi Taxis]({{site.baseurl}}/assets/img/output_23_1.png)



```python
train.head()
```



```python
sns.set(rc={'figure.figsize':(6,6)})
sns.countplot(x='month',data=train)
plt.xticks(rotation=0)
plt.title('Number of taxi rides per month')

```




    Text(0.5,1,'Number of taxi rides per month')




![Nairobi Taxis]({{site.baseurl}}/assets/img/output_25_1.png)



```python
sns.set(rc={'figure.figsize':(6,6)})
sns.countplot(x='day',data=train)
plt.xticks(rotation=0)
plt.title('Number of taxi rides distributed across specific days of the months')

```




    Text(0.5,1,'Number of taxi rides distributed across specific days of the months')




![Nairobi Taxis]({{site.baseurl}}/assets/img/output_26_1.png)



```python
train.head()
```



```python
train.travel_time=pd.to_datetime(train.travel_time)
train['hour']=train.travel_time.dt.hour
train.loc[train['hour']<7,'hours']='Very Early'
train.loc[(train['hour']>=7)&(train['hour']<8),'hours']='Early'
train.loc[(train['hour']>=9)&(train['hour']<11),'hours']='Late Morning'
train.loc[(train['hour']>=12)&(train['hour']<=24),'hours']='Rest of the day'
```


```python
sns.set(rc={'figure.figsize':(6,6)})
sns.violinplot(x='hours',y='number_of_tickets',data=train)
plt.xticks(rotation=0)
plt.title('Number of taxi rides distributed across specific days of the months')

```




    Text(0.5,1,'Number of taxi rides distributed across specific days of the months')




![Nairobi Taxis]({{site.baseurl}}/assets/img/output_29_1.png)



```python
sns.set(rc={'figure.figsize':(8,10)})
sns.countplot(x='hours',data=train)
plt.xticks(rotation=0)
plt.title('Number of taxi rides distributed across specific days of the months')

```




    Text(0.5,1,'Number of taxi rides distributed across specific days of the months')




![Nairobi Taxis]({{site.baseurl}}/assets/img/output_30_1.png)



```python
sns.set(rc={'figure.figsize':(6,6)})
sns.lmplot(x='day',y='number_of_tickets',data=train)
plt.xticks(rotation=0)
plt.title('Number of taxi rides distributed across specific days of the months')

```




    Text(0.5,1,'Number of taxi rides distributed across specific days of the months')




![Nairobi Taxis]({{site.baseurl}}/assets/img/output_31_1.png)



```python
def date_split(data,date):
    a=pd.to_datetime(data[date])
    data['weekday']=a.dt.dayofweek
    data['year']=a.dt.dayofyear
    data['quarter']=a.dt.quarter
    data['is_weekend']=0
    data.loc[(data['weekday']>=4),'is_weekend']=1
```


```python
date_split(train,'travel_date')
```


```python
train.head()
```







```python
sns.set(rc={'figure.figsize':(6,6)})
sns.countplot(x='is_weekend',data=train)
plt.xticks(rotation=0)
plt.title('Countplot for the taxi rides on non-weekends and weekends')

```




    Text(0.5,1,'Number of taxi rides distributed across specific days of the months')




![Nairobi Taxis]({{site.baseurl}}/assets/img/output_35_1.png)



```python
sns.set(rc={'figure.figsize':(6,6)})
sns.swarmplot(x='is_weekend',y='number_of_tickets',data=train)
plt.xticks(rotation=0)
plt.title('Number of taxi rides distributed across weekends and none weekends')
```




    Text(0.5,1,'Number of taxi rides distributed across weekends and none weekends')




![Nairobi Taxis]({{site.baseurl}}/assets/img/output_36_1.png)



```python
train.head()
```







```python
train.drop('hours',inplace=True,axis=1)
```


```python
train.drop(['travel_date','ride_id','travel_time'],inplace=True,axis=1)
```


```python
train.columns
```




    Index(['travel_from', 'car_type', 'max_capacity', 'number_of_tickets', 'month',
           'day', 'hour', 'weekday', 'year', 'quarter', 'is_weekend'],
          dtype='object')




```python
test.columns
```




    Index(['ride_id', 'travel_date', 'travel_time', 'travel_from', 'car_type',
           'max_capacity', 'hour', 'weekday', 'year', 'quarter', 'is_weekend',
           'month', 'day'],
          dtype='object')




```python
test.drop(['travel_to'],axis=1,inplace=True)
test.drop(['travel_date','travel_time'],axis=1,inplace=True)
test.travel_time=pd.to_datetime(test.travel_time)
test['hour']=test.travel_time.dt.hour
date_split(test,'travel_date')
test['month']=test.travel_date.dt.month
test['day']=test.travel_date.dt.day
```


```python
#catcoding the string variables
train.travel_from=train.travel_from.astype('category')
train.car_type=train.car_type.astype('category')
#do the same for the test set
test.travel_from=test.travel_from.astype('category')
test.car_type=test.car_type.astype('category')

```


```python
train.travel_from=train.travel_from.cat.codes
train.car_type=train.car_type.cat.codes
test.travel_from=test.travel_from.cat.codes
test.car_type=test.car_type.cat.codes
```


```python
train.head()
```



```python
X=train.drop('number_of_tickets',axis=1)
y=train.number_of_tickets
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```


```python
model = RandomForestRegressor(bootstrap=True, criterion='mae', max_depth=23,
          max_features=0.5, max_leaf_nodes=45, min_impurity_decrease=0.0,
          min_impurity_split=None, min_samples_leaf=10,
          min_samples_split=4, min_weight_fraction_leaf=0.0,
          n_estimators=67, n_jobs=1, oob_score=True, random_state=42,
          verbose=0, warm_start=False)
```


```python
model.fit(X_train,y_train)
mean_absolute_error(model.predict(X_test),y_test)
```




    3.712574364263137


