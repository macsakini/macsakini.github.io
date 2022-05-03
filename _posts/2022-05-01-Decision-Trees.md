```python
#Importing dependancies
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
```


```python
#Import data stored 
train_data = pd.read_csv("train-data.csv")
test_data = pd.read_csv("test-data.csv")

train_data.head()
test_data.head()
```





  <div id="df-1be64de1-1ad6-4638-bb47-a408ab6b5c02">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Name</th>
      <th>Location</th>
      <th>Year</th>
      <th>Kilometers_Driven</th>
      <th>Fuel_Type</th>
      <th>Transmission</th>
      <th>Owner_Type</th>
      <th>Mileage</th>
      <th>Engine</th>
      <th>Power</th>
      <th>Seats</th>
      <th>New_Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Maruti Alto K10 LXI CNG</td>
      <td>Delhi</td>
      <td>2014</td>
      <td>40929</td>
      <td>CNG</td>
      <td>Manual</td>
      <td>First</td>
      <td>32.26 km/kg</td>
      <td>998 CC</td>
      <td>58.2 bhp</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Maruti Alto 800 2016-2019 LXI</td>
      <td>Coimbatore</td>
      <td>2013</td>
      <td>54493</td>
      <td>Petrol</td>
      <td>Manual</td>
      <td>Second</td>
      <td>24.7 kmpl</td>
      <td>796 CC</td>
      <td>47.3 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Toyota Innova Crysta Touring Sport 2.4 MT</td>
      <td>Mumbai</td>
      <td>2017</td>
      <td>34000</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>First</td>
      <td>13.68 kmpl</td>
      <td>2393 CC</td>
      <td>147.8 bhp</td>
      <td>7.0</td>
      <td>25.27 Lakh</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Toyota Etios Liva GD</td>
      <td>Hyderabad</td>
      <td>2012</td>
      <td>139000</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>First</td>
      <td>23.59 kmpl</td>
      <td>1364 CC</td>
      <td>null bhp</td>
      <td>5.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Hyundai i20 Magna</td>
      <td>Mumbai</td>
      <td>2014</td>
      <td>29000</td>
      <td>Petrol</td>
      <td>Manual</td>
      <td>First</td>
      <td>18.5 kmpl</td>
      <td>1197 CC</td>
      <td>82.85 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1be64de1-1ad6-4638-bb47-a408ab6b5c02')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-1be64de1-1ad6-4638-bb47-a408ab6b5c02 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1be64de1-1ad6-4638-bb47-a408ab6b5c02');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
#preprocess the data and clean it
class Preprocess():
    def __init__(self, dataset):
        self.dataset = dataset
        return None

    #This function drops unnecessary columns
    def dropcolumns(self):
        to_drop = ["Unnamed: 0", "Name", "Location", "Kilometers_Driven", "Fuel_Type", "Transmission", "Owner_Type", "Power", "Seats", "New_Price"]
        return self.dataset.drop(
            to_drop, inplace=True, axis=1
        )

    #This function drops all rows with null values from the dataset
    def dropna(self):
        col_drop = self.dropcolumns()
        dropped = self.dataset.dropna()
        return dropped

    #This function cleans the units from the string columns
    #It also concerts the clean columns which are in string format to numeric
    def removeString(self):
        self.dataset['Engine'] = self.dataset['Engine'].str.replace(r'\D', '')
        self.dataset['Engine'] = pd.to_numeric(self.dataset['Engine'])
        self.dataset['Mileage'] = self.dataset['Mileage'].str.replace(r'\D', '')
        self.dataset['Mileage'] = pd.to_numeric(self.dataset['Mileage'])
        return self.dataset

    #This function calls the rest of the class
    def clean(self):
      self.removeString()
      return self.dropna()

```


```python
#clean the training and test data
clean_train_data = Preprocess(train_data).clean()

clean_test_data = Preprocess(test_data).clean()

#show the clean data
clean_train_data.head()
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:23: FutureWarning: The default value of regex will change from True to False in a future version.
    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:25: FutureWarning: The default value of regex will change from True to False in a future version.






  <div id="df-f0eec93a-8b55-4c52-86f2-6dad96004394">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Mileage</th>
      <th>Engine</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010</td>
      <td>266.0</td>
      <td>998.0</td>
      <td>1.75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>1967.0</td>
      <td>1582.0</td>
      <td>12.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011</td>
      <td>182.0</td>
      <td>1199.0</td>
      <td>4.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012</td>
      <td>2077.0</td>
      <td>1248.0</td>
      <td>6.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013</td>
      <td>152.0</td>
      <td>1968.0</td>
      <td>17.74</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-f0eec93a-8b55-4c52-86f2-6dad96004394')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-f0eec93a-8b55-4c52-86f2-6dad96004394 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-f0eec93a-8b55-4c52-86f2-6dad96004394');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
#split the data to features(X) and targets(labels)(Y)
class splitXY():
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label
        return None
    #this function creates the features and labels
    def splitlabel(self):
        X = self.dataset.drop(self.label, axis = 1).values
        y = self.dataset[self.label[0]].values

        return X, y 

    #splot the data to train and test data
    def splitdata(self):
        X, y = self.splitlabel()

        X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=25)

        # print(f"No. of training examples: {training_data.shape[0]}")
        # print(f"No. of testing examples: {testing_data.shape[0]}")
        return X_train, X_test, y_train, y_test
```

#First Approach: Using the Target Variable Price as a continous variable and thus regression by decision trees


```python
X_train, X_test, y_train, y_test = splitXY(clean_train_data, ["Price"]).splitdata()
#Show the training data
X_train, y_train
```




    (array([[2013.,  284., 1248.],
            [2015.,  182., 1248.],
            [2015., 1757., 1193.],
            ...,
            [2005.,  110., 2987.],
            [2018., 1602., 1373.],
            [2016., 2014., 1498.]]),
     array([ 4.95,  4.3 ,  4.52, ..., 10.  ,  8.25,  6.3 ]))




```python
#Instanciate the decision tree regressors
fit_1 = DecisionTreeRegressor(max_depth=2)
fit_2 = DecisionTreeRegressor(max_depth=5)


#Fit the data to the instanciated model
fit_1.fit(X_train, y_train)
fit_2.fit(X_train, y_train)
```




    DecisionTreeRegressor(max_depth=5)




```python
fit_2.score(X_train, y_train)
```




    0.7678476226395207




```python
fit_2.get_n_leaves()
```




    32




```python
cross_val_score(fit_2, X_train, y_train, cv=10)
```




    array([0.81216411, 0.76445921, 0.70394034, 0.66493376, 0.64491868,
           0.71417788, 0.61902476, 0.7185033 , 0.72539609, 0.68439817])




```python
from math import sqrt
#Make predictions of the model using the test dataset
#X_test = clean_test_data
y_1 = fit_1.predict(X_test)
y_2 = fit_2.predict(X_test)


#Calculate sum of squared errors
err = y_test - y_2
print((sum(err**2)))
```

    30397.07422407224


#Second Approach: Make classes/bins using Target Variable Price and thus classifcation using decision trees


```python
#create three classes of cheap, middle and expensive
clean_train_data['Label'] = pd.cut(x = clean_train_data['Price'], bins = [0, 4, 7, 15, 40, 200], labels=['Cheap', 'Low-Mid', 'Mid-High','Expensive', "Super-Expensive"])
clean_train_data['Label'].value_counts()
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      





    Cheap              1951
    Low-Mid            1793
    Mid-High           1258
    Expensive           809
    Super-Expensive     170
    Name: Label, dtype: int64




```python
X_train, X_test, y_train, y_test = splitXY(clean_train_data, ["Label","Price"]).splitdata()
#Show the training data
X_train, y_train
```




    (array([[2013.,  284., 1248.],
            [2015.,  182., 1248.],
            [2015., 1757., 1193.],
            ...,
            [2005.,  110., 2987.],
            [2018., 1602., 1373.],
            [2016., 2014., 1498.]]),
     ['Low-Mid', 'Low-Mid', 'Low-Mid', 'Low-Mid', 'Low-Mid', ..., 'Cheap', 'Low-Mid', 'Mid-High', 'Mid-High', 'Low-Mid']
     Length: 4784
     Categories (5, object): ['Cheap' < 'Low-Mid' < 'Mid-High' < 'Expensive' < 'Super-Expensive'])




```python
clf = DecisionTreeClassifier(random_state = 34)# max_depth = 5)
clf.fit(X_train, y_train)
```




    DecisionTreeClassifier(random_state=34)




```python
cross_val_score(clf, X_train, y_train , cv=10)
```




    array([0.77244259, 0.76617954, 0.76617954, 0.76200418, 0.80125523,
           0.75313808, 0.76569038, 0.78870293, 0.77824268, 0.76987448])




```python
clf.score(X_test, y_test)
```




    0.772765246449457




```python
clf.predict(clean_test_data)
```

    /usr/local/lib/python3.7/dist-packages/sklearn/base.py:444: UserWarning: X has feature names, but DecisionTreeClassifier was fitted without feature names
      f"X has feature names, but {self.__class__.__name__} was fitted without"





    array(['Cheap', 'Cheap', 'Expensive', ..., 'Cheap', 'Cheap', 'Expensive'],
          dtype=object)


