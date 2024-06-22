from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
import numpy as np
from tensorflow import keras
import keras as k 
from sklearn.model_selection import train_test_split
from keras import regularizers 
# from keras.models import Sequential,model_from_json 
# from keras.layers import Dense,Dropout,Activation 
from keras import models,layers
from keras import optimizers 
from numpy.random import seed

def homePage(request):
    # data={
    #     'title':"Home New",
    #     'bdata':"Welcome to DJANGO homepage",
    #     'clist':['python','C','C++'],
    #     'student_details':[
    #         {'name':'Asfar','phone':1234567},
    #         {'name':'Alli','phone':9876543}
    #     ],
    #     'numbers':[]
    # }

    return render(request,'index.html')

def prediction(request):
    return render(request,"live.html")
# def aboutUs(request):
#     return HttpResponse("This is the about us page")

# def Course(request):
#     return HttpResponse("This is a course")

# def courseDetails(request,course_id):
#     return HttpResponse(course_id)

def results(request):
    df_users = pd.read_csv('static/users.csv')
    df_fusers = pd.read_csv('static/fusers.csv')

    isNotFake = np.zeros(df_users.shape[0])
    df_users['isFake'] = isNotFake
    df_fusers['isFake'] = np.ones(df_fusers.shape[0])
    df_allUsers = pd.concat([df_fusers, df_users], ignore_index=True)
    df_allUsers.columns = df_allUsers.columns.str.strip()

    Y = df_allUsers.isFake
    X = df_allUsers.drop(['isFake'], axis=1)

    Y.reset_index(drop=True, inplace=True)
    lang_list = list(enumerate(np.unique(X['lang'])))
    lang_dict = {name: i for i, name in lang_list}
    X.loc[:, "lang_num"] = X['lang'].map(lambda x: lang_dict[x]).astype(int)
    X.drop(['name'], axis=1, inplace=True)

    X = X[['statuses_count', 'followers_count', 'friends_count', 'favourites_count',
           'lang_num', 'listed_count', 'geo_enabled', 'profile_use_background_image']]
    X = X.replace(np.nan, 0)

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=0.8, test_size=0.2)
    train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, train_size=0.8, test_size=0.2)

    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_dim=8))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_X, train_Y, epochs=10, batch_size=32, validation_data=(val_X, val_Y))

    n1 = int(request.GET.get('n1', 0))
    n2 = int(request.GET.get('n2', 0))
    n3 = int(request.GET.get('n3', 0))
    n4 = int(request.GET.get('n4', 0))
    n5 = int(request.GET.get('n5', 0))

    inputs = np.array([[n1, n2, n3, n4, n5, 0, 0, 0]])  # Adjust this line to include correct number of features
    pred = model.predict(inputs)

    result1 = "Fake" if pred[0][0] >= 0.5 else "Not Fake"
    return render(request, template_name="live.html", context={"result2": result1})
