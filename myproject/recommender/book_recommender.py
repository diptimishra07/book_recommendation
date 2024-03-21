#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:

# Read the files
books = pd.read_csv('/Users/dipti/Desktop/Datasets/books.csv')
users = pd.read_csv('/Users/dipti/Desktop/Datasets/users.csv')
ratings = pd.read_csv('/Users/dipti/Desktop/Datasets/ratings.csv')


# In[3]:

## Get the size
print(books.shape)
print(users.shape)
print(ratings.shape)


# # Popularity Based Recommender System

# In[10]:


book_ratings = ratings.merge(books,on='ISBN')


# In[11]:


book_ratings.head()


# In[14]:


num_ratings = book_ratings.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_ratings.rename(columns={'Book-Rating':'num_ratings'},inplace=True)
num_ratings.head(5)


# In[15]:


avg_rating=book_ratings.groupby('Book-Title').mean(numeric_only=True)['Book-Rating'].reset_index()
avg_rating.rename(columns={'Book-Rating':'avg_rating'},inplace=True)
avg_rating.head(5)


# In[16]:


popular_books = num_ratings.merge(avg_rating,on='Book-Title')
popular_books


# In[17]:


def top_n_books(data, n, num_rating):
    recommendations= data[data['num_ratings']>=num_rating]
    recommendations=recommendations.sort_values(by='avg_rating',ascending=False)
    return recommendations.iloc[:n]


# In[18]:


popular_books=top_n_books(popular_books, 50, 250)



# In[20]:


popular_books = popular_books.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_ratings','avg_rating']]


# In[22]:


import pickle
pickle.dump(popular_books,open('popular.pkl','wb'))


# # Collaborative Filtering Based Recommender System

# In[24]:


x = book_ratings.groupby('User-ID').count()['Book-Rating'] > 200
imp_users = x[x].index
imp_users


# In[25]:


filtered_rating = book_ratings[book_ratings['User-ID'].isin(imp_users)]


# In[26]:


y = filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books = y[y].index


# In[27]:


final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]


# In[28]:


pivot_table = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')


# In[29]:


pivot_table.fillna(0,inplace=True)
pivot_table


# In[30]:


from sklearn.metrics.pairwise import cosine_similarity


# In[31]:

# Get similarity scores
similarity_scores = cosine_similarity(pivot_table)


# In[32]:


similarity_scores.shape


# In[35]:


def recommend(book_name):
    # index fetch
    index = np.where(pivot_table.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:5]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pivot_table.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(item)
    
    return data


# In[36]:


recommend('The Notebook')


# In[38]:

# Dump to pickle
import pickle
pickle.dump(pivot_table,open('pt.pkl','wb'))
pickle.dump(books,open('books.pkl','wb'))
pickle.dump(similarity_scores,open('similarity_scores.pkl','wb'))





