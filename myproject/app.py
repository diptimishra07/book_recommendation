from flask import Flask
from flask import render_template
from flask import request
import pickle
import numpy as np

popular_book= pickle.load(open('/Users/dipti/popular.pkl','rb'))
pivot_table= pickle.load(open('/Users/dipti/pt.pkl','rb'))
books= pickle.load(open('/Users/dipti/books.pkl','rb'))
sim_scores= pickle.load(open('/Users/dipti/similarity_scores.pkl','rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html',
                            book_name= list(popular_book['Book-Title'].values),
                            author_name= list(popular_book['Book-Author'].values),
                            img= list(popular_book['Image-URL-M'].values),
                            votes= list(popular_book['num_ratings'].values),
                            avg_rating= list(popular_book['avg_rating'].values)
                           )
@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_input= request.form.get('user_input')
    index = np.where(pivot_table.index== user_input)[0][0]
    similar_items = sorted(list(enumerate(sim_scores[index])),key=lambda x:x[1],reverse=True)[1:5]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pivot_table.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(item)
    print(data)
    return render_template('recommend.html',data=data)

if __name__ =='__main__':
    app.run(debug=True)