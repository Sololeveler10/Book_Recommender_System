import pickle
import streamlit as st
import numpy as np


st.header('Books Recommended System using Machine Learning')

# Load the pre-trained model and data
model = pickle.load(open('artifacts/model.pkl', 'rb'))
books_name = pickle.load(open('artifacts/book_names.pkl', 'rb'))
final_ratings = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))

def recommend_book(book_name):
    try:
        book_id = np.where(book_pivot.index == book_name)[0][0]
        distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
        
        book_list = []
        poster_url = fetch_poster(suggestion)

        # Loop through suggestions to get the book names
        for i in range(len(suggestion[0])):
            book_name = book_pivot.index[suggestion[0][i]]
            book_list.append(book_name)

        return book_list, poster_url
    except Exception as e:
        st.error(f"Error in recommend_book: {e}")
        return [], []

def fetch_poster(suggestion):
    books_name = []
    ids_index = []
    poster_url = []

    # Extract the book names from the suggestions
    for book_id in suggestion[0]:  # suggestion[0] gives the list of suggested book indices
        books_name.append(book_pivot.index[book_id])

    # Loop through the book names to fetch the poster URLs
    for name in books_name:
        try:
            # Make sure the 'Title' column exists in final_ratings
            ids = np.where(final_ratings['title'] == name)[0][0]
            ids_index.append(ids)
            url = final_ratings.iloc[ids]['image_url']
            poster_url.append(url)
        except IndexError:
            # Handle the case if a book name is not found in final_ratings
            poster_url.append('')  # Empty URL if not found

    return poster_url

selected_books = st.selectbox(
    "Type or Select a Book",
    books_name
)

if st.button('Show Recommendation'):
    recommendation_books, poster_url = recommend_book(selected_books)

    if len(recommendation_books) >= 6:
        # Display the recommended books and their poster images in columns
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.text(recommendation_books[1])
            st.image(poster_url[1])

        with col2:
            st.text(recommendation_books[2])
            st.image(poster_url[2])

        with col3:
            st.text(recommendation_books[3])
            st.image(poster_url[3])

        with col4:
            st.text(recommendation_books[4])
            st.image(poster_url[4])

        with col5:
            st.text(recommendation_books[5])
            st.image(poster_url[5])
    else:
        st.error("Not enough recommendations to display.")
