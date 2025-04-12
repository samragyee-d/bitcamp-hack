# app.py
import bcrypt
from flask import Flask, render_template, request, redirect, url_for, Response, flash
import cv2
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash

#SQL setup
import mysql.connector
from dotenv import load_dotenv
import os
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

# Import Environment Variables
load_dotenv()

from ml_utils import generate_frames
# Load enviro

MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
MYSQL_HOST = os.getenv('MYSQL_HOST')
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE')
FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY')

app = Flask(__name__)
import cv2
import torch

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_input = request.form['username']
        password = request.form['password']

        # Connect to MySQL
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )
        cursor = conn.cursor()

        # Fetch the stored hashed password from the database
        cursor.execute("SELECT password FROM users WHERE username=%s OR email=%s", (user_input, user_input))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        

        # Check if user exists and if the password matches
        if result and bcrypt.checkpw(password.encode('utf-8'), result[0].encode('utf-8')):
            return video()
        else:
            flash("Invalid credentials. Please try again.")
            return redirect(url_for('login'))

    return render_template('Login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')


        if not username or not email or not password:
            return render_template('registration.html', message='Please fill out all fields.')


        try:
            # Hash the password using bcrypt
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())


            # Connect to MySQL
            connection = mysql.connector.connect(
                host=MYSQL_HOST,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE
            )
           
            cursor = connection.cursor()
            


            # Insert the user into the database
            insert_query = "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)"
            cursor.execute(insert_query, (username, email, hashed_password))
            connection.commit()


            cursor.close()
            connection.close()
            

            return render_template('video.html', message='Registration successful!')

        except mysql.connector.Error as err:
            print(f"Error during insert: {err}")
            return render_template('registration.html', message=f"Error: {err}")
        
    return render_template('registration.html')



@app.route('/video')
def video():
    return render_template('video.html')  # HTML with <img src="/video_feed">

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# BASIC FLASK
@app.route('/pagename')
def pagename():
    return render_template('pagename.html')

'''
@app.route('/submit', methods=['POST'])
def submit():
    # Get form data
    name = request.form['name']
    email = request.form['email']
    
    # You can process or store the data here
    return render_template('result.html', name=name, email=email)
'''

if __name__ == '__main__':
    app.run(debug=True)
