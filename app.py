# app.py

from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
@app.route('/')
def index():
    return render_template()

# BASIC FLASK
@app.route('/pagename')
def pagename():
    return render_template('corresponding.html')
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
