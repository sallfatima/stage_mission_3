#!flask/bin/python
from flask import Flask, render_template, request, make_response
import os
import sys
import json
import random

app = Flask(__name__)

def makedata():
    with open('vehicle-data.json') as data_file:
        data = json.load(data_file)
    return data

def makedataBarChart():
    with open('vehicle-data.json') as data_file:
        data = json.load(data_file)
    return data


@app.route('/')
def main():
    return render_template('index.html', data=json.dumps(makedata()))

@app.route('/barchart')
def barchart():
    return render_template('workingIndex.html', data=json.dumps(makedata()))

@app.route('/bubblechart')
def bubbleChart():
    return render_template('index.html', data=json.dumps(makedata()))




if __name__ == '__main__':
    # main()
    app.run(debug=True)
