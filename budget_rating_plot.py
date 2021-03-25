import numpy as np
import csv
import matplotlib.pyplot as plt

def initialize_movies(yearRange = (1970, 2014), budgetCap = 1000000000):
    movies = []
    with open('321_movies.csv', newline='', encoding='cp850') as csvfile:
        reader = csv.DictReader(csvfile)
        for film in reader:
            if int(film['year']) in range(yearRange[0], yearRange[1]) and float(film['budget_2013']) < budgetCap:
                movies.append(film)
    return movies

"""
def get_budgets(movies):
    budgets = []
    for film in movies:
        budgets.append(film['budget_2013'])
    return budgets
"""
def get_metascore_budgets(movies):
    scores = []
    budgets = []
    for film in movies:
        if film['imdb_rating'] != 'NA' and film['budget_2013'] != 'NA':
            scores.append(float(film['imdb_rating']))
            budgets.append(float(film['budget_2013']))
    return scores, budgets


movies2010 = initialize_movies()


scores, budgets = get_metascore_budgets(movies2010)


plt.plot(scores, budgets, 'r.')
plt.show()
