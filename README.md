# Baseball Machine Learning Project

## Background
- I first became interested in Data Science through baseball. Shohei Ohtani's first MVP season, and the Giant's miracle 107 win season in 2021 reignited a passion for baseball. In addition to the action on the field, I became interested in the underlying numbers and statistics. Starting from batting average and ERA, I then learned more about stats like WRC+, barrel rate, exit velocity, etc. 
- This motivated me to start taking data science courses in college, to learn more about the theory of applying data. 
- I started a data science project related to baseball a few years ago, but did not finish it, due to my lack of knowledge at the time. Now, with my enhanced knowledge in both programming and statistics, I wanted to revisit this topic and build something I could really be proud of.

## Goal
- Build a machine learning model that can predict a player's batting performance next season, as measured by WRC+
    - Learn how to apply machine learning principles you have learned in class and build a fully fleged machine learning model.

## Context
- WRC+ (short for weighted runs created plus) is a statistic that measures a player's offensive output, that takes into account ballpark factors and run environment. This means that the statistic takes into account both factors specific to each baseball stadium (baseball stadiums are not standardized and differ from team to team), and the average offensive performance of all players at the time --some years have generally higher offensive output compared to other years. In 2019 the league offensive performance increased accross the board due to MLB using baseball that bounced off of the bat more. \
- Since WRC+ adjusts for these factors, we can isolate the player's performance mostly to each player making out analysis more thorough and accurate.

## Data Sources
Used data scraped from baseball reference, statcast, and fangraphs using the `pybaseball` package.
More info here: https://github.com/jldbc/pybaseball/tree/master
