# 🗺️ Data Exploration

The notebooks in this folder explore the (missing) data in the dataset.

## 🔎 exploration.ipynb

In this notebook investigates some graph building techniques, the final implemantation can be found in 📁utils/helpers.py

## 🚧 missing_data.ipynb

This notebook analyses when and which data is missing.
Only `sm_mean` and `sm_var` are missing, however in 2014 there is a substantial amount missing, since none of the stations reported any data on these two variables.

Also stations near the cost are missing `sm_mean` and `sm_var` more often than stations inland stations.

## 🎲 crps_plot.ipynb

Plots of CRPS for explanations in the thesis

## 🌦️ stations_plot.ipynb

Plot of Map with stations in Germany