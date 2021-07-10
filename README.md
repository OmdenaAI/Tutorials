# Streamlit Workshop code for Omdena's Learning Session
## Live on July 1st 2021
All the code for the Streamlit workshop for Omdena. Please be aware that some things are intentionally simplified because this is the code used for the Workshop. 

## Data
For this workshop I used the famous [Titanic Dataset](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html) to reduce the amount of preprocessing and focus on Streamlit features. The file is the [titanic.csv](https://github.com/victoraccete/omdena_streamlit_workshop/blob/c9330fd2c4a8b62a07d21a35386a421497c159dc/titanic.csv) file within the folder. 

## Relevant links
- Streamlit website: https://streamlit.io
- Request invite to Streamlit share: https://streamlit.io/sharing
- Streamlit share main page: https://share.streamlit.io
- Streamlit documentation: https://docs.streamlit.io/en/stable/installation.html
- Streamlit widgets: https://docs.streamlit.io/en/stable/main_concepts.html#widgets

## Requirements
- pandas==1.0.5
- seaborn==0.10.1
- streamlit==0.83.0
- plotly==4.11.0
- matplotlib==3.2.2
- numpy==1.17.0
- joblib==0.16.0
- scikit-learn==0.23.1

I used Python 3.7. Streamlit currently requires Python 3.6-3.8. 

## ML Model
The [.joblib file](https://github.com/victoraccete/omdena_streamlit_workshop/blob/c9330fd2c4a8b62a07d21a35386a421497c159dc/rf.joblib) was generated using joblib and the code in [this notebook](https://github.com/victoraccete/omdena_streamlit_workshop/blob/c9330fd2c4a8b62a07d21a35386a421497c159dc/Very_simple_titanic_ML_model.ipynb). 

## Hosting on Streamlit Share
Obs: locally the code was running fine, but in order to work on Streamlit Share I also had to change the return of this function:
https://github.com/victoraccete/omdena_streamlit_workshop/blob/c9330fd2c4a8b62a07d21a35386a421497c159dc/app.py#L53-L56

The [requirements.txt](https://github.com/victoraccete/omdena_streamlit_workshop/blob/c9330fd2c4a8b62a07d21a35386a421497c159dc/requirements.txt) file is necessary for streamlit share.  

[This is the link](https://share.streamlit.io/victoraccete/omdena_streamlit_workshop/main/app.py) to access this app on Streamlit Share. 
