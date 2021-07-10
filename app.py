import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np
from joblib import load
import sklearn

TITLE = "Streamlit demonstration"
st.title(TITLE)
DESCRIPTION = "Survivorship analysis from the Titanic Dataset"
st.markdown(DESCRIPTION)
st.sidebar.title(TITLE)
st.sidebar.markdown("Choose your visualizations.")

@st.cache
def load_data():
    data = pd.read_csv("titanic.csv")
    data = data.drop(columns=["Name"])
    return data
data = load_data()
st.write(data)

@st.cache
def subset_by_sex(df):
    df = df.groupby(['Sex', 'Survived']).count().iloc[:, :1].rename(columns={"Pclass": "Count"})
    df = df.reset_index()
    df["Survived"] = df["Survived"].astype(str)
    return df

if st.sidebar.checkbox("Male and Female comparative", True):
    sex_subset = subset_by_sex(data)
    st.header("Male and female comparative")
    if st.checkbox("Show table", False):
        st.write(sex_subset)
        pass

    plot_choice = st.radio("Choose plotting tool", ['Pyplot', 'Plotly express'])
    st.header("Survivors and non-survivors by Sex")
    if plot_choice == 'Pyplot':
        plt_fig = plt.figure()
        sns.countplot(data=data, x="Sex", hue="Survived", palette=["royalblue", "orangered"])
        sns.despine()
        st.pyplot(plt_fig)
        pass
    else:
        px_fig = px.bar(sex_subset, y="Count", x="Sex", color="Survived", barmode="group")
        st.plotly_chart(px_fig)
        pass

@st.cache
def get_fares_min_max(df):
    min_fare = df['Fare'].min()
    max_fare = df['Fare'].max()
    return float(min_fare), float(max_fare)

def subset_by_fare(df, lower, upper):
    df = df[df['Fare'] >= lower]
    df = df[df['Fare'] <= upper]
    return df

if st.sidebar.checkbox("Analysis by fares", False):
    st.header("Analysis by fares")
    min_fare, max_fare = get_fares_min_max(data)
    lower, upper = st.slider("Fare range", min_fare, max_fare, (100.0, float(max_fare/2)))
    fare_subset = subset_by_fare(data, lower, upper)
    st.write(f"Number of observations: {fare_subset.shape[0]}")
    if st.checkbox("Show table", False, key=0): # TRY WITHOUT KEY FIRST
        st.write(fare_subset)
        pass

    plt_fig = plt.figure()
    sns.countplot(data=fare_subset, x="Survived", palette=["royalblue", "orangered"])
    sns.despine()
    plt.title("Survivors vs. Non survivors.")
    st.pyplot(plt_fig)
    pass

def subset_by_family(df, col, values):
    if values:
        df = df[df[col].isin(values)]
    else:
        st.error("Please select values.")
    return df

if st.sidebar.checkbox("Family aboard analysis", False):
    st.header("Family aboard analysis")
    col1, col2 = st.beta_columns(2)

    col1.header("Siblings/Spouses aboard")
    sibl_spou = col1.multiselect("Quantities", np.sort(data["Siblings/Spouses Aboard"].unique()), key=0)
    siblings_spouses_df = subset_by_family(data, "Siblings/Spouses Aboard", sibl_spou)
    plt_fig = plt.figure()
    sns.countplot(data=siblings_spouses_df, x="Survived", palette=["royalblue", "orangered"])
    sns.despine()
    plt.title("Survivors and non survivors")
    col1.pyplot(plt_fig, use_column_width=True)

    col2.header("Parents/Children aboard")
    par_chil = col2.multiselect("Quantities", np.sort(data["Parents/Children Aboard"].unique()), key=1)
    parents_children_df = subset_by_family(data, "Parents/Children Aboard", par_chil)
    plt_fig = plt.figure()
    sns.countplot(data=parents_children_df, x="Survived", palette=["royalblue", "orangered"])
    sns.despine()
    plt.title("Survivors and non survivors")
    col2.pyplot(plt_fig, use_column_width=True)


clf = load('rf.joblib')
if st.sidebar.checkbox("Machine Learning: would I survive the Titanic disaster?", False):
    st.header("Would I survive the disaster?")

    f = st.form(key='ml_form')
    p_class = f.selectbox("Class (1 is highest and 3 is lowest)", [1, 2, 3])
    sex = f.radio("Sex", ["Male", "Female"])
    if sex == "Male":
        sex_female = 0
        sex_male = 1
    elif sex == "Female":
        sex_female = 1
        sex_male = 0
    else:
        sex_female = None
        sex_male = None

    age = f.slider(label='Age', min_value=0, max_value=120)
    siblings_spouses = f.selectbox('Siblings and spouse aboard',
                                    [i for i in range(0,10)])
    parents_children = f.selectbox("Parents/Children aboard",
                                    [i for i in range(0,10)])
    min_fare, max_fare = get_fares_min_max(data)
    fare = f.slider("Fare", min_fare, max_fare, 150.0)

    submitted = f.form_submit_button(label='Submit!')
    if submitted:
        my_data = np.array([p_class, age, siblings_spouses, parents_children,
                            fare, sex_female, sex_male])
        try:
            prediction = clf.predict([my_data])
        except Exception as e:
            f.error(f"Could not make prediction. {e}")
        if prediction[0] == 0:
            st.error("Sorry! You would not survive :(")
        else:
            st.warning("Woot! You would survive!")

    pass
