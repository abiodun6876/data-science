import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


# Load the dataset
@st.cache_data
def load_data():
    # Replace 'bank_customers.csv' with the path to your CSV file
   
    data = pd.read_csv(r"bank_customers.csv")

    # Clean column names by stripping spaces (if necessary)
    data.columns = data.columns.str.strip()


    

    # Ensure numeric columns are treated correctly
    data['Balance'] = pd.to_numeric(data['Balance'], errors='coerce')
    data['Loan_Amount'] = pd.to_numeric(data['Loan_Amount'], errors='coerce')

    return data

# Load the data
df = load_data()

# Check column names to confirm they are correct
st.write(df.columns)

# Sidebar for filters
st.sidebar.header("Filters")
age_range = st.sidebar.slider("Select Age Range", int(df['Age'].min()), int(df['Age'].max()), (20, 50))
income_range = st.sidebar.slider("Select Income Range", int(df['Balance'].min()), int(df['Balance'].max()), (30000, 100000))  # Using Balance as income
gender_filter = st.sidebar.multiselect("Select Gender", options=df['Gender'].unique(), default=df['Gender'].unique())

# Filter data based on sidebar inputs
filtered_data = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]
filtered_data = filtered_data[(filtered_data['Balance'] >= income_range[0]) & (filtered_data['Balance'] <= income_range[1])]  # Using Balance as income
filtered_data = filtered_data[filtered_data['Gender'].isin(gender_filter)]

# Dashboard Title
st.title("Bank Customers Dashboard")
st.markdown("A professional dashboard displaying insights about bank customers.")

# Display filtered data
st.subheader("Filtered Data")
st.dataframe(filtered_data)

# Age distribution
st.subheader("Age Distribution")
age_chart = px.histogram(filtered_data, x='Age', nbins=20, title="Age Distribution of Customers", labels={'Age': 'Age'}, color_discrete_sequence=['#636EFA'])
st.plotly_chart(age_chart)

# Gender ratio
st.subheader("Gender Ratio")
gender_chart = px.pie(filtered_data, names='Gender', title="Gender Ratio", color_discrete_sequence=px.colors.sequential.RdBu)
st.plotly_chart(gender_chart)

# Average balance by age group
st.subheader("Average Balance by Age Group")
filtered_data['Age Group'] = pd.cut(filtered_data['Age'], bins=[0, 25, 35, 45, 55, 65, 75], labels=["0-25", "26-35", "36-45", "46-55", "56-65", "66-75"])
balance_chart = px.bar(filtered_data.groupby('Age Group')['Balance'].mean().reset_index(), 
                       x='Age Group', y='Balance', 
                       title="Average Balance by Age Group", 
                       color_discrete_sequence=['#EF553B'])
st.plotly_chart(balance_chart)

# Loan ownership by income group
st.subheader("Loan Ownership by Income Group")
filtered_data['Income Group'] = pd.cut(filtered_data['Balance'], bins=[0, 50000, 100000, 150000, 200000], 
                                       labels=["0-50k", "50k-100k", "100k-150k", "150k-200k"])
loan_chart = px.bar(filtered_data.groupby('Income Group')['Loan_Amount'].sum().reset_index(),  # Assuming Loan_Amount represents loan ownership
                    x='Income Group', y='Loan_Amount', 
                    title="Loan Ownership by Income Group", 
                    color_discrete_sequence=['#00CC96'])
st.plotly_chart(loan_chart)

# Correlation heatmap
st.subheader("Correlation Heatmap")

# Select only numeric columns for correlation
numeric_data = filtered_data.select_dtypes(include=['float64', 'int64'])

# Compute correlation matrix
corr = numeric_data.corr()

loan_status_filter = st.sidebar.multiselect(
    "Select Loan Status", 
    options=filtered_data['Loan_Status'].unique(), 
    default=filtered_data['Loan_Status'].unique()
)
filtered_data = filtered_data[filtered_data['Loan_Status'].isin(loan_status_filter)]

st.subheader("Key Performance Indicators")
total_customers = len(filtered_data)
average_balance = filtered_data['Balance'].mean()
total_loans = filtered_data['Loan_Amount'].sum()
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", total_customers)
col2.metric("Average Balance", f"${average_balance:,.2f}")
col3.metric("Total Loans Issued", f"${total_loans:,.2f}")

if 'Date' in filtered_data.columns:
    filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])
    monthly_data = filtered_data.groupby(filtered_data['Date'].dt.to_period('M')).sum().reset_index()
    time_series_chart = px.line(monthly_data, x='Date', y='Balance', title="Balance Trends Over Time")
    st.plotly_chart(time_series_chart)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
filtered_data['Cluster'] = kmeans.fit_predict(filtered_data[['Balance', 'Loan_Amount']])
cluster_chart = px.scatter(filtered_data, 
                            x='Balance', y='Loan_Amount', 
                            color='Cluster', 
                            size='Age', 
                            hover_data=['Customer_ID'], 
                            title="Customer Segmentation (KMeans Clustering)")
st.plotly_chart(cluster_chart)

search_query = st.sidebar.text_input("Search Customer by ID or Name", "")
if search_query:
    filtered_data = filtered_data[
        filtered_data['Customer_ID'].astype(str).str.contains(search_query, case=False) | 
        filtered_data['Name'].str.contains(search_query, case=False)
    ]

from sklearn.linear_model import LogisticRegression
# Example model training (replace with actual model logic)
model = LogisticRegression()
X = filtered_data[['Age', 'Balance']]
y = (filtered_data['Loan_Status'] == 'Approved').astype(int)
model.fit(X, y)
st.sidebar.markdown("### Predict Loan Approval")
age_input = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
balance_input = st.sidebar.number_input("Balance", min_value=0, value=50000)
prediction = model.predict([[age_input, balance_input]])
st.sidebar.write("Loan Approval Prediction:", "Approved" if prediction[0] else "Rejected")

st.download_button(
    label="Download Filtered Data",
    data=filtered_data.to_csv(index=False),
    file_name="filtered_data.csv",
    mime="text/csv"
)


balance_threshold = st.sidebar.slider("Balance Threshold for Loan Default", 0, 100000, 20000)
high_risk_customers = filtered_data[filtered_data['Balance'] < balance_threshold]
st.subheader("High-Risk Customers (Loan Default)")
st.write(f"Number of High-Risk Customers: {len(high_risk_customers)}")
risk_chart = px.scatter(high_risk_customers, x='Balance', y='Loan_Amount', title="High-Risk Customers")
st.plotly_chart(risk_chart)

if 'Location' in filtered_data.columns:
    st.subheader("Customer Distribution Map")
    map_chart = px.scatter_geo(filtered_data, locationmode='country names', 
                               locations='Location', 
                               title="Customer Distribution")
    st.plotly_chart(map_chart)

# Plot correlation heatmap
heatmap = go.Figure(data=go.Heatmap(z=corr.values, 
                                    x=corr.columns, 
                                    y=corr.columns, 
                                    colorscale='Viridis'))
heatmap.update_layout(title="Correlation Heatmap", xaxis_nticks=36)
st.plotly_chart(heatmap)

# Customer Segmentation
st.subheader("Customer Segmentation")
cluster_chart = px.scatter(filtered_data, 
                           x='Balance', y='Loan_Amount',  # Plotting Balance vs Loan_Amount for segmentation
                           color='Loan_Status', 
                           size='Age', 
                           hover_data=['Customer_ID'], 
                           title="Customer Segmentation (Balance vs Loan Amount)")
st.plotly_chart(cluster_chart)

# Add Footer
st.markdown("---")
st.markdown("**Developed by AKINPELU ABIODUN MOSES**")
