# Import packages
from dash import Dash, html, dash_table, dcc
import pandas as pd
import plotly.express as px

# Incorporate data
df = pd.read_csv('C:/Users/Paulj/.cache/kagglehub/datasets/asinow/car-price-dataset/versions/1/car_price_dataset.csv')

print(df.columns)
# Initialize the app
app = Dash()

# App layout
app.layout = [
    html.Div(children='My First App with Data and a Graph'),
    dash_table.DataTable(data=df.to_dict('records'), page_size=10),
    dcc.Graph(figure=px.histogram(df, x='Mileage', y='Price', histfunc='avg'))
]

# Run the app
if __name__ == '__main__':
    app.run(debug=True)