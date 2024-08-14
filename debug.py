import pandas as pd
from jinja2 import Environment, FileSystemLoader
import pdfkit
import os
import base64

# Load the data
file_path = 'runs/2024-08-12 1944_adjusted_close.csv'
data = pd.read_csv(file_path)

# Ensure Z-Score column is numeric and handle NaN values
data['Z_Score'] = pd.to_numeric(data['Z-Score'], errors='coerce').fillna(0)

# Handle edge cases for current price being zero or very small
data['Current Price'] = data['Current Price'].replace(0, pd.NA).fillna(1e-6)  # Avoid division by zero

# Calculate prediction percentage change for both next week and next month
data['Next_Week_Prediction_Change'] = ((data['Next Week Prediction'] - data['Current Price']) / data['Current Price']) * 100
data['Next_Month_Prediction_Change'] = ((data['Next Month Prediction'] - data['Current Price']) / data['Current Price']) * 100

# Cap percentage changes to avoid extreme values
data['Next_Week_Prediction_Change'] = data['Next_Week_Prediction_Change'].clip(-500, 500)
data['Next_Month_Prediction_Change'] = data['Next_Month_Prediction_Change'].clip(-500, 500)

# Check if any percentage changes are abnormally high or low
abnormal_changes = data[(data['Next_Week_Prediction_Change'].abs() > 1000) | (data['Next_Month_Prediction_Change'].abs() > 1000)]
if not abnormal_changes.empty:
    print("Warning: Some percentage changes are abnormally high or low:")
    print(abnormal_changes[['CODE', 'Next_Week_Prediction_Change', 'Next_Month_Prediction_Change']])

# Inspect data for EOH.JO
eoh_data = data[data['CODE'] == 'EOH.JO']
print(eoh_data[['CODE', 'Current Price', 'Next Week Prediction', 'Next Month Prediction', 'Next_Week_Prediction_Change', 'Next_Month_Prediction_Change']])

# Define the metrics for which we need top 10 and bottom 10
metrics = [
    'Z_Score',
    'Next_Week_Prediction_Change',
    'Next_Month_Prediction_Change'
]

# Prepare the top 10 and bottom 10 for each metric and add metric columns directly
top_bottom_data = {}
for metric in metrics:
    top_10 = data.nlargest(10, metric)
    bottom_10 = data.nsmallest(10, metric)
    top_bottom_data[metric] = {
        'top_10': top_10,
        'bottom_10': bottom_10
    }

# Define the path for plots
plot_base_path = "plots"

# Function to encode image to base64
def encode_image_base64(image_path):
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    else:
        print(f"Image not found: {image_path}")
        return None

# Encode images for each stock in top and bottom 10 lists
for metric, data in top_bottom_data.items():
    for category in ['top_10', 'bottom_10']:
        for i, row in data[category].iterrows():
            code = row['CODE']
            close_prediction_img_path = os.path.join(plot_base_path, code.replace('.JO', ''), 'adj_close_prediction_compressed.jpg')
            adj_bollinger_img_path = os.path.join(plot_base_path, code.replace('.JO', ''), 'adj_bollinger_compressed.jpg')
            adj_overbought_img_path = os.path.join(plot_base_path, code.replace('.JO', ''), 'adj_overbought_oversold_compressed.jpg')
            
            # Add encoded images and current price to the dataframe
            data[category].at[i, 'close_prediction_img'] = encode_image_base64(close_prediction_img_path)
            data[category].at[i, 'adj_bollinger_img'] = encode_image_base64(adj_bollinger_img_path)
            data[category].at[i, 'adj_overbought_img'] = encode_image_base64(adj_overbought_img_path)
            data[category].at[i, 'current_price'] = row['Current Price']

# Create the HTML template for the report
env = Environment(loader=FileSystemLoader('.'))
template = env.get_template('report_template.html')

# Render the HTML content with the top and bottom 10 data
html_content = template.render(
    top_bottom_data=top_bottom_data
)

# Save the HTML content to a file
html_file_path = 'report.html'
with open(html_file_path, 'w') as file:
    file.write(html_content)

# Convert the HTML report to PDF
pdf_file_path = 'report.pdf'
pdfkit.from_file(html_file_path, pdf_file_path)

# Provide the path to the generated PDF file
print(f"PDF report generated: {os.path.abspath(pdf_file_path)}")