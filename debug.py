import pandas as pd
from jinja2 import Environment, FileSystemLoader
import pdfkit
import base64
import json

# Load the data
file_path = 'runs/2024-08-15 1949_adjusted_close.csv'
data = pd.read_csv(file_path)

def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        return None

def prepare_stock_images(top_bottom_data):
    stock_images = []
    added_tickers = set()

    for metric in top_bottom_data:
        for group in ['top_10', 'bottom_10']:
            for entry in top_bottom_data[metric][group]:
                ticker = entry['code']
                if ticker not in added_tickers:
                    stock_img = {
                        'name': entry['share_name'],
                        'ticker': ticker,
                        'prediction': encode_image_to_base64(f'plots/{ticker.replace(".JO", "")}/adj_close_prediction_compressed.jpg'),
                        'bollinger': encode_image_to_base64(f'plots/{ticker.replace(".JO", "")}/adj_bollinger_compressed.jpg'),
                        'overbought_oversold': encode_image_to_base64(f'plots/{ticker.replace(".JO", "")}/adj_overbought_oversold_compressed.jpg')
                    }
                    stock_images.append(stock_img)
                    added_tickers.add(ticker)

    return stock_images

def create_html_summary(data, total_value_next_week, total_value_next_month, template):
    summary = create_summary(data, total_value_next_week, total_value_next_month)
    html_content = template.render(stocks=data.to_dict(orient='records'), summary=summary)
    return html_content

def create_summary(data, total_value_next_week, total_value_next_month):
    try:
        total_invested = data['Initial Purchase Amount'].sum()
    except Exception:
        total_invested = 1

    try:
        current_value = data['Current Value'].sum()
    except Exception:
        current_value = 0

    profit_loss = current_value - total_invested
    summary = (
        f"Total Invested: R{total_invested:,.2f}<br>"
        f"Current Value: R{current_value:,.2f}<br>"
        f"Profit/Loss: R{profit_loss:,.2f} ({(profit_loss / total_invested) * 100:,.2f}%)<br>"
        f"Projected Portfolio Value (Next Week): R{total_value_next_week:,.2f}<br>"
        f"Projected Portfolio Value (Next Month): R{total_value_next_month:,.2f}"
    )
    return summary

def create_detailed_pdf(data, stock_images, filename, total_value_next_week, total_value_next_month, summary_report=False):
    print(f"Creating PDF report: {filename}")
    options = {
        'page-size': 'Letter',
        'encoding': "UTF-8"
    }

    env = Environment(loader=FileSystemLoader('.'))

    if summary_report:
        print("Preparing summary report...")
        data['Z_Score'] = pd.to_numeric(data['Z-Score'], errors='coerce').fillna(0)
        data['Current Price'] = data['Current Price'].replace(0, pd.NA).fillna(1e-6)
        data['Next_Week_Prediction_Change'] = ((data['Next Week Prediction'] - data['Current Price']) / data['Current Price']) * 100
        data['Next_Month_Prediction_Change'] = ((data['Next Month Prediction'] - data['Current Price']) / data['Current Price']) * 100

        metrics = ['Z_Score', 'Next_Week_Prediction_Change', 'Next_Month_Prediction_Change', 'Overbought_Oversold_Value', 'SECTOR RSI 1M', 'SECTOR RSI 3M', 'SECTOR RSI 6M', 'MARKET RSI 1M', 'MARKET RSI 3M', 'MARKET RSI 6M']
        top_bottom_data = {
            metric: {
                'top_10': data.nlargest(10, metric).to_dict(orient='records'),
                'bottom_10': data.nsmallest(10, metric).to_dict(orient='records')
            }
            for metric in metrics
        }

        # Prepare stock images based on top/bottom data
        stock_images = prepare_stock_images(top_bottom_data)

        print(json.dumps(top_bottom_data, indent=4))

        template = env.get_template('summary_template.html')
        rendered = template.render(
            top_bottom_data=top_bottom_data,
            summary=create_summary(data, total_value_next_week, total_value_next_month),
            stock_images=stock_images
        )

    else:
        template = env.get_template('detailed_template.html')
        rendered = template.render(
            stocks=data.to_dict(orient='records'),
            summary=create_summary(data, total_value_next_week, total_value_next_month),
            stock_images=stock_images
        )

    # Write the HTML to a file for inspection
    html_file_path = filename.replace('.pdf', '.html')
    with open(html_file_path, 'w') as file:
        file.write(rendered)

    # Convert the HTML report to PDF
    pdfkit.from_file(html_file_path, filename, options=options)

    print(f"PDF report created at: {filename}")

# Generate the PDF report with the stock images
create_detailed_pdf(data=data, stock_images=[], filename='reports/test.pdf', summary_report=True, total_value_next_month=0, total_value_next_week=0)
