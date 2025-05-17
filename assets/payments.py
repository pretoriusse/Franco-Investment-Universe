import requests
import hashlib
import urllib

# Helper function to generate PayFast signature
def generate_signature(data, passphrase=''):
    # Ensure the order of parameters as required by PayFast
    order = [
        'merchant_id', 'merchant_key', 'return_url', 'cancel_url', 'notify_url',
        'name_first', 'email_address', 'm_payment_id', 'amount', 'item_name',
        'item_description', 'subscription_type', 'billing_date', 'recurring_amount',
        'frequency', 'cycles', 'subscription_notify_email', 'subscription_notify_webhook',
        'subscription_notify_buyer'
    ]
    
    # Build the payload in the specified order
    payload = ""
    for key in order:
        if key in data and data[key] != "":
            # URL encode values correctly, encode spaces as '+', and ensure uppercase encoding
            value = urllib.parse.quote_plus(data[key]).replace('%20', '+')
            payload += f"{key}={value}&"

    # Remove the trailing '&' or append the passphrase if it's not empty
    payload = payload[:-1]
    if passphrase:
        payload += f"&passphrase={passphrase}"
    
    # Generate the MD5 hash in lowercase
    return hashlib.md5(payload.encode()).hexdigest().lower()

def charge_customer(token, amount, item_name, item_description, app):
    # Define the API endpoint for charging a customer with a token
    url = 'https://www.payfast.co.za/eng/process'
    
    # Prepare the data for the request
    data = {
        'merchant_id': app.config['PAYFAST_MERCHANT_ID'],
        'merchant_key': app.config['PAYFAST_MERCHANT_KEY'],
        'token': token,
        'amount': f"{int(amount)}",
        'item_name': item_name,
        'item_description': item_description,
        'currency': 'ZAR',
    }

    # Include passphrase if set in the configuration
    passphrase = app.config.get('PAYFAST_PASSPHRASE', '')
    
    # Generate the signature
    signature = generate_signature(data, passphrase)
    data['signature'] = signature

    # Make the API request
    response = requests.post(url, data=data)

    # Check the response
    if response.status_code == 200:
        print("Payment successful.")
    else:
        print(f"Payment failed: {response.text}")
