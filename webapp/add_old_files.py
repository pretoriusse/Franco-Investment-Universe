from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
from assets.const import DB_PARAMS_WEBAPP
from webapp.models import HTMLWebView
import os
from datetime import datetime
import re
from typing import Optional

# Establish database connection
webapp_engine = create_engine(
    f"postgresql://{DB_PARAMS_WEBAPP['user']}:{DB_PARAMS_WEBAPP['password']}@{DB_PARAMS_WEBAPP['host']}:{DB_PARAMS_WEBAPP['port']}/{DB_PARAMS_WEBAPP['dbname']}"
)
WebApp_Session = sessionmaker(bind=webapp_engine)

# Get the path to reports
path = '/mnt/backups/Shares/Reports' #os.path.abspath(__file__).replace('add_old_files.py', '').replace('webapp', 'reports')

files = os.walk(path)
files_dict = {}

def extract_user_id(text: str) -> Optional[int]:
    # Search for "user" optionally followed by an underscore and then digits.
    match = re.search(r'user[_]?(\d+)', text)
    if match:
        return int(match.group(1))
    return None

# Loop over files to build dictionary structure
for root, subdirs, filenotused in files:
    for subdir in subdirs:
        files_dict[subdir] = {}
        subdirfiles = os.walk(os.path.join(root, subdir))
        for rootdir, _, files1 in subdirfiles:
            for file in files1:
                key = file.replace('.', '_')
                files_dict[subdir][key] = os.path.join(root, subdir, file)
                print(os.path.join(root, subdir, file))

# Insert data into the database
with WebApp_Session() as session:
    for key in files_dict.keys():
        try:
            display_date = key  # Assuming the folder name is the display date
            actual_run_date = datetime.strptime(display_date, '%Y-%m-%d')  # Convert to actual date
            print(files_dict[key])

            # Prepare paths for HTML and PDFs
            html_path = files_dict[key].get('adjusted_close_detailed_html', '')
            html_summary_path = files_dict[key].get('adjusted_close_summary_html', '')
            pdf_summary_path = files_dict[key].get('adjusted_close_summary_pdf', '')
            pdf_detailed_path = files_dict[key].get('adjusted_close_detailed_pdf', '')

            # Loop through all file paths to find a user id
            user = None
            for path_item in files_dict[key].values():
                user = extract_user_id(path_item)
                if user is not None:
                    break

            if user is None:
                print(f"User ID not found in any path: {files_dict[key]}")
                continue

            # Create a new instance of HTMLWebView
            webview_entry = HTMLWebView(
                display_date=f"{display_date}",
                report_type='Adjusted Close',
                html_detailed_path=html_path,
                html_summary_path=html_summary_path,
                pdf_summary_path=pdf_summary_path,
                pdf_detailed_path=pdf_detailed_path,
                actual_run_date=actual_run_date,
                subscriber_id=user
            )

            # Add to session and commit
            session.add(webview_entry)
            try:
                # Prepare paths for HTML and PDFs
                html_path = files_dict[key]['Close_detailed_html']
                html_summary_path = files_dict[key]['Close_summary_html']
                pdf_summary_path = files_dict[key]['Close_summary_pdf']
                pdf_detailed_path = files_dict[key]['Close_detailed_pdf']

                # Create a new instance of HTMLWebView
                webview_entry1 = HTMLWebView(
                    display_date=f"{display_date}",
                    report_type='Close',
                    html_detailed_path=html_path,
                    html_summary_path=html_summary_path,
                    pdf_summary_path=pdf_summary_path,
                    pdf_detailed_path=pdf_detailed_path,
                    actual_run_date=actual_run_date,
                    subscriber_id=user
                )

                # Add to session and commit
                session.add(webview_entry1)
            except KeyError:
                pass

            try:
                session.commit()
                #print(f"Inserted record for {display_date}")
            except Exception as e:
                session.rollback()
                print(f"Failed to insert record for {display_date}: {e}")
        except Exception as e:
            continue
