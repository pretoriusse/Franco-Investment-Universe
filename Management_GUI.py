import json
from tkinter import filedialog
import pandas as pd
import yfinance as yf
import tkinter as tk
from tkinter import ttk, messagebox
from sqlalchemy import create_engine, Column, String, Numeric, Date, CHAR
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import PendingRollbackError, IntegrityError, SQLAlchemyError
from datetime import datetime, date
from assets.models import Portfolio, Stock, Industry, SubIndustry, Stock, TickerName, RSIComparisonMarket, RSIComparisonSector, PortfolioStock
import os

base_dir = os.path.abspath(os.path.dirname(__file__))
print(base_dir)

# Load Database Configuration from JSON
def load_db_config():
    with open(os.path.join(base_dir, 'db_config.json'), 'r') as config_file:
        return json.load(config_file)

def save_db_config(config):
    with open(os.path.join(base_dir, 'db_config.json'), 'w') as config_file:
        json.dump(config, config_file, indent=4)

config = load_db_config()

# Create the Database Engine Using Configurations
engine = create_engine(f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}")
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

class ViData(Base):
    __tablename__ = 'vi_data'

    code = Column(String(20), primary_key=True)  # Part of the composite primary key
    eps = Column(Numeric)
    nav = Column(Numeric)
    sales = Column(Numeric)
    eps_growth_f = Column(String(10))
    roe_f = Column(Numeric)
    inst_profit_margin_f = Column(Numeric)
    sales_growth_f = Column(String(10))
    holding = Column(CHAR(1))
    shares = Column(Numeric)
    interest_cover = Column(Numeric)
    comment = Column(String(50))
    tnav = Column(Numeric)
    rote = Column(Numeric)
    actual_roe = Column(Numeric)
    last_update = Column(String(20))
    o_margin = Column(Numeric)
    div = Column(Numeric)
    cash_ps = Column(Numeric)
    act = Column(Numeric)
    heps = Column(Numeric)
    quality_rating = Column(String(10))
    div_decl = Column(Numeric)
    div_ldt = Column(Date)
    div_pay = Column(Date)
    rec = Column(String(20))
    rec_on = Column(String(20))
    ye_release = Column(Date)
    int_release = Column(Date)
    rec_price = Column(Numeric)
    share_price = Column(Numeric)
    peg = Column(Numeric)
    peg_pe = Column(Numeric)
    peg_pe_value = Column(Numeric)
    peg_nav = Column(String(10))
    peg_pe_nav_value = Column(Numeric)
    run_date = Column(Date, primary_key=True)  # Part of the composite primary key

    __table_args__ = (
        {'primary_key': (code, run_date)}  # Define the composite primary key
    )

# Create tables if not exists
Base.metadata.create_all(engine)

class InvestmentManagerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Investment Universe Manager")
        self.root.geometry("1400x700")  # Set the window size

        # Set the window icon
        try:
            self.root.iconbitmap(os.path.join(base_dir, 'stocks.ico'))
        except:
            pass

        # Create Notebook for Tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        # Create tabs
        self.create_rsi_comparison_markets_tab()  # Call this before the stocks management tab
        self.create_rsi_comparison_sectors_tab()  # Call this before the stocks management tab
        self.create_industry_management_tab()
        self.create_sub_industry_management_tab()  # New tab for sub-industries
        self.create_stocks_management_tab()
        self.create_excel_upload_tab() 
        self.create_settings_tab()
        self.create_portfolio_tab()

        # Start the periodic update of industries
        self.update_industries_periodically()
        self.industry_listbox.bind("<Double-1>", self.edit_industry_popup)
        self.sub_industry_listbox.bind("<Double-1>", self.edit_sub_industry_popup)
        self.stock_listbox.bind("<Double-1>", self.edit_stock_popup)
        self.portfolio_tree.bind("<Double-1>", self.edit_portfolio_popup)

    def create_industry_management_tab(self):
        industry_frame = ttk.Frame(self.notebook)
        self.notebook.add(industry_frame, text="Manage Industries")

        # Industry List
        self.industry_listbox = tk.Listbox(industry_frame, width=40)
        self.industry_listbox.pack(side="left", fill="y", padx=10, pady=10)

        # Add New Industry
        add_industry_frame = ttk.Frame(industry_frame)
        add_industry_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.new_industry_name = tk.StringVar()
        ttk.Label(add_industry_frame, text="New Industry Name:").pack(anchor="w")
        ttk.Entry(add_industry_frame, textvariable=self.new_industry_name, width=40).pack(anchor="w")

        ttk.Button(add_industry_frame, text="Add Industry", command=self.add_industry).pack(anchor="w")

        self.load_industries()

    def create_sub_industry_management_tab(self):
        sub_industry_frame = ttk.Frame(self.notebook)
        self.notebook.add(sub_industry_frame, text="Manage Sub-Industries")

        # Sub-Industry List
        self.sub_industry_listbox = tk.Listbox(sub_industry_frame, width=80)
        self.sub_industry_listbox.pack(side="left", fill="y", padx=10, pady=10)

        # Add New Sub-Industry
        add_sub_industry_frame = ttk.Frame(sub_industry_frame)
        add_sub_industry_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.selected_industry_for_sub = tk.StringVar()
        self.new_sub_industry_name = tk.StringVar()

        ttk.Label(add_sub_industry_frame, text="Industry:").pack(anchor="w")
        self.industry_dropdown_for_sub = ttk.Combobox(add_sub_industry_frame, textvariable=self.selected_industry_for_sub, width=40)
        self.industry_dropdown_for_sub.pack(anchor="w")

        ttk.Label(add_sub_industry_frame, text="New Sub-Industry Name:").pack(anchor="w")
        ttk.Entry(add_sub_industry_frame, textvariable=self.new_sub_industry_name, width=42).pack(anchor="w")

        ttk.Button(add_sub_industry_frame, text="Add Sub-Industry", command=self.add_sub_industry).pack(anchor="w")

        self.load_industries_dropdown_for_sub()
        self.load_sub_industries()

    def create_stocks_management_tab(self):
        stock_frame = ttk.Frame(self.notebook)
        self.notebook.add(stock_frame, text="Manage Stocks")

        # Stocks List
        self.stock_listbox = tk.Listbox(stock_frame, width=110)
        self.stock_listbox.pack(side="left", fill="y", padx=10, pady=10)

        # Add/Edit Stock
        add_stock_frame = ttk.Frame(stock_frame)
        add_stock_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.stock_code = tk.StringVar()
        self.selected_industry = tk.StringVar()
        self.selected_sub_industry = tk.StringVar()
        self.rsi_comparison_market = tk.StringVar(value='%5EJ300.JO')
        self.rsi_comparison_sector = tk.StringVar()
        self.is_commodity = tk.IntVar()
        self.override_market = tk.IntVar()

        ttk.Label(add_stock_frame, text="Stock Code:").pack(anchor="w")
        stock_code_entry = ttk.Entry(add_stock_frame, textvariable=self.stock_code, width=20)
        stock_code_entry.pack(anchor="w")
        stock_code_entry.bind("<KeyRelease>", lambda e: self.stock_code.set(self.stock_code.get().upper()))

        ttk.Label(add_stock_frame, text="Industry:").pack(anchor="w")
        self.industry_dropdown = ttk.Combobox(add_stock_frame, textvariable=self.selected_industry, width=40)
        self.industry_dropdown.pack(anchor="w")
        self.industry_dropdown.bind("<<ComboboxSelected>>", self.load_sub_industries_dropdown)

        ttk.Label(add_stock_frame, text="Sub-Industry:").pack(anchor="w")
        self.sub_industry_dropdown = ttk.Combobox(add_stock_frame, textvariable=self.selected_sub_industry, width=40)
        self.sub_industry_dropdown.pack(anchor="w")

        ttk.Label(add_stock_frame, text="RSI Comparison Market:").pack(anchor="w")
        self.rsi_market_dropdown = ttk.Combobox(add_stock_frame, textvariable=self.rsi_comparison_market, width=40)  # Define rsi_market_dropdown
        self.rsi_market_dropdown.pack(anchor="w")

        ttk.Label(add_stock_frame, text="RSI Comparison Sector:").pack(anchor="w")
        self.rsi_sector_dropdown = ttk.Combobox(add_stock_frame, textvariable=self.rsi_comparison_sector, width=40)
        self.rsi_sector_dropdown.pack(anchor="w")

        ttk.Checkbutton(add_stock_frame, text="Commodity", variable=self.is_commodity).pack(anchor="w")

        ttk.Button(add_stock_frame, text="Add Stock", command=self.add_stock).pack(anchor="w")

        self.load_industries_dropdown()
        self.load_rsi_comparison_sectors()
        self.load_rsi_comparison_markets()  # Load RSI comparison markets into the dropdown

    def load_industries_dropdown(self):
        industries = session.query(Industry).all()
        self.industry_dropdown['values'] = [industry.name for industry in industries]

    def load_sub_industries_dropdown(self, event=None):
        selected_industry_name = self.selected_industry.get()
        industry = session.query(Industry).filter_by(name=selected_industry_name).first()
        if industry:
            sub_industries = session.query(SubIndustry).filter_by(industry_id=industry.id).all()
            self.sub_industry_dropdown['values'] = [sub.name for sub in sub_industries]

    def add_industry(self):
        try:
            name = self.new_industry_name.get().strip()
            if name:
                industry = Industry(name=name)
                session.add(industry)
                session.commit()
                self.load_industries()  # Update the list immediately after adding a new industry
                self.load_industries_dropdown_for_sub()  # Update dropdown for sub-industry management
                self.show_temp_message(title='Success', message=f"Industry '{name}' has been added!", duration=2000)
                self.new_industry_name.set('')
                self.update_data()
            else:
                messagebox.showwarning("Input Error", "Industry name cannot be empty.")

        except (PendingRollbackError, IntegrityError):
            session.rollback()
            messagebox.showerror("Input Error", f"{name} already exists.")
            self.load_industries()  # Update the list immediately after adding a new industry
            self.load_industries_dropdown_for_sub()  # Update dropdown for sub-industry management

    def add_sub_industry(self):
        industry_name = self.selected_industry_for_sub.get().strip()
        sub_industry_name = self.new_sub_industry_name.get().strip()
        try:
            if industry_name and sub_industry_name:
                industry = session.query(Industry).filter_by(name=industry_name).first()
                if industry:
                    sub_industry = SubIndustry(name=sub_industry_name, industry_id=industry.id)
                    session.add(sub_industry)
                    session.commit()
                    self.load_sub_industries()  # Update the list immediately after adding a new sub-industry
                    self.show_temp_message(title='Success', message=f"{sub_industry_name} has been added!", duration=1000)
                    self.new_sub_industry_name.set('')
                    self.update_data()
                else:
                    messagebox.showwarning("Input Error", "Selected industry does not exist.")
            else:
                messagebox.showwarning("Input Error", "Both industry and sub-industry names cannot be empty.")

        except (PendingRollbackError, IntegrityError):
            session.rollback()
            messagebox.showerror("Input Error", f"{sub_industry_name} already exists.")
            self.load_industries()  # Update the list immediately after adding a new industry
            self.load_industries_dropdown_for_sub()  # Update dropdown for sub-industry management

    def load_industries_dropdown_for_sub(self):
        industries = session.query(Industry).all()
        self.industry_dropdown_for_sub['values'] = [industry.name for industry in industries]

    def load_sub_industries(self):
        self.sub_industry_listbox.delete(0, tk.END)
        sub_industries = session.query(SubIndustry).all()
        for sub_industry in sub_industries:
            industry = session.query(Industry).filter_by(id=sub_industry.industry_id).first()
            self.sub_industry_listbox.insert(tk.END, f"{sub_industry.name} (Industry: {industry.name})")

    def add_stock(self):
        # Force uppercase for CODE, RSI Comparison Market, and RSI Comparison Sector
        codes = self.stock_code.get().strip().upper()
        industry_name = self.selected_industry.get().strip()
        sub_industry_name = self.selected_sub_industry.get().strip()
        rsi_market = self.rsi_comparison_market.get().strip().upper()
        rsi_sector = self.rsi_comparison_sector.get().strip().upper()
        commodity = self.is_commodity.get()

        codes = codes.replace(' ', '')

        if not ',' in codes:
            codes = codes + ','

        if not codes or not industry_name or not sub_industry_name:
            messagebox.showwarning("Input Error", "Please fill in all required fields.")
            return
        codes = codes.split(',')
        if codes[-1] == '':
            codes.pop(-1)
        
        for code in codes:
            try:
                
                self.show_temp_message("Searching", f"Searching Yahoo Finance for {code}", duration=2000)

                # Check if the ticker already exists in the database
                existing_ticker = session.query(Stock).filter(Stock.code.ilike(code)).first()
                if existing_ticker:
                    messagebox.showerror("Duplicate Ticker", f"The ticker '{code}' already exists in the database.")
                    continue

                # Check Yahoo Finance for the ticker
                stock_data = yf.Ticker(code)
                print(stock_data.info)  # Debug print
                if 'shortName' not in stock_data.info:
                    messagebox.showerror("Invalid Ticker", "The stock code entered does not exist on Yahoo Finance.")
                    continue

                # Get the correct stock name from Yahoo Finance
                correct_share_name = stock_data.info['shortName']

                # Ask user if the share name is correct
                if not messagebox.askyesno("Confirm Share Name", f"Is the share name '{correct_share_name}' correct?"):
                    return

                # Add the stock to the stocks table
                industry = session.query(Industry).filter_by(name=industry_name).first()
                sub_industry = session.query(SubIndustry).filter_by(name=sub_industry_name, industry_id=industry.id).first()

                stock = Stock(
                    code=code,
                    share_name=correct_share_name,
                    industry_id=industry.id,
                    sub_industry_id=sub_industry.id,
                    rsi_comparison_market=rsi_market,
                    rsi_comparison_sector=rsi_sector,
                    commodity=commodity
                )
                session.add(stock)

                # Commit to the database
                session.commit()
                
                print(f"Added stock {code}: {correct_share_name}")  # Debug print
                session.commit()

                # Add to the ticker_name table
                existing_ticker = session.query(TickerName).filter_by(ticker=code).first()
                if not existing_ticker:
                    ticker_name_entry = TickerName(ticker=code, name=correct_share_name)
                    session.add(ticker_name_entry)

                    # Commit both entries to the database
                    session.commit()

                # Reload the stock list and show a success message
                self.load_stocks()
                messagebox.showinfo("Stock Added", f"The stock '{correct_share_name}' with ticker '{code}' has been added successfully.")
                self.stock_code.set('')
                self.selected_industry.set('')
                self.selected_sub_industry.set('')  # Clear the sub-industry selection
                self.rsi_comparison_market.set('%5EJ300.JO')
                self.rsi_comparison_sector.set('')
                self.is_commodity.set(0)
                self.update_data()
            
            except Exception as e:
                session.rollback()  # Rollback in case of error
                print(f"Error: {str(e)}")  # Debug print
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
                self.load_stocks()
    
    def create_settings_tab(self):
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")

        # Settings form fields
        self.db_name = tk.StringVar(value=config['database'])
        self.db_user = tk.StringVar(value=config['user'])
        self.db_password = tk.StringVar(value=config['password'])
        self.db_host = tk.StringVar(value=config['host'])
        self.db_port = tk.StringVar(value=config['port'])

        ttk.Label(settings_frame, text="Database Name:").pack(anchor="w", padx=10, pady=5)
        ttk.Entry(settings_frame, textvariable=self.db_name).pack(anchor="w", padx=10, pady=5)

        ttk.Label(settings_frame, text="Username:").pack(anchor="w", padx=10, pady=5)
        ttk.Entry(settings_frame, textvariable=self.db_user).pack(anchor="w", padx=10, pady=5)

        ttk.Label(settings_frame, text="Password:").pack(anchor="w", padx=10, pady=5)
        
        # Show/Hide Password
        self.show_password = tk.BooleanVar(value=False)
        self.password_entry = ttk.Entry(settings_frame, textvariable=self.db_password, show="*")
        self.password_entry.pack(anchor="w", padx=10, pady=5)
        
        self.toggle_password_button = ttk.Button(settings_frame, text="Show", command=self.toggle_password_visibility)
        self.toggle_password_button.pack(anchor="w", padx=10, pady=5)

        ttk.Label(settings_frame, text="Host:").pack(anchor="w", padx=10, pady=5)
        ttk.Entry(settings_frame, textvariable=self.db_host).pack(anchor="w", padx=10, pady=5)

        ttk.Label(settings_frame, text="Port:").pack(anchor="w", padx=10, pady=5)
        ttk.Entry(settings_frame, textvariable=self.db_port).pack(anchor="w", padx=10, pady=5)

        ttk.Button(settings_frame, text="Save Settings", command=self.save_settings).pack(anchor="w", padx=10, pady=10)
    
    def create_portfolio_tab(self):
        portfolio_tab = ttk.Frame(self.notebook)
        self.notebook.add(portfolio_tab, text="Portfolio")

        ttk.Label(portfolio_tab, text="Portfolio Name:").pack(anchor="w", padx=10, pady=5)
        self.portfolio_name_var = tk.StringVar()
        ttk.Entry(portfolio_tab, textvariable=self.portfolio_name_var).pack(anchor="w", padx=10, pady=5)

        ttk.Button(portfolio_tab, text="Create Portfolio", command=self.create_portfolio).pack(anchor="w", padx=10, pady=10)
        
        # Widget to show existing portfolios like in the stock management tab
        self.portfolio_tree = ttk.Treeview(portfolio_tab, columns=("Name",), show="headings", height=8)
        self.portfolio_tree.heading("Name", text="Portfolio Name")
        self.portfolio_tree.pack(anchor="w", padx=10, pady=5, fill="both", expand=True)
        
        # Populate the portfolio tree with current portfolios
        self.populate_portfolio_tree()

    def populate_portfolio_tree(self):
        # Clear previous entries
        for row in self.portfolio_tree.get_children():
            self.portfolio_tree.delete(row)
        
        # Assume get_all_portfolios returns a list of portfolio objects with a 'name' attribute
        portfolios = self.get_all_portfolios()  
        for portfolio in portfolios:
            self.portfolio_tree.insert("", "end", values=(portfolio.name,))
    
    def get_all_portfolios(self):
        # For now, return a list of dummy portfolio objects.
        # Replace this with your actual portfolio retrieval logic.
        portfolios = session.query(Portfolio).all()
        return portfolios

    def create_portfolio(self):
        name = self.portfolio_name_var.get().strip()
        if not name:
            messagebox.showerror("Error", "Enter a valid portfolio name.")
            return
        
        new_portfolio = Portfolio(name=name)
        try:
            session.add(new_portfolio)
            session.commit()

            view_name = name.lower().replace(" ", "_")

            session.execute(f"""
            CREATE OR REPLACE VIEW public.{view_name}_portfolio
                AS
                SELECT adj_runs.code,
                    stocks.share_name,
                    adj_runs.current_price,
                    portfolio_stocks.shares,
                    portfolio_stocks.shares * adj_runs.current_price::double precision AS share_total_value,
                    adj_runs.z_score,
                    adj_runs.sector_rsi_1m,
                    adj_runs.sector_rsi_3m,
                    adj_runs.sector_rsi_6m,
                    adj_runs.market_rsi_1m,
                    adj_runs.market_rsi_3m,
                    adj_runs.market_rsi_6m,
                    adj_runs.overbought_oversold_value,
                    adj_runs.run_date,
                    adj_runs.industry,
                    adj_runs.sub_industry,
                    stocks.rsi_comparison_sector,
                    stocks.rsi_comparison_market,
                    portfolios.name AS portfolio_name,
                    adj_runs.next_week_prediction,
                    adj_runs.next_month_prediction
                FROM adj_runs
                    JOIN stocks ON stocks.code::text = adj_runs.code::text
                    JOIN portfolio_stocks ON portfolio_stocks.stock_id = stocks.id
                    JOIN portfolios ON portfolio_stocks.portfolio_id = portfolios.id
                WHERE adj_runs.run_date = CURRENT_DATE AND portfolios.name::text = '{name}'::text
                GROUP BY portfolios.name, portfolio_stocks.shares, stocks.code, adj_runs.code, stocks.share_name, adj_runs.industry, adj_runs.sub_industry, stocks.rsi_comparison_sector, stocks.rsi_comparison_market, adj_runs.current_price, adj_runs.next_week_prediction, adj_runs.next_month_prediction, adj_runs.z_score, adj_runs.overbought_oversold_value, adj_runs.sector_rsi_1m, adj_runs.sector_rsi_3m, adj_runs.sector_rsi_6m, adj_runs.market_rsi_1m, adj_runs.market_rsi_3m, adj_runs.market_rsi_6m, adj_runs.run_date, adj_runs.ma24, adj_runs.ma55
                ORDER BY adj_runs.run_date, portfolios.name, stocks.code;

            ALTER TABLE public.{view_name}_portfolio
                OWNER TO postgres;
            """)

            session.commit()

            messagebox.showinfo("Success", f"Portfolio '{name}' created.")
            self.populate_portfolio_tree()

        except Exception as e:
            self.session.rollback()
            messagebox.showerror("Error", f"Error creating portfolio: {e}")

    def save_settings(self):
        # Update the config dictionary with the new values
        config['database'] = self.db_name.get()
        config['user'] = self.db_user.get()
        config['password'] = self.db_password.get()
        config['host'] = self.db_host.get()
        config['port'] = self.db_port.get()

        # Save the updated config to the JSON file
        save_db_config(config)

        messagebox.showinfo("Settings Saved", "Database configuration has been updated.")
    
    def toggle_password_visibility(self):
        if self.show_password.get():
            self.password_entry.config(show="*")
            self.toggle_password_button.config(text="Show")
            self.show_password.set(False)
        else:
            self.password_entry.config(show="")
            self.toggle_password_button.config(text="Hide")
            self.show_password.set(True)

    def show_temp_message(self, title, message, duration=4000):
        # Create a top-level window that mimics a messagebox
        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.geometry("450x100")  # Size similar to a standard messagebox
        
        #popup.iconbitmap('stocks.ico')
        popup.resizable(False, False)  # Prevent resizing

        # Add an icon to mimic the messagebox appearance
        icon_label = ttk.Label(popup, image='::tk::icons::information')
        icon_label.grid(row=0, column=0, padx=10, pady=10)

        # Add the message text
        label = ttk.Label(popup, text=message, padding=10)
        label.grid(row=0, column=1, padx=10, pady=10)

        # Center the popup on the screen
        popup.update_idletasks()
        x = (popup.winfo_screenwidth() - popup.winfo_reqwidth()) // 2
        y = (popup.winfo_screenheight() - popup.winfo_reqheight()) // 2
        popup.geometry(f"+{x}+{y}")

        # Close the popup after the specified duration
        popup.after(duration, popup.destroy)

        # Ensure the popup stays on top and behaves like a modal dialog
        popup.transient(self.root)
        popup.grab_set()
        self.root.wait_window(popup)

    def update_industries_periodically(self):
        self.update_data()
        self.root.after(180000, self.update_industries_periodically)

    def update_data(self):
        self.load_industries()
        self.load_industries_dropdown_for_sub()
        self.load_rsi_comparison_sectors()
        self.load_stocks()
        self.load_sub_industries()

    def load_industries(self):
        self.industry_listbox.delete(0, tk.END)
        industries = session.query(Industry).order_by(Industry.name).all()  # Order by name
        for industry in industries:
            self.industry_listbox.insert(tk.END, industry.name)

    def load_industries_dropdown(self):
        industries = session.query(Industry).order_by(Industry.name).all()  # Order by name
        self.industry_dropdown['values'] = [industry.name for industry in industries]

    def load_industries_dropdown_for_sub(self):
        industries = session.query(Industry).order_by(Industry.name).all()  # Order by name
        self.industry_dropdown_for_sub['values'] = [industry.name for industry in industries]

    def load_sub_industries(self):
        self.sub_industry_listbox.delete(0, tk.END)
        sub_industries = session.query(SubIndustry).all()
        for sub_industry in sub_industries:
            industry = session.query(Industry).filter_by(id=sub_industry.industry_id).first()
            self.sub_industry_listbox.insert(tk.END, f"{sub_industry.name} (Industry: {industry.name})")

    def load_sub_industries_dropdown(self, event=None):
        selected_industry_name = self.selected_industry.get()
        industry = session.query(Industry).filter_by(name=selected_industry_name).first()
        if industry:
            sub_industries = session.query(SubIndustry).filter_by(industry_id=industry.id).all()
            self.sub_industry_dropdown['values'] = [sub.name for sub in sub_industries]

    def load_stocks(self):
        self.stock_listbox.delete(0, tk.END)
        
        # Load non-commodities (stocks) first, ordered by code
        non_commodities = session.query(Stock).filter_by(commodity=False).order_by(Stock.code).all()
        for stock in non_commodities:
            industry = session.query(Industry).filter_by(id=stock.industry_id).first()
            sub_industry = session.query(SubIndustry).filter_by(id=stock.sub_industry_id).first()
            self.stock_listbox.insert(tk.END, f"{stock.code} - {stock.share_name} (Industry: {industry.name}, Sub-Industry: {sub_industry.name})")
        
        # Load commodities next, ordered by code
        commodities = session.query(Stock).filter_by(commodity=True).order_by(Stock.code).all()
        for stock in commodities:
            industry = session.query(Industry).filter_by(id=stock.industry_id).first()
            sub_industry = session.query(SubIndustry).filter_by(id=stock.sub_industry_id).first()
            self.stock_listbox.insert(tk.END, f"{stock.code} - {stock.share_name} (Industry: {industry.name}, Sub-Industry: {sub_industry.name})")
        self.stocks = non_commodities + commodities

    # Popup to edit industry
    def edit_industry_popup(self, event):
        selected_index = self.industry_listbox.curselection()
        if selected_index:
            selected_name = self.industry_listbox.get(selected_index)
            industry = session.query(Industry).filter_by(name=selected_name).first()
            if industry:
                self.show_edit_popup("Edit Industry", industry.name, lambda new_name: self.update_industry(industry, new_name))

    # Popup to edit sub-industry
    def edit_sub_industry_popup(self, event):
        selected_index = self.sub_industry_listbox.curselection()
        if selected_index:
            selected_name = self.sub_industry_listbox.get(selected_index).split(" (")[0]  # Extract just the sub-industry name
            sub_industry = session.query(SubIndustry).filter_by(name=selected_name).first()
            if sub_industry:
                self.show_edit_popup("Edit Sub-Industry", sub_industry.name, lambda new_name: self.update_sub_industry(sub_industry, new_name))

    # Popup to edit stock
    def edit_stock_popup(self, event):
        selected_index = self.stock_listbox.curselection()
        if selected_index:
            selected_name = self.stock_listbox.get(selected_index).split(" - ")[0]  # Extract just the stock code
            stock = session.query(Stock).filter_by(code=selected_name).first()
            if stock:
                edit_popup = tk.Toplevel(self.root)
                edit_popup.title("Edit Stock")
                edit_popup.geometry("800x600")
                
                # Stock Name
                tk.Label(edit_popup, text="Stock Name:").pack(pady=10)
                new_name_var = tk.StringVar(value=stock.share_name)
                tk.Entry(edit_popup, textvariable=new_name_var, width=50).pack(pady=10)

                # RSI Comparison Market
                tk.Label(edit_popup, text="RSI Comparison Market:").pack(pady=10)
                new_market_var = tk.StringVar(value=stock.rsi_comparison_market)
                market_dropdown = ttk.Combobox(edit_popup, textvariable=new_market_var, width=48)
                markets = session.query(RSIComparisonMarket).order_by(RSIComparisonMarket.name).all()
                market_dropdown['values'] = [market.name for market in markets]
                market_dropdown.pack(pady=10)

                # RSI Comparison Sector
                tk.Label(edit_popup, text="RSI Comparison Sector:").pack(pady=10)
                new_sector_var = tk.StringVar(value=stock.rsi_comparison_sector)
                sector_dropdown = ttk.Combobox(edit_popup, textvariable=new_sector_var, width=48)
                sectors = session.query(RSIComparisonSector).order_by(RSIComparisonSector.name).all()
                sector_dropdown['values'] = [sector.name for sector in sectors]
                sector_dropdown.pack(pady=10)

                 # Display portfolio checkboxes
                ttk.Label(edit_popup, text="Portfolios:").pack(anchor="w", padx=10, pady=5)
                portfolios = session.query(Portfolio).order_by(Portfolio.name).all()
                portfolio_vars = {}
                for p in portfolios:
                    var = tk.IntVar(value=1 if p in stock.portfolios else 0)
                    portfolio_vars[p] = var
                    ttk.Checkbutton(edit_popup, text=p.name, variable=var).pack(anchor="w", padx=20, pady=2)

                # Save Changes Button
                def save_stock_changes():
                    # Update stock fields
                    stock.share_name = new_name_var.get().strip()

                    # Update portfolio associations
                    new_portfolios = [p for p, var in portfolio_vars.items() if var.get() == 1]
                    stock.portfolios = new_portfolios

                    try:
                        session.commit()
                        messagebox.showinfo("Success", "Stock updated with portfolios.")
                        edit_popup.destroy()
                        #self.load_stocks()  # Refresh stocks list if necessary
                    except Exception as e:
                        session.rollback()
                        messagebox.showerror("Error", f"Failed to update stock: {e}")

                tk.Button(edit_popup, text="Save", command=save_stock_changes).pack(pady=10)

                edit_popup.transient(self.root)
                edit_popup.grab_set()
                self.root.wait_window(edit_popup)

    # General popup for editing
    def show_edit_popup(self, title, current_name, on_save):
        edit_popup = tk.Toplevel(self.root)
        edit_popup.title(title)
        edit_popup.geometry("600x150")
        tk.Label(edit_popup, text="New Name:").pack(pady=10)
        new_name_var = tk.StringVar(value=current_name)
        tk.Entry(edit_popup, textvariable=new_name_var, width=50).pack(pady=10)

        def save_changes():
            new_name = new_name_var.get().strip()
            if new_name:
                on_save(new_name)
                edit_popup.destroy()
            else:
                messagebox.showwarning("Input Error", "Name cannot be empty.")

        tk.Button(edit_popup, text="Save", command=save_changes).pack(pady=10)

    # Update methods
    def update_industry(self, industry, new_name):
        industry.name = new_name
        session.commit()
        self.load_industries()
        self.load_industries_dropdown_for_sub()
        self.load_stocks()

    def update_sub_industry(self, sub_industry, new_name):
        sub_industry.name = new_name
        session.commit()
        self.load_sub_industries()
        self.load_stocks()

    def update_stock(self, stock, new_name):
        stock.share_name = new_name
        session.commit()
        self.load_stocks()

    def load_rsi_comparison_sectors(self):
        sectors = session.query(RSIComparisonSector).order_by(RSIComparisonSector.name).all()
        
        # Update dropdown for RSI sectors in stock management tab
        self.rsi_sector_dropdown['values'] = [sector.name for sector in sectors]
        
        # Update the listbox in the RSI management tab
        self.rsi_sector_listbox.delete(0, tk.END)  # Clear existing items
        for sector in sectors:
            self.rsi_sector_listbox.insert(tk.END, sector.name)

    def create_rsi_comparison_sectors_tab(self):
        rsi_sector_frame = ttk.Frame(self.notebook)
        self.notebook.add(rsi_sector_frame, text="Manage RSI Comparison Sectors")

        # RSI Sectors List
        self.rsi_sector_listbox = tk.Listbox(rsi_sector_frame, width=40)
        self.rsi_sector_listbox.pack(side="left", fill="y", padx=10, pady=10)

        # Add New RSI Sector
        add_rsi_sector_frame = ttk.Frame(rsi_sector_frame)
        add_rsi_sector_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.new_rsi_sector_name = tk.StringVar()
        ttk.Label(add_rsi_sector_frame, text="New RSI Comparison Sector:").pack(anchor="w")
        ttk.Entry(add_rsi_sector_frame, textvariable=self.new_rsi_sector_name).pack(anchor="w")

        ttk.Button(add_rsi_sector_frame, text="Add Sector", command=self.add_rsi_sector).pack(anchor="w")

        ttk.Button(add_rsi_sector_frame, text="Remove Selected Sector", command=self.remove_selected_rsi_sector).pack(anchor="w")

        # Load sectors into both the dropdown and listbox

    def add_rsi_sector(self):
        name = self.new_rsi_sector_name.get().strip()
        if name:
            # Validate RSI Sector Ticker
            sector_data = yf.Ticker(name)
            if 'shortName' not in sector_data.info or sector_data.info['longBusinessSummary'] is None:
                messagebox.showerror("Invalid Sector Ticker", "The RSI Comparison Sector code entered does not exist on Yahoo Finance.")
                return

            sector = RSIComparisonSector(name=name)
            try:
                session.add(sector)
                session.commit()
                self.load_rsi_comparison_sectors()  # Reload the sectors list
                self.new_rsi_sector_name.set('')  # Clear the input field
                self.show_temp_message(title='Success', message=f"{name} has been added!", duration=1000)
            except IntegrityError:
                session.rollback()
                messagebox.showerror("Duplicate Sector", f"The sector '{name}' already exists.")
        else:
            messagebox.showwarning("Input Error", "RSI Comparison Sector name cannot be empty.")

    def remove_selected_rsi_sector(self):
        selected_index = self.rsi_sector_listbox.curselection()
        if selected_index:
            selected_name = self.rsi_sector_listbox.get(selected_index)
            sector = session.query(RSIComparisonSector).filter_by(name=selected_name).first()
            if sector:
                session.delete(sector)
                session.commit()
                self.load_rsi_comparison_sectors()

    def create_rsi_comparison_markets_tab(self):
        rsi_market_frame = ttk.Frame(self.notebook)
        self.notebook.add(rsi_market_frame, text="Manage RSI Comparison Markets")

        # RSI Markets List
        self.rsi_market_listbox = tk.Listbox(rsi_market_frame, width=40)
        self.rsi_market_listbox.pack(side="left", fill="y", padx=10, pady=10)

        # Add New RSI Market
        add_rsi_market_frame = ttk.Frame(rsi_market_frame)
        add_rsi_market_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.new_rsi_market_name = tk.StringVar()
        ttk.Label(add_rsi_market_frame, text="New RSI Comparison Market:").pack(anchor="w")
        ttk.Entry(add_rsi_market_frame, textvariable=self.new_rsi_market_name).pack(anchor="w")

        ttk.Button(add_rsi_market_frame, text="Add Market", command=self.add_rsi_market).pack(anchor="w")
        ttk.Button(add_rsi_market_frame, text="Remove Selected Market", command=self.remove_selected_rsi_market).pack(anchor="w")

    def add_rsi_market(self):
        name = self.new_rsi_market_name.get().strip()
        if name:
            # Validate RSI Market Ticker
            market_data = yf.Ticker(name)
            print(market_data.info)
            if 'shortName' not in market_data.info or market_data.info['underlyingSymbol'] is None:
                messagebox.showerror("Invalid Market Ticker", "The RSI Comparison Market code entered does not exist on Yahoo Finance.")
                return

            market = RSIComparisonMarket(name=name)
            try:
                session.add(market)
                session.commit()
                self.load_rsi_comparison_markets()  # Reload the markets list
                self.new_rsi_market_name.set('')  # Clear the input field
                self.show_temp_message(title='Success', message=f"{name} has been added!", duration=1000)
                self.update_data()
            except IntegrityError:
                session.rollback()
                messagebox.showerror("Duplicate Market", f"The market '{name}' already exists.")
        else:
            messagebox.showwarning("Input Error", "RSI Comparison Market name cannot be empty.")

    def remove_selected_rsi_market(self):
        selected_index = self.rsi_market_listbox.curselection()
        if selected_index:
            selected_name = self.rsi_market_listbox.get(selected_index)
            market = session.query(RSIComparisonMarket).filter_by(name=selected_name).first()
            if market:
                session.delete(market)
                session.commit()
                self.load_rsi_comparison_markets()

    def load_rsi_comparison_markets(self):
        markets = session.query(RSIComparisonMarket).order_by(RSIComparisonMarket.name).all()
        
        # Update dropdown for RSI markets in stock management tab
        self.rsi_market_dropdown['values'] = [market.name for market in markets]  # Use rsi_market_dropdown
        
        # Update the listbox in the RSI market management tab
        self.rsi_market_listbox.delete(0, tk.END)  # Clear existing items
        for market in markets:
            self.rsi_market_listbox.insert(tk.END, market.name)

    def edit_rsi_market_popup(self, event):
        selected_index = self.rsi_market_listbox.curselection()
        if selected_index:
            selected_name = self.rsi_market_listbox.get(selected_index)
            market = session.query(RSIComparisonMarket).filter_by(name=selected_name).first()
            if market:
                def on_save(new_name):
                    # Validate the market ticker
                    market_data = yf.Ticker(new_name)
                    if 'shortName' not in market_data.info or market_data.info['longBusinessSummary'] is None:
                        messagebox.showerror("Invalid Market Ticker", "The RSI Comparison Market code entered does not exist on Yahoo Finance.")
                        return

                    market.name = new_name
                    session.commit()
                    self.load_rsi_comparison_markets()

                self.show_edit_popup("Edit Market", market.name, on_save)
    
    def edit_rsi_sector_popup(self, event):
        selected_index = self.rsi_sector_listbox.curselection()
        if selected_index:
            selected_name = self.rsi_sector_listbox.get(selected_index)
            sector = session.query(RSIComparisonSector).filter_by(name=selected_name).first()
            if sector:
                def on_save(new_name):
                    # Validate the sector ticker
                    sector_data = yf.Ticker(new_name)
                    if 'shortName' not in sector_data.info or sector_data.info['longBusinessSummary'] is None:
                        messagebox.showerror("Invalid Sector Ticker", "The RSI Comparison Sector code entered does not exist on Yahoo Finance.")
                        return

                    sector.name = new_name
                    session.commit()
                    self.load_rsi_comparison_sectors()

                self.show_edit_popup("Edit Sector", sector.name, on_save)

    def create_excel_upload_tab(self):
        excel_frame = ttk.Frame(self.notebook)
        self.notebook.add(excel_frame, text="Upload Excel Data")

        # Button to open file dialog and select the Excel file
        ttk.Button(excel_frame, text="Select Excel File", command=self.select_excel_file).pack(pady=20)

        # Label to show the selected file path
        self.file_path_label = ttk.Label(excel_frame, text="")
        self.file_path_label.pack(pady=10)

        # Button to upload data to the database
        ttk.Button(excel_frame, text="Upload Data", command=self.upload_excel_data).pack(pady=20)
    
    def select_excel_file(self):
        # Open file dialog to select the Excel file
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xls *.xlsx")])
        if file_path:
            self.file_path = file_path
            self.file_path_label.config(text=f"Selected: {file_path}")
        else:
            self.file_path = None
            self.file_path_label.config(text="No file selected")

    def upload_excel_data(self):
        if not self.file_path:
            messagebox.showerror("Error", "No file selected. Please select an Excel file.")
            return
        try:
            # Read the specific sheet from the Excel file
            df = pd.read_excel(self.file_path, sheet_name="Summary")  # Replace "Sheet Name" with your actual sheet name

            # Replace NA-like strings with None
            df.replace(['NA', 'N/A', '#DIV/0!', '#N/A'], None, inplace=True)

            # Round numerical columns to two decimal places where applicable
            df = df.apply(lambda x: round(x, 2) if x.dtype == 'float' else x)

            # Function to convert values to date type, ensuring compatibility and handling NA/invalid data
            def convert_to_date(value):
                if pd.isnull(value) or value in ['NA', 'N/A', '#DIV/0!', '#N/A']:
                    return None
                try:
                    if isinstance(value, datetime):
                        return value.date()  # Extract date part if it's a datetime
                    elif isinstance(value, date):
                        return value  # Already the correct type
                    elif isinstance(value, str):
                        return pd.to_datetime(value, errors='coerce').date()  # Convert string to date
                except Exception as e:
                    print(f"Date conversion error: {e}, value: {value}")
                return None

            # Apply conversion to relevant date columns explicitly casting to date
            date_columns = ['DIV_LDT', 'DIV_Pay', 'YE_release', 'Int_release']
            for col in date_columns:
                df[col] = df[col].apply(convert_to_date)

            # Function to handle truncation of strings to match column length constraints
            def truncate_string(value, max_length):
                if value and isinstance(value, str) and len(value) > max_length:
                    return value[:max_length]
                return value

            # Iterate over DataFrame rows and insert data into the database
            for index, row in df.iterrows():
                try:
                    vi_data = ViData(
                        code=truncate_string(row['Code'], 10),  # Assuming 'Code' has a limit of 10 characters
                        eps=row.get('EPS', None),
                        nav=row.get('NAV', None),
                        sales=row.get('Sales', None),
                        eps_growth_f=truncate_string(row.get('EPS_Growth_F', ''), 10),  # Adjust the length as needed
                        roe_f=row.get('ROE_F', None),
                        inst_profit_margin_f=row.get('Net_Profit_Margin_F', None),
                        sales_growth_f=truncate_string(row.get('Sales_Growth_F', ''), 10),  # Adjust the length as needed
                        holding=truncate_string(row.get('Holding', ''), 1),  # Assuming Holding is a single character
                        shares=row.get('Shares', None),
                        interest_cover=row.get('Interest_Cover', None),
                        comment=truncate_string(row.get('Comment', ''), 50),  # Assuming 'Comment' has a 50-character limit
                        tnav=row.get('TNAV', None),
                        rote=row.get('ROTE', None),
                        actual_roe=row.get('Actual_ROE', None),
                        last_update=row.get('Last_update', None),
                        o_margin=row.get('O_Margin', None),
                        div=row.get('Dividend', None),
                        cash_ps=row.get('Cash_ps', None),
                        act=row.get('ACT_HEPS', None),  # Assuming ACT_HEPS corresponds to the correct column
                        heps=row.get('ACT_HEPS', None),  # Assuming ACT_HEPS corresponds to the correct column
                        quality_rating=truncate_string(row.get('Quality_Rating', ''), 10),  # Adjust the length as needed
                        div_decl=row.get('Div_Decl', None),
                        div_ldt=row.get('DIV_LDT', None),  # Should now be the correct date type
                        div_pay=row.get('DIV_Pay', None),  # Should now be the correct date type
                        rec=truncate_string(row.get('REC', ''), 20),  # Assuming 'REC' has a 20-character limit
                        rec_on=truncate_string(row.get('REC_on', ''), 20),  # Adjust the length as needed
                        ye_release=row.get('YE_release', None),  # Should now be the correct date type
                        int_release=row.get('Int_release', None),  # Should now be the correct date type
                        rec_price=row.get('Rec_Price', None),
                        share_price=row.get('Share Price', None),
                        peg=row.get('PEG', None),
                        peg_pe=row.get('PEG_PE', None),
                        peg_pe_value=row.get('PEG_PE_value', None),
                        peg_nav=truncate_string(row.get('PEG_NAV', ''), 10),  # Adjust the length as needed
                        peg_pe_nav_value=row.get('PEG_PE_value', None),
                        run_date=date.today()
                    )
                    session.add(vi_data)
                except Exception as row_error:
                    print(f"Error processing row {index}: {row_error}")

            session.commit()
            messagebox.showinfo("Success", "Data uploaded successfully!")

        except SQLAlchemyError as e:
            session.rollback()
            e = str(e)
            e_to_show = e.split('\n')[1] + "\n" + e.split('\n')[-1]
            messagebox.showerror("Upload Error", f"An error occurred while uploading data: {e_to_show}")
        except Exception as e:
            session.rollback()
            messagebox.showerror("Upload Error", f"An error occurred: {e}")

    def edit_portfolio_popup(self, event):
        selected = self.portfolio_tree.selection()
        if not selected:
            return

        # Get the portfolio name from the selected row
        portfolio_name = self.portfolio_tree.item(selected[0])['values'][0]
        portfolio = session.query(Portfolio).filter_by(name=portfolio_name).first()
        if not portfolio:
            messagebox.showerror("Error", "Portfolio not found.")
            return

        # Create popup window
        popup = tk.Toplevel(self.root)
        popup.title(f"Edit Portfolio: {portfolio_name}")
        popup.geometry("1200x800")

        # Create Treeview to list stocks linked to the portfolio with their share counts
        tree = ttk.Treeview(popup, columns=("Code", "Name", "Shares"), show="headings")
        tree.heading("Code", text="Stock Code")
        tree.heading("Name", text="Stock Name")
        tree.heading("Shares", text="Shares")
        tree.column("Shares", width=100)
        tree.pack(fill="both", expand=True, padx=10, pady=10)

        # Query join records for this portfolio
        portfolio_stocks = (
            session.query(PortfolioStock, Stock)
            .join(Stock, PortfolioStock.stock_id == Stock.id)
            .filter(PortfolioStock.portfolio_id == portfolio.id)
            .order_by(Stock.share_name)
            .all()
        )
        for ps, stock in portfolio_stocks:
            tree.insert("", "end", values=(stock.code, stock.share_name, ps.shares))

        # Allow editing the "Shares" field on double-click
        def on_double_click(event):
            region = tree.identify("region", event.x, event.y)
            if region != "cell":
                return
            col = tree.identify_column(event.x)
            # Only allow editing the Shares column (#3)
            if col != "#3":
                return

            item = tree.selection()[0]
            x, y, width, height = tree.bbox(item, col)
            current_val = tree.item(item, "values")[2]

            entry = tk.Entry(popup)
            entry.place(x=x, y=y, width=width, height=height)
            entry.insert(0, current_val)
            entry.focus()

            def save_edit(event):
                tree.set(item, column="Shares", value=entry.get())
                entry.destroy()

            entry.bind("<Return>", save_edit)
            entry.bind("<FocusOut>", lambda e: entry.destroy())

        tree.bind("<Double-1>", on_double_click)

        def save_changes():
            # Update each PortfolioStock record with the new shares value
            for item in tree.get_children():
                code, name, shares = tree.item(item, "values")
                try:
                    new_shares = float(shares)
                except ValueError:
                    messagebox.showerror("Error", f"Invalid share number: {shares}")
                    return

                # Retrieve the PortfolioStock record based on portfolio id and stock id
                stock = session.query(Stock).filter_by(code=code).first()
                if stock:
                    ps = session.query(PortfolioStock).filter_by(portfolio_id=portfolio.id, stock_id=stock.id).first()
                    if ps:
                        ps.shares = new_shares

            try:
                session.commit()
                messagebox.showinfo("Success", "Portfolio updated successfully.")
                popup.destroy()
            except Exception as e:
                session.rollback()
                messagebox.showerror("Error", f"Error updating portfolio: {e}")

        save_btn = ttk.Button(popup, text="Save Changes", command=save_changes)
        save_btn.pack(pady=10)

        popup.transient(self.root)
        popup.grab_set()
        self.root.wait_window(popup)

if __name__ == "__main__":
    root = tk.Tk()
    app = InvestmentManagerApp(root)
    root.mainloop()
