import json
import yfinance as yf
import tkinter as tk
from tkinter import ttk, messagebox
from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import PendingRollbackError, IntegrityError

# Load Database Configuration from JSON
def load_db_config():
    with open('db_config.json', 'r') as config_file:
        return json.load(config_file)

def save_db_config(config):
    with open('db_config.json', 'w') as config_file:
        json.dump(config, config_file, indent=4)

config = load_db_config()

# Create the Database Engine Using Configurations
engine = create_engine(f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}")
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

class Industry(Base):
    __tablename__ = 'industries'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)

    sub_industries = relationship("SubIndustry", back_populates="industry", cascade="all, delete")

class SubIndustry(Base):
    __tablename__ = 'sub_industries'
    id = Column(Integer, primary_key=True)
    industry_id = Column(Integer, ForeignKey('industries.id'), nullable=False)
    name = Column(String, unique=True, nullable=False)

    industry = relationship("Industry", back_populates="sub_industries")
    stocks = relationship("Stock", back_populates="sub_industry", cascade="all, delete")

class Stock(Base):
    __tablename__ = 'stocks'
    id = Column(Integer, primary_key=True)
    code = Column(String, nullable=False)
    share_name = Column(String, nullable=False)
    industry_id = Column(Integer, ForeignKey('industries.id'), nullable=False)
    sub_industry_id = Column(Integer, ForeignKey('sub_industries.id'), nullable=False)
    rsi_comparison_market = Column(String)
    rsi_comparison_sector = Column(String)
    commodity = Column(Boolean, default=False)

    industry = relationship("Industry", back_populates="stocks")
    sub_industry = relationship("SubIndustry", back_populates="stocks")

class TickerName(Base):
    __tablename__ = 'ticker_name'
    ticker = Column(String, primary_key=True)  # Use ticker as the primary key
    name = Column(String, nullable=False)

class RSIComparisonSector(Base):
    __tablename__ = 'rsi_comparison_sectors'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)

class RSIComparisonMarket(Base):
    __tablename__ = 'rsi_comparison_markets'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)

Industry.stocks = relationship("Stock", order_by=Stock.id, back_populates="industry")

# Create tables if not exists
Base.metadata.create_all(engine)

class InvestmentManagerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Investment Universe Manager")
        self.root.geometry("1200x500")  # Set the window size

        # Set the window icon
        self.root.iconbitmap('stocks.ico')

        # Create Notebook for Tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        # Create tabs
        self.create_rsi_comparison_markets_tab()  # Call this before the stocks management tab
        self.create_rsi_comparison_sectors_tab()  # Call this before the stocks management tab
        self.create_industry_management_tab()
        self.create_sub_industry_management_tab()  # New tab for sub-industries
        self.create_stocks_management_tab()
        self.create_settings_tab()

        # Start the periodic update of industries
        self.update_industries_periodically()
        self.industry_listbox.bind("<Double-1>", self.edit_industry_popup)
        self.sub_industry_listbox.bind("<Double-1>", self.edit_sub_industry_popup)
        self.stock_listbox.bind("<Double-1>", self.edit_stock_popup)

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
                """existing_ticker = session.query(TickerName).filter_by(ticker=code).first()
                if not existing_ticker:
                    ticker_name_entry = TickerName(ticker=code, name=correct_share_name)
                    session.add(ticker_name_entry)

                    # Commit both entries to the database
                    session.commit()"""

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
    
    def load_stocks(self):
        self.stock_listbox.delete(0, tk.END)
        stocks = session.query(Stock).all()
        for stock in stocks:
            industry = session.query(Industry).filter_by(id=stock.industry_id).first()
            sub_industry = session.query(SubIndustry).filter_by(id=stock.sub_industry_id).first()
            self.stock_listbox.insert(tk.END, f"{stock.code} - {stock.share_name} (Industry: {industry.name}, Sub-Industry: {sub_industry.name})")

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
        
        popup.iconbitmap('stocks.ico')
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
        self.root.after(30000, self.update_industries_periodically)

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
                edit_popup.geometry("600x400")
                
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

                # Save Changes Button
                def save_changes():
                    new_name = new_name_var.get().strip()
                    new_market = new_market_var.get().strip()
                    new_sector = new_sector_var.get().strip()

                    if new_name and new_market and new_sector:
                        stock.share_name = new_name
                        stock.rsi_comparison_market = new_market
                        stock.rsi_comparison_sector = new_sector
                        session.commit()
                        self.load_stocks()
                        edit_popup.destroy()
                    else:
                        messagebox.showwarning("Input Error", "All fields must be filled out.")

                tk.Button(edit_popup, text="Save", command=save_changes).pack(pady=10)

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
            if 'shortName' not in market_data.info or market_data.info['longBusinessSummary'] is None:
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


if __name__ == "__main__":
    root = tk.Tk()
    app = InvestmentManagerApp(root)
    root.mainloop()
