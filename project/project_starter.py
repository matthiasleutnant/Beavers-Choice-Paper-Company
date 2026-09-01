import pandas as pd
import numpy as np
import os
import time
import asyncio
import dotenv
import ast
import json
import logging
import argparse
import re
from dataclasses import dataclass
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Optional, Union
from sqlalchemy import create_engine, Engine
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from tavily import TavilyClient

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "multi_agent.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("munder_difflin")


def _log_event(event_name: str, **details: object) -> None:
    """Write a compact structured event without exposing credentials."""
    safe_details = {}
    for key, value in details.items():
        rendered = json.dumps(value, default=str, ensure_ascii=True)
        safe_details[key] = rendered if len(rendered) <= 1200 else rendered[:1200] + "...[truncated]"
    logger.info("%s %s", event_name, " ".join(f"{key}={value}" for key, value in safe_details.items()))


def _to_builtin(value):
    """Convert NumPy and pandas scalar values into JSON-serializable values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create the transactions table with an explicit SQLite schema
        # ----------------------------
        with db_engine.begin() as connection:
            connection.execute(text("DROP TABLE IF EXISTS transactions"))
            connection.execute(
                text(
                    """
                    CREATE TABLE transactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        item_name TEXT,
                        transaction_type TEXT NOT NULL
                            CHECK (transaction_type IN ('stock_orders', 'sales')),
                        units INTEGER,
                        price REAL NOT NULL,
                        transaction_date TEXT NOT NULL
                    )
                    """
                )
            )

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


@dataclass
class AgentDependencies:
    """Shared runtime dependencies for all agents."""

    engine: Engine


class InventoryLine(BaseModel):
    item_name: str
    requested_units: int = Field(gt=0)
    available_units: int = Field(ge=0)
    status: str


class RequestedLine(BaseModel):
    item_name: str
    units: int = Field(gt=0)


class CustomerRequestPayload(BaseModel):
    request_date: str
    customer_request: str
    customer_context: str = ""
    negotiation_points: List[str] = Field(default_factory=list)


class CustomerContextResult(BaseModel):
    request_date: str
    customer_request: str
    customer_context: str
    negotiation_points: List[str] = Field(default_factory=list)
    decision: Literal["accept", "decline", "revised_request"] = "accept"
    revised_request: Optional[str] = None
    negotiation_round: int = Field(default=0, ge=0, le=1)


class InventoryResult(BaseModel):
    request_date: str
    items: List[InventoryLine]
    replenishment_needed: bool
    summary: str


class QuoteLine(BaseModel):
    item_name: str
    units: int = Field(gt=0)
    unit_price: float = Field(ge=0)
    line_total: float = Field(ge=0)


class QuoteResult(BaseModel):
    request_date: str
    items: List[QuoteLine]
    subtotal: float = Field(ge=0)
    discount: float = Field(ge=0)
    total_amount: float = Field(ge=0)
    explanation: str


class OrderResult(BaseModel):
    request_date: str
    accepted: bool
    delivery_date: str
    transactions: List[int]
    message: str


class FinancialResult(BaseModel):
    as_of_date: str
    cash_balance: float
    inventory_value: float
    total_assets: float
    top_selling_products: List[Dict]


class WorkflowResult(BaseModel):
    response: str
    inventory_result: InventoryResult
    quote_result: QuoteResult
    order_result: OrderResult
    financial_result: FinancialResult
    customer_result: CustomerContextResult
    negotiation_attempts: List[Dict] = Field(default_factory=list)


def canonical_item_name(item_name: str) -> str:
    """Map common customer wording to one exact catalog item name."""
    value = re.sub(r"[^a-z0-9]+", " ", item_name.lower()).strip()
    value = " ".join(value.split())
    aliases = {
        "a4 glossy paper": "Glossy paper",
        "glossy a4 paper": "Glossy paper",
        "a3 glossy paper": "Glossy paper",
        "a4 matte paper": "Matte paper",
        "a3 matte paper": "Matte paper",
        "heavy cardstock": "Cardstock",
        "heavy cardstock white": "Cardstock",
        "recycled cardstock": "Cardstock",
        "colored cardstock": "Cardstock",
        "standard printer paper": "Standard copy paper",
        "printer paper": "Standard copy paper",
        "white printer paper": "Standard copy paper",
        "a4 printing paper": "A4 paper",
        "a4 size printer paper": "A4 paper",
        "a4 white paper": "A4 paper",
        "colored paper assorted colors": "Colored paper",
        "colorful construction paper": "Construction paper",
        "colorful poster paper": "Poster paper",
        "poster board": "Large poster paper (24x36 inches)",
        "decorative washi tape": "Decorative adhesive tape (washi tape)",
        "washi tape": "Decorative adhesive tape (washi tape)",
        "paper napkins": "Paper napkins",
        "table napkins": "Paper napkins",
    }
    if value in aliases:
        return aliases[value]
    exact = {item["item_name"].lower(): item["item_name"] for item in paper_supplies}
    if value in exact:
        return exact[value]
    if "recycled" in value and "cardstock" in value:
        return "Cardstock"
    if "cardstock" in value and ("colored" in value or "various colors" in value):
        return "Cardstock"
    if "glossy" in value and ("a4" in value or "a3" in value):
        return "Glossy paper"
    if "matte" in value and ("a4" in value or "a3" in value):
        return "Matte paper"
    if "construction" in value and "paper" in value:
        return "Construction paper"
    if "printer paper" in value and "a4" in value:
        return "A4 paper"
    return item_name


def _create_model() -> OpenAIChatModel:
    """Create the OpenAI-compatible model configured by the project environment."""
    dotenv.load_dotenv()
    api_key = os.getenv("UDACITY_OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("UDACITY_OPENAI_API_KEY is not configured")
    provider = OpenAIProvider(
        base_url=os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1"),
        api_key=api_key,
    )
    return OpenAIChatModel(
        os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        provider=provider,
    )


def inventory_tool(
    ctx: RunContext[AgentDependencies],
    request_date: str,
    items: List[RequestedLine],
) -> Dict:
    """Return dated inventory levels and replenishment decisions."""
    if isinstance(items, list):
        items = [item if isinstance(item, RequestedLine) else RequestedLine.model_validate(item) for item in items]
    _log_event("tool.start", tool="inventory", request_date=request_date, items=items)
    inventory = get_all_inventory(request_date)
    rows = []
    replenishment = False
    for requested in items:
        item_name = canonical_item_name(requested.item_name)
        requested_units = requested.units
        stock_level = get_stock_level(item_name, request_date)
        raw_available_units = (
            int(stock_level.iloc[0]["current_stock"])
            if not stock_level.empty
            else int(inventory.get(item_name, 0))
        )
        available_units = max(0, raw_available_units)
        if raw_available_units < 0:
            _log_event(
                "inventory.invariant_violation",
                item_name=item_name,
                request_date=request_date,
                calculated_stock=raw_available_units,
            )
        minimum = pd.read_sql(
            "SELECT min_stock_level FROM inventory WHERE item_name = :item_name",
            ctx.deps.engine,
            params={"item_name": item_name},
        )
        if minimum.empty:
            status = "unknown item"
        elif available_units < requested_units:
            status = "replenishment required"
            replenishment = True
        elif available_units <= int(minimum.iloc[0]["min_stock_level"]):
            status = "available; replenishment recommended"
            replenishment = True
        else:
            status = "available"
        rows.append(
            {
                "item_name": item_name,
                "requested_units": requested_units,
                "available_units": available_units,
                "status": status,
            }
        )
    response = InventoryResult(
        request_date=request_date,
        items=rows,
        replenishment_needed=replenishment,
        summary="; ".join(f"{row['item_name']}: {row['status']}" for row in rows),
    ).model_dump()
    _log_event("tool.end", tool="inventory", response=response)
    return response


def quote_history_tool(
    ctx: RunContext[AgentDependencies],
    search_terms: List[str],
) -> List[Dict]:
    """Find comparable historical quotes."""
    _log_event("tool.start", tool="quote_history", search_terms=search_terms)
    response = search_quote_history(search_terms)
    _log_event("tool.end", tool="quote_history", response=response)
    return response


def catalog_tool(ctx: RunContext[AgentDependencies]) -> List[Dict]:
    """Return the canonical item catalog and prices."""
    _log_event("tool.call", tool="catalog", item_count=len(paper_supplies))
    return paper_supplies


def supplier_research_tool(
    ctx: RunContext[AgentDependencies],
    item_name: str,
    quantity: int,
) -> Dict:
    """Research supplier options when replenishment is required."""
    _log_event("tool.start", tool="supplier_research", item_name=item_name, quantity=quantity)
    dotenv.load_dotenv()
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not configured")
    client = TavilyClient(api_key=api_key)
    result = client.search(
        query=f"supplier availability lead time price {quantity} units {item_name}",
        search_depth="advanced",
        max_results=3,
        include_answer=True,
    )
    response = {
        "item_name": item_name,
        "quantity": quantity,
        "answer": result.get("answer", ""),
        "sources": [
            {"title": item.get("title", ""), "url": item.get("url", "")}
            for item in result.get("results", [])
        ],
        "disclaimer": (
            "External research is advisory only. SQLite inventory and the "
            "internal delivery calculator remain authoritative."
        ),
    }
    _log_event("tool.end", tool="supplier_research", response=response)
    return response


def delivery_date_tool(
    ctx: RunContext[AgentDependencies],
    request_date: str,
    quantity: int,
) -> str:
    """Estimate supplier delivery using the starter helper."""
    _log_event("tool.start", tool="delivery_date", request_date=request_date, quantity=quantity)
    response = get_supplier_delivery_date(request_date, quantity)
    _log_event("tool.end", tool="delivery_date", response=response)
    return response


def transaction_tool(
    ctx: RunContext[AgentDependencies],
    item_name: str,
    transaction_type: Literal["stock_orders", "sales"],
    quantity: int,
    price: float,
    date: str,
) -> int:
    """Record one validated stock or sales transaction."""
    item_name = canonical_item_name(item_name)
    _log_event(
        "tool.start",
        tool="transaction",
        item_name=item_name,
        transaction_type=transaction_type,
        quantity=quantity,
        price=price,
        date=date,
    )
    if transaction_type == "sales":
        available = get_stock_level(item_name, date)
        available_units = int(available.iloc[0]["current_stock"]) if not available.empty else 0
        if available_units < quantity:
            raise ValueError(
                f"Cannot record sale for {item_name}: "
                f"available {available_units}, requested {quantity}"
            )
    response = create_transaction(item_name, transaction_type, quantity, price, date)
    _log_event("tool.end", tool="transaction", transaction_id=response)
    return response


def cash_balance_tool(
    ctx: RunContext[AgentDependencies],
    as_of_date: str,
) -> float:
    """Return the cash balance for a date."""
    _log_event("tool.start", tool="cash_balance", as_of_date=as_of_date)
    response = get_cash_balance(as_of_date)
    _log_event("tool.end", tool="cash_balance", response=response)
    return response


def financial_report_tool(
    ctx: RunContext[AgentDependencies],
    as_of_date: str,
) -> Dict:
    """Return the complete financial report for a date."""
    _log_event("tool.start", tool="financial_report", as_of_date=as_of_date)
    response = _to_builtin(generate_financial_report(as_of_date))
    _log_event("tool.end", tool="financial_report", response=response)
    return response


def _build_agents() -> Dict[str, Agent]:
    """Build the five internal agents and the external customer partner."""
    model = _create_model()
    inventory_agent = Agent(
        model,
        name="inventory-agent",
        deps_type=AgentDependencies,
        output_type=InventoryResult,
        system_prompt=(
            "You are the inventory agent. Input and output are JSON. Normalize item names to the exact "
            "inventory names and call inventory_tool for every requested item. "
            "If replenishment is required or recommended, call "
            "supplier_research_tool for the affected item."
        ),
        tools=[inventory_tool, supplier_research_tool],
    )
    quoting_agent = Agent(
        model,
        name="quoting-agent",
        deps_type=AgentDependencies,
        output_type=QuoteResult,
        system_prompt=(
            "You are the quoting agent. Input and output are JSON. Use quote_history_tool, calculate "
            "catalog_tool for canonical names and prices, and apply a transparent "
            "volume discount. Return valid monetary totals."
        ),
        tools=[quote_history_tool, catalog_tool],
    )
    ordering_agent = Agent(
        model,
        name="ordering-agent",
        deps_type=AgentDependencies,
        output_type=OrderResult,
        system_prompt=(
            "You are the ordering agent. Input and output are JSON. Confirm stock before accepting an "
            "order, call delivery_date_tool, and call transaction_tool for "
            "each accepted sales line. For transaction_tool, transaction_type must be exactly "
            "'sales' (never 'sale') or 'stock_orders' (never 'stock order'). "
            "Never invent transaction IDs."
        ),
        tools=[inventory_tool, delivery_date_tool, transaction_tool],
    )
    financial_agent = Agent(
        model,
        name="financial-agent",
        deps_type=AgentDependencies,
        output_type=FinancialResult,
        system_prompt=(
            "You are the financial agent. Input and output are JSON. Use both cash_balance_tool and "
            "financial_report_tool and report their dated results."
        ),
        tools=[cash_balance_tool, financial_report_tool],
    )

    orchestrator_agent = Agent(
        model,
        name="orchestrator-agent",
        deps_type=AgentDependencies,
        output_type=str,
        system_prompt=(
            "You are the orchestrator. Synthesize the supplied JSON results "
            "from the external customer partner and the inventory, quoting, "
            "ordering, and financial agents into "
            "one concise text response. Do not invent values or claim an order "
            "is accepted unless the ordering result confirms it. Treat cash "
            "balance, inventory valuation, and other financial metrics as "
            "internal information: include them only when the customer "
            "explicitly requests a financial report."
        ),
    )
    customer_agent = Agent(
        model,
        name="customer-agent",
        deps_type=AgentDependencies,
        output_type=CustomerContextResult,
        system_prompt=(
            "You are an external customer-partner agent. Input and output are JSON. "
            "Preserve the customer's request exactly, identify useful context such as "
            "event, urgency, budget constraints, and delivery expectations, and list "
            "negotiation points. When an offer is supplied, evaluate it and set decision "
            "to accept, decline, or revised_request. A revised_request must be a complete "
            "replacement request, and you may request at most one revision. Never invent "
            "prices, stock, or promises."
        ),
    )
    return {
        "customer": customer_agent,
        "orchestrator": orchestrator_agent,
        "inventory": inventory_agent,
        "quoting": quoting_agent,
        "ordering": ordering_agent,
        "financial": financial_agent,
    }


async def _run_multi_agent_workflow(request: str, request_date: str) -> WorkflowResult:
    """Run independent preparation tasks concurrently, then fulfill and report."""
    _log_event("orchestrator.start", request_date=request_date, prompt=request)
    agents = _build_agents()
    deps = AgentDependencies(engine=db_engine)
    customer_input = CustomerRequestPayload(
        request_date=request_date,
        customer_request=request,
    ).model_dump_json()
    _log_event("orchestrator.delegate", agent="customer-agent", payload=customer_input)
    customer_result = (await agents["customer"].run(customer_input, deps=deps)).output
    _log_event("agent.response", agent="customer-agent", response=customer_result.model_dump())
    attempts = []
    current_request = customer_result.customer_request
    for negotiation_round in range(2):
        request_payload = CustomerRequestPayload(
            request_date=request_date,
            customer_request=current_request,
            customer_context=customer_result.customer_context,
            negotiation_points=customer_result.negotiation_points,
        ).model_dump_json()
        _log_event("orchestrator.parallel_start", round=negotiation_round,
                   agents=["inventory-agent", "quoting-agent"])
        _log_event("orchestrator.delegate", agent="inventory-agent",
                   round=negotiation_round, payload=request_payload)
        _log_event("orchestrator.delegate", agent="quoting-agent",
                   round=negotiation_round, payload=request_payload)
        inventory_run, quote_run = await asyncio.gather(
            agents["inventory"].run(request_payload, deps=deps),
            agents["quoting"].run(request_payload, deps=deps),
        )
        inventory_result = inventory_run.output
        quote_result = quote_run.output
        _log_event("agent.response", agent="inventory-agent", round=negotiation_round,
                   response=inventory_result.model_dump())
        _log_event("agent.response", agent="quoting-agent", round=negotiation_round,
                   response=quote_result.model_dump())
        offer_payload = json.dumps({
            "request_date": request_date, "customer_request": current_request,
            "customer_context": customer_result.model_dump(),
            "inventory_result": inventory_result.model_dump(),
            "quote_result": quote_result.model_dump(),
            "negotiation_round": negotiation_round,
            "instruction": "Evaluate this offer. Accept, decline, or provide one complete revised_request.",
        }, ensure_ascii=True)
        _log_event("orchestrator.delegate", agent="customer-agent",
                   round=negotiation_round, payload=offer_payload)
        customer_result = (await agents["customer"].run(offer_payload, deps=deps)).output
        customer_result.negotiation_round = negotiation_round
        decision = customer_result.decision
        revised = customer_result.revised_request
        attempts.append({
            "round": negotiation_round, "request": current_request,
            "decision": decision, "revised_request": revised,
            "quote": quote_result.model_dump(), "inventory": inventory_result.model_dump(),
        })
        _log_event("customer.evaluation", round=negotiation_round, decision=decision,
                   revised_request=revised)
        if decision == "accept":
            break
        if decision == "revised_request" and revised and negotiation_round == 0:
            current_request = revised
            continue
        decision = "decline"
        customer_result.decision = decision
        break
    if customer_result.decision == "accept":
        order_payload = json.dumps(
            {
                "request_date": request_date,
                "customer_request": current_request,
                "inventory_result": inventory_result.model_dump(),
                "quote_result": quote_result.model_dump(),
            },
            ensure_ascii=True,
        )
        _log_event("orchestrator.delegate", agent="ordering-agent", payload=order_payload)
        order_result = (await agents["ordering"].run(order_payload, deps=deps)).output
    else:
        order_result = OrderResult(
            request_date=request_date, accepted=False, delivery_date="",
            transactions=[], message="Customer did not accept the offer; no order was placed.",
        )
        _log_event("ordering.skipped", reason="customer_not_accepted",
                   decision=customer_result.decision)
    _log_event("agent.response", agent="ordering-agent", response=order_result.model_dump())
    _log_event(
        "orchestrator.delegate",
        agent="financial-agent",
        payload=json.dumps({"as_of_date": request_date}, ensure_ascii=True),
    )
    financial_result = (await agents["financial"].run(
        json.dumps({"as_of_date": request_date}, ensure_ascii=True),
        deps=deps,
    )).output
    _log_event("agent.response", agent="financial-agent", response=financial_result.model_dump())
    final_payload = json.dumps(
        {
            "request_date": request_date,
            "customer_context": customer_result.model_dump(),
            "inventory_result": inventory_result.model_dump(),
            "quote_result": quote_result.model_dump(),
            "order_result": order_result.model_dump(),
            "financial_result": financial_result.model_dump(),
        },
        ensure_ascii=True,
    )
    result = await agents["orchestrator"].run(final_payload, deps=deps)
    _log_event("orchestrator.response", request_date=request_date, response=result.output)
    return WorkflowResult(
        response=result.output,
        inventory_result=inventory_result,
        quote_result=quote_result,
        order_result=order_result,
        financial_result=financial_result,
        customer_result=customer_result,
        negotiation_attempts=attempts,
    )


def call_multi_agent_system(request: str, request_date: str) -> WorkflowResult:
    """Process one customer request through the external partner and team."""
    return asyncio.run(_run_multi_agent_workflow(request, request_date))


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios(full_evaluation: bool = True):
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
        if not full_evaluation:
            quote_requests_sample = quote_requests_sample.iloc[[0, 3, 9, 10]].copy()
            logger.info(
                "evaluation.mode mode=smoke scenarios=%s; use --full for all scenarios",
                len(quote_requests_sample),
            )
        else:
            logger.info(
                "evaluation.mode mode=full scenarios=%s",
                len(quote_requests_sample),
            )
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")
        _log_event(
            "scenario.start",
            request_id=idx + 1,
            request_date=request_date,
            job=row["job"],
            event=row["event"],
        )

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        workflow_result = call_multi_agent_system(request_with_date, request_date)
        response = workflow_result.response
        _log_event("scenario.end", request_id=idx + 1, response=response)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        order_result = workflow_result.order_result
        # Keep one structured row per negotiation attempt. Financial state is
        # emitted only on the final attempt to avoid misleading duplicate snapshots.
        for attempt_index, attempt in enumerate(workflow_result.negotiation_attempts):
            is_final = attempt_index == len(workflow_result.negotiation_attempts) - 1
            quote_data = attempt["quote"]
            inventory_data = attempt["inventory"]
            results.append(
                {
                    "request_id": f"{idx + 1}.{attempt_index}",
                    "request_date": request_date,
                    "negotiation_round": attempt["round"],
                    "negotiation_decision": attempt["decision"],
                    "revised_request": attempt["revised_request"] or "",
                    "accepted": order_result.accepted if is_final else False,
                    "delivery_date": order_result.delivery_date if is_final else "",
                    "order_message": order_result.message if is_final else "Negotiation continued.",
                    "transaction_ids": json.dumps(order_result.transactions) if is_final else "[]",
                    "quote_subtotal": quote_data["subtotal"],
                    "quote_discount": quote_data["discount"],
                    "quote_total": quote_data["total_amount"],
                    "inventory_status": inventory_data["summary"],
                    "cash_balance": current_cash if is_final else "",
                    "inventory_value": current_inventory if is_final else "",
                    "total_assets": workflow_result.financial_result.total_assets if is_final else "",
                }
            )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Export structured scenario results; conversation transcripts remain in the log.
    pd.DataFrame(results).to_csv(
        "test_results.csv",
        index=False,
        lineterminator="\n",
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the multi-agent evaluation.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Explicitly request the default full evaluation.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the four-case smoke test instead of all scenarios.",
    )
    args = parser.parse_args()
    results = run_test_scenarios(full_evaluation=args.full or not args.smoke)
