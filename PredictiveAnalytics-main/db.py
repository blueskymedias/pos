import json
from flask import Flask, jsonify, request, send_file, render_template_string
import psycopg2
from psycopg2 import sql
from flask_bcrypt import Bcrypt
from flask_cors import CORS
import jwt
from datetime import datetime, timedelta
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import io

app = Flask(__name__)
CORS(app)

MODEL_FILE = "sales_forecast.pkl"
DB_HOST = 'localhost'
DB_NAME = 'postgres'
DB_USER = 'postgres'
DB_PASSWORD = '1616'  # Update if different
SECRET_KEY = "this is a secret key this is a secret keyyyy!!!!"

logging.basicConfig(level=logging.DEBUG)

def get_db_connection():
    try:
        connection = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return connection
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        raise

def create_users_table_if_not_exists():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id SERIAL PRIMARY KEY,
            email_id TEXT NOT NULL UNIQUE,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    connection.commit()
    cursor.close()
    connection.close()

def create_sales_table_if_not_exists():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            sales_id SERIAL PRIMARY KEY,
            date DATE NOT NULL UNIQUE,
            sales INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    connection.commit()
    cursor.close()
    connection.close()

def create_predictions_table_if_not_exists():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id SERIAL PRIMARY KEY,
            date DATE UNIQUE,
            predicted_sales FLOAT
        );
    """)
    connection.commit()
    cursor.close()
    connection.close()

def create_inventory_table_if_not_exists():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inventory (
            id SERIAL PRIMARY KEY,
            days INT NOT NULL,
            safety_stock DECIMAL(10,2) NOT NULL,
            optimized_inventory DECIMAL(10,2) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    connection.commit()
    cursor.close()
    connection.close()

def create_pricing_table_if_not_exists():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pricing (
            id SERIAL PRIMARY KEY,
            product_id INTEGER NOT NULL,
            current_price DECIMAL(10,2) NOT NULL,
            optimized_price DECIMAL(10,2) NOT NULL,
            demand_elasticity DECIMAL(10,2) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    connection.commit()
    cursor.close()
    connection.close()

create_users_table_if_not_exists()
create_sales_table_if_not_exists()
create_predictions_table_if_not_exists()
create_inventory_table_if_not_exists()
create_pricing_table_if_not_exists()

bcrypt = Bcrypt(app)

def encode_password(password):
    return bcrypt.generate_password_hash(password).decode('utf-8')

def check_password(hashed_password, password):
    return bcrypt.check_password_hash(hashed_password, password)

@app.route('/')
def serve_frontend():
    with open('index.html', 'r') as f:
        html_content = f.read()
    return render_template_string(html_content)

@app.route('/register-users', methods=['POST'])
def register_user():
    data = request.json
    email_id = data.get('email_id')
    username = data.get('username')
    password = data.get('password')
    if not all([email_id, username, password]):
        logging.warning("Missing fields in registration request")
        return jsonify({"message": "Missing required fields"}), 400
    hashed_password = encode_password(password)
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("""
            INSERT INTO users (email_id, username, password) VALUES (%s, %s, %s);
        """, (email_id, username, hashed_password))
        connection.commit()
        logging.info(f"User {username} registered successfully")
        return jsonify({"message": "User registered successfully"}), 201
    except psycopg2.IntegrityError as e:
        logging.error(f"Integrity error: {str(e)}")
        return jsonify({"message": "Email or username already exists"}), 400
    except Exception as e:
        logging.error(f"Registration error: {str(e)}")
        return jsonify({"message": "Registration failed"}), 500
    finally:
        cursor.close()
        connection.close()

@app.route('/login', methods=['POST'])
def login_user():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    if not all([username, password]):
        logging.warning("Missing fields in login request")
        return jsonify({"message": "Missing username or password"}), 400
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT * FROM users WHERE username = %s;", (username,))
        user = cursor.fetchone()
        if user is None or not check_password(user[3], password):
            logging.warning(f"Login failed for username: {username}")
            return jsonify({"message": "Invalid username or password"}), 401
        payload = {
            'username': username,
            'user_id': int(user[0]),
            'exp': datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
        logging.info(f"User {username} logged in successfully")
        return jsonify({"message": "Login successful", "token": token}), 200
    except Exception as e:
        logging.error(f"Login error: {str(e)}")
        return jsonify({"message": "Login failed", "error": str(e)}), 500
    finally:
        cursor.close()
        connection.close()

@app.route('/add-sales', methods=['POST'])
def add_sales():
    data = request.json
    date = data.get('date')
    sales = data.get('sales')
    if not all([date, sales]):
        return jsonify({"message": "Missing date or sales"}), 400
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("""
            INSERT INTO sales (date, sales) 
            VALUES (%s, %s) 
            ON CONFLICT (date) 
            DO UPDATE SET sales = EXCLUDED.sales;
        """, (date, int(sales)))
        connection.commit()
        return jsonify({"message": "Sales data added successfully"}), 201
    except Exception as e:
        logging.error(f"Add sales error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        connection.close()

@app.route('/get-sales', methods=['GET'])
def get_sales():
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    if not start_date or not end_date:
        return jsonify({"error": "Missing start_date or end_date"}), 400
    try:
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT date, sales FROM sales WHERE date BETWEEN %s AND %s ORDER BY date ASC;", (start_date, end_date))
        sales = cursor.fetchall()
        cursor.close()
        conn.close()
        if not sales:
            return jsonify({"message": "No sales data found for the given date range"}), 404
        return jsonify([{"date": row[0].strftime("%Y-%m-%d"), "sales": row[1]} for row in sales])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train-sales-model', methods=['POST'])
def train_sales_model():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT date, sales FROM sales ORDER BY date ASC;")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        if not rows:
            return jsonify({"error": "No sales data available"}), 404
        df = pd.DataFrame(rows, columns=['date', 'sales'])
        df['date'] = pd.to_datetime(df['date'])
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        X = df[['days_since_start']]
        y = df['sales']
        model = LinearRegression()
        model.fit(X, y)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)
        return jsonify({"message": "Sales forecasting model trained successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict-sales', methods=['POST'])
def predict_sales():
    try:
        data = request.json
        days = int(data.get("days", 30))
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(date) FROM sales;")
        last_date = cursor.fetchone()[0]
        if not last_date:
            return jsonify({"error": "No sales data available"}), 404
        last_date = pd.to_datetime(last_date)
        future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
        future_days_since_start = [(date - last_date).days for date in future_dates]
        predicted_sales = model.predict(np.array(future_days_since_start).reshape(-1, 1))
        predicted_sales = np.clip(predicted_sales, 0, None)
        predictions = []
        for date, sales in zip(future_dates, predicted_sales):
            formatted_date = date.strftime("%Y-%m-%d")
            predictions.append({"date": formatted_date, "predicted_sales": round(sales, 2)})
            cursor.execute("DELETE FROM predictions WHERE date = %s;", (formatted_date,))
            cursor.execute("INSERT INTO predictions (date, predicted_sales) VALUES (%s, %s);", (formatted_date, float(sales)))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify(predictions), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/optimize-pricing', methods=['POST'])
def optimize_pricing():
    try:
        data = request.json
        product_id = int(data.get("product_id"))
        current_price = float(data.get("current_price"))
        demand_elasticity = float(data.get("demand_elasticity"))
        if not all([product_id, current_price, demand_elasticity]):
            return jsonify({"error": "Missing required fields"}), 400
        optimized_price = round(current_price * (1 + demand_elasticity), 2)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO pricing (product_id, current_price, optimized_price, demand_elasticity)
            VALUES (%s, %s, %s, %s) RETURNING id;
        """, (product_id, current_price, optimized_price, demand_elasticity))
        record_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Price optimization successful", "optimized_price": optimized_price, "record_id": record_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/optimize-inventory', methods=['POST'])
def optimize_inventory():
    try:
        data = request.json
        days = int(data.get("days", 30))
        safety_stock = float(data.get("safety_stock", 10))
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(date) FROM sales;")
        last_date = cursor.fetchone()[0]
        if not last_date:
            return jsonify({"error": "No sales data available"}), 404
        last_date = pd.to_datetime(last_date)
        future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
        future_days_since_start = [(date - last_date).days for date in future_dates]
        predicted_sales = model.predict(np.array(future_days_since_start).reshape(-1, 1))
        predicted_sales = np.clip(predicted_sales, 0, None)  # Ensure non-negative predictions
        total_demand = sum(predicted_sales)
        optimized_inventory = max(float(total_demand + safety_stock), 0)  # Ensure non-negative inventory
        cursor.execute("""
            INSERT INTO inventory (days, safety_stock, optimized_inventory)
            VALUES (%s, %s, %s) RETURNING id;
        """, (days, safety_stock, optimized_inventory))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"optimized_inventory": round(optimized_inventory, 2)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/visualize-sales', methods=['GET'])
def visualize_sales():
    try:
        days = int(request.args.get("days", 30))
        safety_stock = int(request.args.get("safety_stock", 10))
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT date, sales FROM sales ORDER BY date ASC;")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        if not rows:
            return jsonify({"error": "No sales data available"}), 404
        df = pd.DataFrame(rows, columns=['date', 'sales'])
        df['date'] = pd.to_datetime(df['date'])
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        last_date = df['date'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
        future_days_since_start = [(date - df['date'].min()).days for date in future_dates]
        predicted_sales = model.predict(np.array(future_days_since_start).reshape(-1, 1))
        predicted_sales = np.clip(predicted_sales, 0, None)
        total_demand = np.sum(predicted_sales)
        optimized_inventory = total_demand + safety_stock
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Sales", color="blue")
        ax1.plot(df['date'], df['sales'], 'bo-', label="Actual Sales")
        ax1.plot(future_dates, predicted_sales, 'r--', label="Predicted Sales")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.legend(loc="upper left")
        ax2 = ax1.twinx()
        ax2.set_ylabel("Optimized Inventory", color="green")
        ax2.axhline(y=optimized_inventory, color="green", linestyle="dotted", label="Optimized Inventory")
        ax2.tick_params(axis="y", labelcolor="green")
        ax2.legend(loc="upper right")
        plt.title("Sales Forecasting & Inventory Optimization")
        plt.grid(True)
        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plt.close()
        return send_file(img, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)