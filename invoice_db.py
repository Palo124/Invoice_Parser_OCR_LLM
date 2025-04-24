# invoice_db.py
import os
import psycopg2
from dotenv import load_dotenv
from datetime import datetime
import re

def to_numeric(value):
    """
    Converts a numeric value given as a string that might include unit labels 
    (e.g., '0.026 [GJ]', '28,370 [GJ]', '0,001 [M3]') to a float.
    
    It does the following:
      1. Strips the value.
      2. Removes any content within square brackets.
      3. Removes spaces.
      4. If both comma and period are present, assumes comma is a thousand separator.
         If only comma is present, assumes it's a decimal separator.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return value

    value_str = str(value).strip()
    # Remove any unit annotations in square brackets (e.g., [GJ], [M3])
    value_str = re.sub(r'\[.*?\]', '', value_str)
    # Remove spaces
    value_str = value_str.replace(" ", "")

    # Determine if comma is a decimal separator or a thousand separator.
    if ',' in value_str and '.' in value_str:
        # Assume comma is a thousand separator and remove it.
        value_str = value_str.replace(',', '')
    elif ',' in value_str and '.' not in value_str:
        # Assume comma is a decimal separator and replace it with period.
        value_str = value_str.replace(',', '.')

    try:
        return float(value_str)
    except ValueError as e:
        print(f"Conversion error for value {value}: {e}")
        return None

class InvoiceDB:
    def __init__(self, env_file="secret_keys.env", db_host="localhost", db_name="mydb", db_user="postgres"):
        load_dotenv(env_file)
        self.db_password = os.getenv("DB_PSWD")
        self.conn = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=self.db_password
        )
        self.cursor = self.conn.cursor()

    def parse_date(self, date_str):
        return datetime.strptime(date_str, "%d.%m.%Y").date() if date_str else None

    def insert_invoice(self, invoice_data):
        # Convert necessary values first
        payment_without_tax = to_numeric(invoice_data.get("Payment without TAX"))
        billing_period_months = to_numeric(invoice_data.get("Billing period months"))
        
        # Calculate average monthly payment if billing_period_months is valid
        avg_monthly_payment_without_tax = (
            payment_without_tax / billing_period_months 
            if billing_period_months and billing_period_months != 0 
            else None
        )
        # Prepare the invoice header data with proper conversion
        invoice_header = {
            "original_number": invoice_data.get("Číslo originálu"),
            "building": invoice_data.get("Building"),
            "invoiced_to": invoice_data.get("Invoiced to"),
            "invoice_date": self.parse_date(invoice_data.get("Invoice date")),
            "billing_start_date": self.parse_date(invoice_data.get("Billing period")),
            "billing_end_date": self.parse_date(invoice_data.get("Billing Period To")),
            "billing_period_days": to_numeric(invoice_data.get("Billing period days")),
            "billing_period_months": billing_period_months,
            "invoice_no": invoice_data.get("Invoice NO"),
            "company": invoice_data.get("Company"),
            "invoice_description": invoice_data.get("Description"),
            "information": invoice_data.get("info"),
            "comment": invoice_data.get("Comment"),
            "tax_percent": to_numeric(invoice_data.get("TAX %")),
            "tax_amount": to_numeric(invoice_data.get("TAX Kč")),
            "payment_without_tax": payment_without_tax,
            "payment": to_numeric(invoice_data.get("Payment")),
            "advances": to_numeric(invoice_data.get("Advances")),
            "advances_without_tax": to_numeric(invoice_data.get("Advances without TAX")),
            "final_payment": to_numeric(invoice_data.get("Final payment")),
            "category": invoice_data.get("Category"),
            "contact": invoice_data.get("Contact"),
            "contract": invoice_data.get("Contract"),
            "payment_type": invoice_data.get("Type of payment"),
            "source": invoice_data.get("Source"),
            "duplicity": invoice_data.get("Duplicity"),
            "avg_monthly_payment_without_tax": avg_monthly_payment_without_tax
        }

        insert_invoice_query = """
            INSERT INTO invoices (
                original_number, building, invoiced_to, invoice_date, billing_start_date, billing_end_date,
                billing_period_days, billing_period_months, invoice_no, company, invoice_description, information,
                comment, tax_percent, tax_amount, payment_without_tax, payment, advances, advances_without_tax,
                final_payment, category, contact, contract, payment_type, source, duplicity, avg_monthly_payment_without_tax
            ) VALUES (
                %(original_number)s, %(building)s, %(invoiced_to)s, %(invoice_date)s, %(billing_start_date)s, %(billing_end_date)s,
                %(billing_period_days)s, %(billing_period_months)s, %(invoice_no)s, %(company)s, %(invoice_description)s, %(information)s,
                %(comment)s, %(tax_percent)s, %(tax_amount)s, %(payment_without_tax)s, %(payment)s, %(advances)s, %(advances_without_tax)s,
                %(final_payment)s, %(category)s, %(contact)s, %(contract)s, %(payment_type)s, %(source)s, %(duplicity)s, %(avg_monthly_payment_without_tax)s
            ) RETURNING id;
        """

        self.cursor.execute(insert_invoice_query, invoice_header)
        invoice_id = self.cursor.fetchone()[0]

        # Insert each invoice item, converting numeric fields as needed.
        insert_item_query = """
            INSERT INTO invoice_items (
                invoice_id, description, quantity, unit_price_excluding_tax,
                tax_percent, tax_amount, total_excluding_tax, total_including_tax
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """

        for item in invoice_data.get("Item Details", []):
            self.cursor.execute(
                insert_item_query,
                (
                    invoice_id,
                    item.get("Popis"),
                    to_numeric(item.get("Množství")),
                    to_numeric(item.get("Jednotková cena bez DPH")),
                    to_numeric(item.get("DPH %")),
                    to_numeric(item.get("DPH Kč")),
                    to_numeric(item.get("Celková cena bez DPH")),
                    to_numeric(item.get("Celková cena s DPH"))
                )
            )

        self.conn.commit()
        return invoice_id

    def close(self):
        self.cursor.close()
        self.conn.close()
