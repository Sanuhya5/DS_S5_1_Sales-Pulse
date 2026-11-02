import io
import re
import pandas as pd


COLUMN_ALIASES = {
    "date": [
        "date", "orderdate", "transactiondate",
        "invoicedate", "salesdate", "timestamp"
    ],
    "revenue": [
        "sales", "revenue", "salesamount",
        "totalamount", "total_sales", "net_sales",
        "amount", "turnover"
    ],
    "quantity": [
        "quantity", "qty", "units", "unitssold",
        "quantityordered", "orderquantity"
    ],
    "unit_price": [
        "unitprice", "price", "sellingprice",
        "eachprice"
    ],
    "unit_cost": [
        "unitcost", "costprice", "purchasecost"
    ],
    "product": [
        "product", "productname", "productcode",
        "item", "itemname", "sku", "stockcode"
    ],
    "category": [
        "category", "productline", "productcategory",
        "department", "family"
    ],
    "store": [
        "store", "storeid", "branch", "branchid", "shop"
    ],
    "region": [
        "region", "location", "country", "state",
        "city", "territory"
    ],
    "order_id": [
        "orderid", "transactionid", "invoiceno",
        "invoice", "receiptid"
    ]
}


def normalize_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def load_dataset(file_bytes, filename):
    extension = filename.lower().rsplit(".", 1)[-1]

    if extension == "csv":
        errors = []

        for encoding in ["utf-8-sig", "utf-8", "cp1252"]:
            try:
                return pd.read_csv(
                    io.BytesIO(file_bytes),
                    encoding=encoding,
                    low_memory=False
                )
            except Exception as error:
                errors.append(str(error))

        raise ValueError(
            "Could not read the CSV file: " + errors[-1]
        )

    if extension in ["xlsx", "xls"]:
        return pd.read_excel(io.BytesIO(file_bytes))

    raise ValueError("Upload a CSV or Excel file.")


def suggest_columns(data):
    normalized_columns = {
        normalize_name(column): column
        for column in data.columns
    }

    suggestions = {}

    for role, aliases in COLUMN_ALIASES.items():
        suggestions[role] = None

        for alias in aliases:
            alias = normalize_name(alias)

            if alias in normalized_columns:
                suggestions[role] = normalized_columns[alias]
                break

    return suggestions


def convert_numeric(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    values = series.astype("string").str.strip()

    # Remove common currency symbols and whitespace.
    values = values.str.replace(
        r"[$£€¥₹\s]", "", regex=True
    )

    # Assumes comma is a thousands separator.
    values = values.str.replace(",", "", regex=False)

    return pd.to_numeric(values, errors="coerce")


def prepare_dataset(data, mapping):
    if data is None or data.empty:
        raise ValueError("The dataset is empty.")

    date_column = mapping.get("date")
    revenue_column = mapping.get("revenue")
    quantity_column = mapping.get("quantity")
    price_column = mapping.get("unit_price")

    if not date_column:
        raise ValueError("Select a date column.")

    if not revenue_column and not (
        quantity_column and price_column
    ):
        raise ValueError(
            "Select a sales/revenue column, or select both "
            "quantity and unit price."
        )

    result = pd.DataFrame(index=data.index)

    result["date"] = pd.to_datetime(
        data[date_column],
        errors="coerce",
        format="mixed"
    )

    result["date"] = result["date"].dt.normalize()

    invalid_dates = int(result["date"].isna().sum())

    result = result.loc[result["date"].notna()].copy()

    if revenue_column:
        result["revenue"] = convert_numeric(
            data.loc[result.index, revenue_column]
        )
    else:
        quantity = convert_numeric(
            data.loc[result.index, quantity_column]
        )
        price = convert_numeric(
            data.loc[result.index, price_column]
        )

        result["revenue"] = quantity * price

    if quantity_column:
        result["quantity"] = convert_numeric(
            data.loc[result.index, quantity_column]
        )

    if price_column:
        result["unit_price"] = convert_numeric(
            data.loc[result.index, price_column]
        )
    elif quantity_column:
        quantity = result["quantity"].replace(0, float("nan"))
        result["unit_price"] = result["revenue"] / quantity

    cost_column = mapping.get("unit_cost")

    if cost_column:
        result["unit_cost"] = convert_numeric(
            data.loc[result.index, cost_column]
        )

    for role in ["product", "category", "store", "region", "order_id"]:
        column = mapping.get(role)

        if column:
            result[role] = (
                data.loc[result.index, column]
                .astype("string")
                .str.strip()
            )

            result[role] = result[role].replace(
                {"": pd.NA, "nan": pd.NA, "None": pd.NA}
            )

    missing_revenue = int(result["revenue"].isna().sum())

    result = result.dropna(subset=["revenue"])

    if result.empty:
        raise ValueError(
            "No valid rows remain after checking dates and sales."
        )

    notes = []

    if invalid_dates:
        notes.append(
            f"{invalid_dates} rows were dropped due to invalid dates."
        )

    if missing_revenue:
        notes.append(
            f"{missing_revenue} rows were dropped due to invalid sales."
        )

    duplicate_count = int(result.duplicated().sum())

    if duplicate_count:
        result = result.drop_duplicates()
        notes.append(
            f"{duplicate_count} exact duplicate rows were removed."
        )

    if "quantity" in result.columns:
        if result["quantity"].isna().any():
            notes.append(
                "Some quantities are missing; unit totals may be incomplete."
            )

    if (result["revenue"] < 0).any():
        notes.append(
            "Negative sales were retained because they may represent returns."
        )

    result = result.sort_values("date").reset_index(drop=True)

    return result, notes


def daily_sales(data, target="revenue"):
    return (
        data.groupby("date", as_index=False)[target]
        .sum(min_count=1)
        .sort_values("date")
    )
