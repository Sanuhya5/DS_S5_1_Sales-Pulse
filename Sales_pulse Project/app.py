import pandas as pd
import plotly.express as px
import streamlit as st

from preprocessing import (
    load_dataset,
    suggest_columns,
    prepare_dataset,
    daily_sales
)
from forecasting import run_forecast


st.set_page_config(
    page_title="Sales Pulse",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Pulse")
st.caption(
    "Multi-dataset sales analytics and demand forecasting"
)


def format_amount(value):
    return f"{value:,.2f}"


def get_period_dates(period, first_date, last_date):
    end = pd.Timestamp(last_date)

    if period == "All time":
        start = pd.Timestamp(first_date)

    elif period == "Last 30 days":
        start = end - pd.Timedelta(days=29)

    elif period == "3 months":
        start = end - pd.DateOffset(months=3) + pd.Timedelta(days=1)

    elif period == "4 months":
        start = end - pd.DateOffset(months=4) + pd.Timedelta(days=1)

    elif period == "6 months":
        start = end - pd.DateOffset(months=6) + pd.Timedelta(days=1)

    elif period == "1 year":
        start = end - pd.DateOffset(years=1) + pd.Timedelta(days=1)

    else:
        start = pd.Timestamp(first_date)

    start = max(start, pd.Timestamp(first_date))

    return start.date(), end.date()


def make_quarterly_report(data):
    work = data.copy()

    work["Quarter"] = (
        work["date"].dt.to_period("Q").astype(str)
    )

    quarterly = work.groupby(
        "Quarter", as_index=False
    ).agg(
        Revenue=("revenue", "sum"),
        Records=("date", "count")
    )

    if "quantity" in work.columns:
        quantities = work.groupby("Quarter")["quantity"].sum()
        quarterly["Units sold"] = quarterly["Quarter"].map(quantities)

    quarterly["QoQ growth (%)"] = (
        quarterly["Revenue"].pct_change() * 100
    )

    return quarterly


# 1. Upload multiple datasets
with st.sidebar:
    st.header("Upload datasets")

    uploads = st.file_uploader(
        "Choose CSV or Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True
    )

if not uploads:
    st.info(
        "Upload one or more sales datasets to begin. "
        "Each dataset should contain a date and a sales amount, "
        "or quantity and unit price."
    )
    st.stop()


raw_datasets = {}
for uploaded in uploads:
    try:
        raw_datasets[uploaded.name] = load_dataset(
            uploaded.getvalue(),
            uploaded.name
        )
    except Exception as error:
        st.error(f"{uploaded.name}: {error}")

if not raw_datasets:
    st.stop()


# 2. Map and normalize each dataset
clean_datasets = {}
quality_notes = {}

st.header("1. Review dataset columns")
st.caption(
    "Check the suggested mappings. Different files can use different column names."
)

roles = [
    ("date", "Date"),
    ("revenue", "Revenue / sales"),
    ("quantity", "Quantity"),
    ("unit_price", "Unit price"),
    ("unit_cost", "Unit cost"),
    ("product", "Product"),
    ("category", "Category"),
    ("store", "Store / branch"),
    ("region", "Region"),
    ("order_id", "Order ID")
]

for file_number, (filename, raw) in enumerate(raw_datasets.items()):
    with st.expander(
        f"{filename} — {len(raw):,} rows",
        expanded=(file_number == 0)
    ):
        st.dataframe(raw.head(5), use_container_width=True)

        guesses = suggest_columns(raw)
        options = ["(Not available)"] + list(raw.columns)
        mapping = {}

        left, right = st.columns(2)

        for position, (role, label) in enumerate(roles):
            column_container = left if position % 2 == 0 else right

            guessed = guesses.get(role)
            default_index = (
                options.index(guessed)
                if guessed in options else 0
            )

            with column_container:
                chosen = st.selectbox(
                    label,
                    options,
                    index=default_index,
                    key=f"{file_number}_{role}_{filename}"
                )

            mapping[role] = (
                None if chosen == "(Not available)" else chosen
            )

        try:
            clean, notes = prepare_dataset(raw, mapping)
            clean_datasets[filename] = clean
            quality_notes[filename] = notes

            st.success(
                f"Validated {len(clean):,} rows. "
                f"Date coverage: {clean['date'].min().date()} "
                f"to {clean['date'].max().date()}."
            )

            for note in notes:
                st.warning(note)

        except Exception as error:
            st.error(f"Cannot analyze this dataset: {error}")


if not clean_datasets:
    st.error(
        "No valid dataset remains. Check your date and sales column mappings."
    )
    st.stop()


# 3. Choose the active dataset
st.header("2. Select dataset")

dataset_name = st.selectbox(
    "Dataset to analyze",
    list(clean_datasets.keys())
)

data = clean_datasets[dataset_name].copy()

first_date = data["date"].min().date()
last_date = data["date"].max().date()

# 4. Global historical date filter
st.header("3. Historical analysis period")

period = st.selectbox(
    "Select period",
    [
        "Last 30 days",
        "3 months",
        "4 months",
        "6 months",
        "1 year",
        "All time",
        "Custom range"
    ],
    index=5
)

if period == "Custom range":
    left, right = st.columns(2)

    with left:
        start_date = st.date_input(
            "Start date",
            value=first_date,
            min_value=first_date,
            max_value=last_date
        )

    with right:
        end_date = st.date_input(
            "End date",
            value=last_date,
            min_value=first_date,
            max_value=last_date
        )
else:
    start_date, end_date = get_period_dates(
        period, first_date, last_date
    )

if start_date > end_date:
    st.error("The start date must be before the end date.")
    st.stop()

filtered = data[
    (data["date"].dt.date >= start_date)
    & (data["date"].dt.date <= end_date)
].copy()

if filtered.empty:
    st.warning("No records exist in this date range.")
    st.stop()

st.caption(
    f"Selected period: {start_date} to {end_date}. "
    f"Available history: {first_date} to {last_date}."
)


# 5. KPIs
st.header("4. Sales overview")

total_revenue = filtered["revenue"].sum()
total_units = (
    filtered["quantity"].sum()
    if "quantity" in filtered.columns
    else None
)

if "order_id" in filtered.columns:
    total_orders = filtered["order_id"].nunique()
else:
    total_orders = len(filtered)

daily_totals = filtered.groupby("date")["revenue"].sum()
average_daily = daily_totals.mean()

kpis = st.columns(4)

kpis[0].metric(
    "Total revenue",
    format_amount(total_revenue)
)

kpis[1].metric(
    "Units sold",
    f"{total_units:,.0f}" if total_units is not None else "Unavailable"
)

kpis[2].metric(
    "Orders / records",
    f"{total_orders:,}"
)

kpis[3].metric(
    "Average daily revenue",
    format_amount(average_daily)
)

if "unit_cost" in filtered.columns and "quantity" in filtered.columns:
    # This is an estimate of gross profit and assumes unit_cost is comparable
    # to unit_price and quantity is the number of units sold.
    unit_price = filtered["unit_price"]

    if "unit_price" in filtered.columns:
        valid_cost_rows = filtered.dropna(
            subset=["unit_cost", "quantity", "unit_price"]
        )
        estimated_profit = (
            (valid_cost_rows["unit_price"] - valid_cost_rows["unit_cost"])
            * valid_cost_rows["quantity"]
        ).sum()

        st.metric(
            "Estimated gross profit",
            format_amount(estimated_profit)
        )
    else:
        st.info(
            "Profit is unavailable because a usable unit price was not identified."
        )


# 6. Historical trend
st.header("5. Historical sales trend")

granularity = st.radio(
    "Chart interval",
    ["Daily", "Weekly", "Monthly", "Quarterly"],
    horizontal=True,
    index=2
)

trend_data = daily_sales(filtered)

if granularity == "Weekly":
    trend_data["Period"] = (
        trend_data["date"].dt.to_period("W").dt.start_time
    )
elif granularity == "Monthly":
    trend_data["Period"] = (
        trend_data["date"].dt.to_period("M").dt.start_time
    )
elif granularity == "Quarterly":
    trend_data["Period"] = (
        trend_data["date"].dt.to_period("Q").dt.start_time
    )
else:
    trend_data["Period"] = trend_data["date"]

if granularity != "Daily":
    trend_data = trend_data.groupby(
        "Period", as_index=False
    )["revenue"].sum()
else:
    trend_data = trend_data.rename(columns={"date": "Period"})

figure = px.line(
    trend_data,
    x="Period",
    y="revenue",
    markers=True,
    title=f"Sales revenue — {granularity}"
)

figure.update_layout(
    xaxis_title="Date",
    yaxis_title="Revenue",
    hovermode="x unified"
)

st.plotly_chart(figure, use_container_width=True)


# 7. Category and product charts
left, right = st.columns(2)

with left:
    if "category" in filtered.columns:
        category_data = (
            filtered.groupby("category", as_index=False)["revenue"]
            .sum()
            .sort_values("revenue", ascending=False)
        )

        st.plotly_chart(
            px.bar(
                category_data.head(15),
                x="category",
                y="revenue",
                title="Revenue by category"
            ),
            use_container_width=True
        )
    else:
        st.info("A category column was not provided.")

with right:
    dimension = next(
        (
            column
            for column in ["store", "region", "product"]
            if column in filtered.columns
        ),
        None
    )

    if dimension:
        breakdown = (
            filtered.groupby(dimension, as_index=False)["revenue"]
            .sum()
            .sort_values("revenue", ascending=False)
            .head(15)
        )

        st.plotly_chart(
            px.pie(
                breakdown,
                names=dimension,
                values="revenue",
                title=f"Revenue share by {dimension}"
            ),
            use_container_width=True
        )
    else:
        st.info("No product, store, or region column was provided.")


# 8. Quarterly report
st.header("6. Quarterly sales report")

quarterly = make_quarterly_report(filtered)

left, right = st.columns(2)

with left:
    st.plotly_chart(
        px.bar(
            quarterly,
            x="Quarter",
            y="Revenue",
            title="Revenue by quarter"
        ),
        use_container_width=True
    )

with right:
    growth_data = quarterly.dropna(
        subset=["QoQ growth (%)"]
    )

    if not growth_data.empty:
        st.plotly_chart(
            px.bar(
                growth_data,
                x="Quarter",
                y="QoQ growth (%)",
                title="Quarter-over-quarter growth"
            ),
            use_container_width=True
        )
    else:
        st.info("More than one quarter is required to calculate growth.")

st.dataframe(
    quarterly,
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "Download quarterly report",
    quarterly.to_csv(index=False).encode("utf-8"),
    file_name="quarterly_sales_report.csv",
    mime="text/csv"
)


# 9. Product performance
if "product" in filtered.columns:
    st.header("7. Product performance")

    product_data = filtered.groupby(
        "product", as_index=False
    ).agg(
        Revenue=("revenue", "sum")
    )

    if "quantity" in filtered.columns:
        quantities = filtered.groupby("product")["quantity"].sum()
        product_data["Units sold"] = product_data["product"].map(quantities)

    product_data = product_data.sort_values(
        "Revenue", ascending=False
    )

    st.dataframe(
        product_data,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download product report",
        product_data.to_csv(index=False).encode("utf-8"),
        file_name="product_sales_report.csv",
        mime="text/csv"
    )


# 10. Forecasting
st.header("8. ML sales forecasting")

target_label = st.selectbox(
    "Forecast target",
    ["Revenue"] + (
        ["Units sold"] if "quantity" in data.columns else []
    )
)

horizon = st.selectbox(
    "Forecast horizon",
    [7, 30, 60, 90],
    index=1,
    format_func=lambda days: f"Next {days} days"
)

model_choice = st.selectbox(
    "Forecasting model",
    ["Compare all", "Baseline only", "ARIMA only", "Prophet only"]
)

st.caption(
    "Forecasting uses the full available history for this dataset. "
    "The historical date filter changes reports and charts, not the "
    "model's training history."
)

if st.button("Run forecast", type="primary"):
    target_column = (
        "revenue" if target_label == "Revenue" else "quantity"
    )

    series = data[["date", target_column]].rename(
        columns={target_column: "target"}
    )

    try:
        results = run_forecast(
            series,
            horizon=horizon,
            model_choice=model_choice
        )

        st.session_state["forecast_results"] = results
        st.session_state["forecast_dataset"] = dataset_name
        st.session_state["forecast_target"] = target_label
        st.session_state["forecast_horizon"] = horizon

    except Exception as error:
        st.error(f"Forecast failed: {error}")

results = st.session_state.get("forecast_results")

if (
    results
    and st.session_state.get("forecast_dataset") == dataset_name
    and st.session_state.get("forecast_target") == target_label
    and st.session_state.get("forecast_horizon") == horizon
):
    st.subheader("Model evaluation")
    st.dataframe(
        results["metrics"],
        use_container_width=True,
        hide_index=True
    )

    st.success(
        f"Selected model: {results['selected_model']}"
    )

    history = results["history"]
    future = results["forecast"]

    import plotly.graph_objects as go

    forecast_figure = go.Figure()

    forecast_figure.add_scatter(
        x=history["date"],
        y=history["target"],
        mode="lines",
        name="Historical actual"
    )

    forecast_figure.add_scatter(
        x=future["date"],
        y=future["prediction"],
        mode="lines+markers",
        name="Forecast"
    )

    forecast_figure.update_layout(
        title=f"Forecast for the next {horizon} days",
        xaxis_title="Date",
        yaxis_title=target_label,
        hovermode="x unified"
    )

    st.plotly_chart(
        forecast_figure,
        use_container_width=True
    )

    st.dataframe(
        future,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download forecast CSV",
        future.to_csv(index=False).encode("utf-8"),
        file_name="sales_forecast.csv",
        mime="text/csv"
    )


# 11. Compare multiple datasets
if len(clean_datasets) > 1:
    st.header("9. Compare uploaded datasets")

    comparison_rows = []

    for name, frame in clean_datasets.items():
        selected_data = frame[
            (frame["date"].dt.date >= start_date)
            & (frame["date"].dt.date <= end_date)
        ]

        comparison_rows.append({
            "Dataset": name,
            "Revenue": selected_data["revenue"].sum(),
            "Rows": len(selected_data),
            "Available from": frame["date"].min().date(),
            "Available until": frame["date"].max().date()
        })

    comparison = pd.DataFrame(comparison_rows)

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.plotly_chart(
        px.bar(
            comparison,
            x="Dataset",
            y="Revenue",
            title="Revenue comparison across datasets"
        ),
        use_container_width=True
    )

    st.download_button(
        "Download dataset comparison",
        comparison.to_csv(index=False).encode("utf-8"),
        file_name="dataset_comparison.csv",
        mime="text/csv"
    )


# 12. Data quality information
with st.expander("Data quality summary"):
    for name, frame in clean_datasets.items():
        st.markdown(f"**{name}**")
        st.write(f"Rows: {len(frame):,}")
        st.write(f"Distinct dates: {frame['date'].nunique():,}")
        st.write(
            f"Date coverage: {frame['date'].min().date()} "
            f"to {frame['date'].max().date()}"
        )

        for note in quality_notes.get(name, []):
            st.write(f"- {note}")

st.caption(
    "Sales Pulse provides estimates for analysis and planning. "
    "Forecast accuracy depends on data quality, historical coverage, "
    "seasonality, promotions, returns, and business changes."
)
