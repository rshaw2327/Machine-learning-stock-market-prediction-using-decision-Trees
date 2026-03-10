from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Premium dark theme for Plotly charts
PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(15, 23, 42, 0)",
        "plot_bgcolor": "rgba(30, 41, 59, 0.6)",
        "font": {"color": "#f1f5f9", "family": "Inter, 'Segoe UI', system-ui, sans-serif", "size": 12},
        "title": {"font": {"size": 20, "color": "#f8fafc"}, "x": 0.5, "xanchor": "center"},
        "xaxis": {
            "gridcolor": "rgba(148, 163, 184, 0.15)",
            "zeroline": False,
            "tickfont": {"color": "#94a3b8", "size": 11},
            "linecolor": "rgba(148, 163, 184, 0.3)",
            "mirror": True,
        },
        "yaxis": {
            "gridcolor": "rgba(148, 163, 184, 0.15)",
            "zeroline": False,
            "tickfont": {"color": "#94a3b8", "size": 11},
            "linecolor": "rgba(148, 163, 184, 0.3)",
            "mirror": True,
        },
        "legend": {
            "bgcolor": "rgba(30, 41, 59, 0.9)",
            "bordercolor": "rgba(148, 163, 184, 0.3)",
            "borderwidth": 1,
            "font": {"color": "#e2e8f0", "size": 11},
        },
        "margin": {"t": 60, "r": 30, "b": 50, "l": 70},
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor": "rgba(30, 41, 59, 0.95)",
            "font": {"color": "#f8fafc", "size": 12},
            "bordercolor": "rgba(148, 163, 184, 0.4)",
        },
    }
}


def calc_rsi(series, period=14):
    """Relative Strength Index (RSI) - Wilder's smoothing."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_macd(series, fast=12, slow=26, signal=9):
    """MACD: line, signal, histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def create_chart_html(chart_df, symbol):
    """Generate multi-panel chart: Price + RSI + MACD (TradingView style)."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.55, 0.2, 0.25],
        subplot_titles=(
            f"{symbol.upper()} — Price & Moving Averages",
            "RSI (14)",
            "MACD (12, 26, 9)",
        ),
    )

    # Row 1: Price chart
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["Close"],
            name="Close",
            line=dict(color="#10b981", width=2.5, shape="spline", smoothing=0.3),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.2)",
            hovertemplate="<b>Close</b>: $%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["MA10"],
            name="MA 10",
            line=dict(color="#f59e0b", width=1.5, dash="dot"),
            hovertemplate="<b>MA 10</b>: $%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["MA50"],
            name="MA 50",
            line=dict(color="#8b5cf6", width=1.5, dash="dash"),
            hovertemplate="<b>MA 50</b>: $%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Row 2: RSI
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["RSI"],
            name="RSI",
            line=dict(color="#06b6d4", width=2),
            hovertemplate="<b>RSI</b>: %{y:.1f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(239, 68, 68, 0.6)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(34, 197, 94, 0.6)", row=2, col=1)

    # Row 3: MACD
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in chart_df["MACD_Hist"]]
    fig.add_trace(
        go.Bar(
            x=chart_df.index,
            y=chart_df["MACD_Hist"],
            name="MACD Hist",
            marker_color=colors,
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["MACD"],
            name="MACD",
            line=dict(color="#f59e0b", width=1.5),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["MACD_Signal"],
            name="Signal",
            line=dict(color="#8b5cf6", width=1, dash="dot"),
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=720,
        margin=dict(b=60),
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    # Show dates on x-axis (bottom subplot)
    fig.update_xaxes(
        tickformat="%b %d\n%Y",
        tickangle=0,
        showgrid=True,
        row=3,
        col=1,
    )

    return fig.to_html(
        include_plotlyjs="cdn",
        div_id="stockChart",
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "responsive": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )


def get_prediction(symbol):
    df = yf.download(symbol, start="2018-01-01", multi_level_index=False)

    df["Return"] = df["Close"].pct_change()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = calc_rsi(df["Close"])
    macd_line, signal_line, histogram = calc_macd(df["Close"])
    df["MACD"] = macd_line
    df["MACD_Signal"] = signal_line
    df["MACD_Hist"] = histogram

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.dropna(inplace=True)

    X = df[["Return", "MA10", "MA50", "RSI", "MACD"]]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    latest = X.iloc[-1].values.reshape(1, -1)
    prediction = model.predict(latest)[0]
    probability = model.predict_proba(latest)[0][prediction]

    chart_df = df.tail(60)
    chart_html = create_chart_html(chart_df, symbol)

    return prediction, probability, accuracy, chart_html


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    accuracy = None
    chart_html = None
    symbol = ""

    if request.method == "POST":
        symbol = request.form["symbol"]
        prediction, probability, accuracy, chart_html = get_prediction(symbol)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        accuracy=accuracy,
        chart_html=chart_html,
        symbol=symbol,
    )


if __name__=="__main__":
    app.run(debug=True)