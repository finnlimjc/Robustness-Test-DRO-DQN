import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from old_src.load_model import *
from old_src.price_data import YahooFinance
from old_src.evaluation import *

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'

class ParamsSelector:
    def __init__(self):
        pass
    
    def select_model(self, header:str):
        folders = [f for f in os.listdir('./models')]
        selected_model = st.selectbox(header, folders, None)
        return selected_model
    
    def select_dates(self):
        st.subheader("📅 Select Date Range")
        default_start = date(1995, 1, 1)
        default_end = date(2025, 10, 31)
        
        start_date = st.date_input("Start Date", default_start, min_value=default_start, max_value=default_end)
        end_date = st.date_input("End Date", default_end, min_value=default_start, max_value=default_end)
        
        if start_date > end_date:
            st.error("Start date must be before end date.")
            return None
        
        # For YahooFinance
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")
        
        return start_date, end_date
    
    def select_finance_params(self):
        st.subheader("💰 Financial Parameters")

        symbol = st.text_input("Ticker (e.g., SPY):", value="^SPX")
        transaction_cost = st.number_input("Transaction Cost (%)", min_value=0.0, max_value=5.0, value=0.05, format="%.2f")
        interest_rate = st.number_input("Risk-free Rate (%)", min_value=0.0, max_value=10.0, value=2.4, format="%.2f")

        symbol = symbol.upper().strip()
        transaction_val, rf_rate = transaction_cost/100, interest_rate/100
        return symbol, transaction_val, rf_rate
    
    def render(self) -> dict:
        with st.sidebar:
            st.header("⚙️ Parameter Selector")
            model_file_name = self.select_model("📂 Choose a model:")
            secondary_model_file_name = self.select_model("📂 Compare to:")
            
            st.divider()
            start_date, end_date = self.select_dates()
            
            st.divider()
            symbol, trans_cost, int_rate = self.select_finance_params()
            
            data = {
                'model_file_name': model_file_name,
                'secondary_model_file_name': secondary_model_file_name,
                'start_date': start_date,
                'end_date': end_date,
                'symbol': symbol,
                'trans_cost': trans_cost,
                'int_rate': int_rate
            }
            
            return data

class RunModel:
    def __init__(self, model_file_name, secondary_model_file_name, symbol, start_date, end_date, trans_cost, int_rate):
        self.model, self.action_values = self._load_model(model_file_name)
        self.model2, self.action_values2 = self._load_model(secondary_model_file_name)
        self.df, self.beta = self._download_price(symbol, start_date, end_date)
        self.trans_cost = trans_cost
        self.int_rate = int_rate
    
    @st.cache_resource
    def _load_model(_self, file_name:str):
        if file_name is None:
            return None, None
        checkpt = load_checkpt(file_name)
        load_model = LoadModel(checkpt)
        robustdqn_agent, action_values = load_model.load_model()
        return robustdqn_agent, action_values
    
    def _download_price(self, symbol, start_date, end_date):
        yf_api = YahooFinance(symbol, start_date, end_date)
        df, beta = yf_api.pipeline()
        return df, beta
    
    def _returns_dist(self, result_df:pd.DataFrame):
        df = result_df.copy()
        df['agent_log_return'] = np.log(df['agent']/ df['agent'].shift(1))
        return_df = df.loc[:, ['log_return', 'agent_log_return']]
        simple_returns = np.exp(return_df) - 1
        return_df = pd.concat([return_df, simple_returns], axis=1)
        return_df.columns = ['buyhold_log_returns', 'agent_log_returns', 'buyhold_returns', 'agent_returns']
        return return_df
    
    def pipeline(self):
        performance = result_df = return_df = secondary_result_df = None
        
        if self.model is not None:
            performance, result_df = simulate_agent_spx(self.df, self.model.q, self.action_values, int_rate=self.int_rate, trans_cost=self.trans_cost)
            return_df = self._returns_dist(result_df)
        if self.model2 is not None:
            _, secondary_result_df = simulate_agent_spx(self.df, self.model2.q, self.action_values, int_rate=self.int_rate, trans_cost=self.trans_cost)
        return performance, result_df, return_df, secondary_result_df
    
class Performance(RunModel):
    def __init__(self, model_file_name, secondary_model_file_name, symbol, start_date, end_date, trans_cost, int_rate):
        super().__init__(model_file_name, secondary_model_file_name, symbol, start_date, end_date, trans_cost, int_rate)
        self.symbol = symbol
        self.performance, self.result_df, self.return_df, self.secondary_result_df = self.pipeline()
    
    def plot_wealth(self):
        """Plot wealth comparison and position allocation"""
        # Color scheme
        AGENT_COLOR = '#FF6B35'  # Vibrant orange
        COMPARE_COLOR = '#000000' # Black
        BUYHOLD_COLOR = '#4ECDC4'  # Teal blue

        result_df = self.result_df.iloc[60:]  # Skip initial history period

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Wealth comparison
        ax1.plot(result_df.index, result_df['spx_normalised'], label='Buy & Hold',
                alpha=0.8, color=BUYHOLD_COLOR, linewidth=2)
        ax1.plot(result_df.index, result_df['agent'], label='Agent',
                alpha=0.8, color=AGENT_COLOR, linewidth=2)
        
        if self.secondary_result_df is not None:
            compare_result = self.secondary_result_df.iloc[60:]
            ax1.plot(result_df.index, compare_result['agent'], label='Comparison Agent',
                alpha=0.8, color=COMPARE_COLOR, linewidth=2, linestyle='--')
        
        ax1.set_ylabel('Normalized Wealth', fontsize=11)
        ax1.set_title('Agent vs Buy & Hold Wealth Over Time', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3, color='#E0E0E0')
        ax1.set_xlabel('')

        # Plot 2: Position over time
        ax2.plot(result_df.index, result_df['position'], color=AGENT_COLOR,
                alpha=0.7, linewidth=1)
        ax2.fill_between(result_df.index, 0, result_df['position'],
                        alpha=0.2, color=AGENT_COLOR)
        ax2.set_ylabel('Position (Allocation)', fontsize=11)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_title('Agent Position Over Time', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, color='#E0E0E0')
        ax2.set_ylim([-1.1, 1.1])
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

        plt.tight_layout()
        return fig
    
    def plot_drawdown(self):
        # Color scheme
        AGENT_COLOR = '#FF6B35'
        BUYHOLD_COLOR = '#4ECDC4'

        result_df = self.result_df.iloc[60:]  # Skip initial history period

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))

        # Plot 1: Drawdown over time (shown as negative values going down)
        ax1.fill_between(result_df.index, 0, result_df['agent_max_drawdown']*100,
                        alpha=0.3, color=AGENT_COLOR, label='Agent')
        ax1.plot(result_df.index, result_df['agent_max_drawdown']*100,
                color=AGENT_COLOR, linewidth=2, alpha=0.8)

        ax1.fill_between(result_df.index, 0, result_df['spx_max_drawdown']*100,
                        alpha=0.2, color=BUYHOLD_COLOR, label='Buy & Hold')
        ax1.plot(result_df.index, result_df['spx_max_drawdown']*100,
                color=BUYHOLD_COLOR, linewidth=2, alpha=0.7)

        # Annotate maximum drawdowns
        agent_max_dd = result_df['agent_max_drawdown'].min()
        spx_max_dd = result_df['spx_max_drawdown'].min()
        agent_max_dd_date = result_df['agent_max_drawdown'].idxmin()
        spx_max_dd_date = result_df['spx_max_drawdown'].idxmin()

        ax1.scatter([agent_max_dd_date], [agent_max_dd*100], color=AGENT_COLOR,
                   s=100, zorder=5, marker='v')
        ax1.annotate(f'{agent_max_dd*100:.1f}%', xy=(agent_max_dd_date, agent_max_dd*100),
                    xytext=(10, -15), textcoords='offset points',
                    fontsize=9, color=AGENT_COLOR, fontweight='bold')

        ax1.scatter([spx_max_dd_date], [spx_max_dd*100], color=BUYHOLD_COLOR,
                   s=100, zorder=5, marker='v')
        ax1.annotate(f'{spx_max_dd*100:.1f}%', xy=(spx_max_dd_date, spx_max_dd*100),
                    xytext=(10, 15), textcoords='offset points',
                    fontsize=9, color=BUYHOLD_COLOR, fontweight='bold')

        ax1.set_ylabel('Drawdown (%)', fontsize=11)
        ax1.set_title('Maximum Drawdown Over Time', fontsize=13, fontweight='bold')
        ax1.legend(loc='lower left', framealpha=0.9)
        ax1.grid(True, alpha=0.3, color='#E0E0E0')
        ax1.set_xlabel('')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Plot 2: Drawdown duration (in days)
        agent_dd_dur = result_df['agent_max_drawdown_dur'].dt.days
        spx_dd_dur = result_df['spx_max_drawdown_dur'].dt.days

        ax2.fill_between(result_df.index, 0, agent_dd_dur,
                        alpha=0.3, color=AGENT_COLOR, label='Agent', step='mid')
        ax2.fill_between(result_df.index, 0, spx_dd_dur,
                        alpha=0.2, color=BUYHOLD_COLOR, label='Buy & Hold', step='mid')

        ax2.set_ylabel('Duration (days)', fontsize=11)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_title('Drawdown Duration', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3, color='#E0E0E0')

        plt.tight_layout()
        return fig

    def display_metrics(self):
        # Calculate daily win rate: % of days agent's return beats buy-hold's return
        return_df = self.return_df.dropna()  # Remove NaN from first day
        win_rate = (return_df['agent_returns'] > return_df['buyhold_returns']).mean() * 100

        # Create metrics dataframe with win rate
        metrics = pd.DataFrame(self.performance)
        metrics.loc['win_rate'] = {'agent': win_rate, 'buy_hold': 50.0}  # Buy-hold always 50% vs itself

        st.subheader("📘 Backtesting Metrics")
        st.dataframe(metrics, use_container_width=True, height='stretch')
    
    def plot_cumulative_returns(self, log_scale=False):
        # Color scheme
        AGENT_COLOR = '#FF6B35'
        BUYHOLD_COLOR = '#4ECDC4'

        result_df = self.result_df.iloc[60:]  # Skip initial history period

        # Calculate cumulative returns (percentage)
        agent_cum_ret = (result_df['agent'] - 1) * 100
        buyhold_cum_ret = (result_df['spx_normalised'] - 1) * 100

        fig, ax = plt.subplots(figsize=(12, 5))

        # Fill between the two lines to show outperformance
        ax.fill_between(result_df.index, buyhold_cum_ret, agent_cum_ret,
                       where=(agent_cum_ret >= buyhold_cum_ret),
                       alpha=0.3, color=AGENT_COLOR, interpolate=True, label='Agent Outperforms')
        ax.fill_between(result_df.index, buyhold_cum_ret, agent_cum_ret,
                       where=(agent_cum_ret < buyhold_cum_ret),
                       alpha=0.3, color=BUYHOLD_COLOR, interpolate=True, label='Buy & Hold Outperforms')

        # Plot lines on top of fills
        ax.plot(result_df.index, buyhold_cum_ret, label='Buy & Hold',
               color=BUYHOLD_COLOR, linewidth=2, alpha=0.9)
        ax.plot(result_df.index, agent_cum_ret, label='Agent',
               color=AGENT_COLOR, linewidth=2, alpha=0.9)

        if log_scale:
            ax.set_yscale('log')
            ax.set_ylabel('Cumulative Return (%, log scale)', fontsize=11)
        else:
            ax.set_ylabel('Cumulative Return (%)', fontsize=11)

        ax.set_xlabel('Date', fontsize=11)
        ax.set_title('Cumulative Returns Over Time', fontsize=13, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, color='#E0E0E0')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        return fig

    def plot_rolling_sharpe(self, window=60):
        # Color scheme
        AGENT_COLOR = '#FF6B35'
        BUYHOLD_COLOR = '#4ECDC4'

        # Calculate rolling Sharpe ratio (annualized)
        agent_returns = self.return_df['agent_log_returns'].dropna()
        buyhold_returns = self.return_df['buyhold_log_returns'].dropna()

        agent_rolling_sharpe = (agent_returns.rolling(window).mean() / agent_returns.rolling(window).std()) * np.sqrt(252)
        buyhold_rolling_sharpe = (buyhold_returns.rolling(window).mean() / buyhold_returns.rolling(window).std()) * np.sqrt(252)

        # Drop NaN values from rolling calculations to ensure same length
        agent_rolling_sharpe = agent_rolling_sharpe.dropna()
        buyhold_rolling_sharpe = buyhold_rolling_sharpe.dropna()

        # Align indices - use intersection of both series
        common_index = agent_rolling_sharpe.index.intersection(buyhold_rolling_sharpe.index)
        agent_rolling_sharpe = agent_rolling_sharpe.loc[common_index]
        buyhold_rolling_sharpe = buyhold_rolling_sharpe.loc[common_index]

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(common_index, buyhold_rolling_sharpe.values,
               label='Buy & Hold', color=BUYHOLD_COLOR, linewidth=2, alpha=0.8)
        ax.plot(common_index, agent_rolling_sharpe.values,
               label='Agent', color=AGENT_COLOR, linewidth=2, alpha=0.8)

        # Use values arrays for comparison - now guaranteed same length
        ax.fill_between(common_index, 0, agent_rolling_sharpe.values,
                       where=(agent_rolling_sharpe.values >= buyhold_rolling_sharpe.values),
                       alpha=0.2, color=AGENT_COLOR, interpolate=True)

        ax.set_ylabel('Sharpe Ratio', fontsize=11)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_title(f'{window}-Day Rolling Sharpe Ratio', fontsize=13, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, color='#E0E0E0')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        return fig

    def plot_returns_dist(self):
        # Color scheme
        AGENT_COLOR = '#FF6B35'
        BUYHOLD_COLOR = '#4ECDC4'

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(self.return_df["buyhold_returns"], bins=40, alpha=0.5,
               label="Buy & Hold", color=BUYHOLD_COLOR, edgecolor='white')
        ax.hist(self.return_df["agent_returns"], bins=40, alpha=0.6,
               label="Agent", color=AGENT_COLOR, edgecolor='white')
        ax.set_xlabel("Return", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title("Return Distribution", fontsize=13, fontweight='bold')
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3, color='#E0E0E0', axis='y')
        plt.tight_layout()
        
        text = f"Buy & Hold Kurtosis: {self.return_df['buyhold_returns'].kurtosis():.4f}\n\
            Agent Kurtosis: {self.return_df['agent_returns'].kurtosis():.4f}"
        props = dict(boxstyle='round', facecolor='white', edgecolor='black')
        ax.text(
            0.95, 0.95, text, transform=ax.transAxes,
            fontsize=14, verticalalignment='top', horizontalalignment='right',
            bbox=props
        )
        return fig
        
    def render(self):
        st.set_page_config(layout="wide", page_title="Model Performance Dashboard")

        # Custom CSS for better styling
        st.markdown("""
        <style>
        .stMetric {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stMetric label {
            font-size: 14px !important;
            font-weight: 600 !important;
        }
        .stMetric [data-testid="stMetricValue"] {
            font-size: 24px !important;
        }
        h1, h2, h3 {
            font-weight: 700 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # Header
        st.title("🎯 Model Performance Dashboard")
        st.markdown(rf"### Analysis for **{self.symbol}** ($\beta$ = {self.beta})")

        # Key Performance Indicators at the top
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            agent_final = self.performance['agent']['final_wealth']
            st.metric("Agent Final Wealth", f"{agent_final:.2f}",
                     delta=f"{(agent_final - 1)*100:.1f}%")
        with col2:
            agent_sharpe = self.performance['agent']['sharpe']
            buyhold_sharpe = self.performance['buy_hold']['sharpe']
            st.metric("Agent Sharpe Ratio", f"{agent_sharpe:.3f}",
                     delta=f"{agent_sharpe - buyhold_sharpe:.3f}")
        with col3:
            agent_dd = self.performance['agent']['max_drawdown']
            st.metric("Agent Max Drawdown", f"{agent_dd*100:.2f}%",
                     delta=None, delta_color="inverse")
        with col4:
            return_df = self.return_df.dropna()
            win_rate = (return_df['agent_returns'] > return_df['buyhold_returns']).mean() * 100
            st.metric("Daily Win Rate", f"{win_rate:.1f}%")

        st.divider()

        # Main wealth chart
        st.subheader("💰 Wealth Evolution")
        st.pyplot(self.plot_wealth(), use_container_width=True)

        st.divider()

        # Drawdown and Return Distribution side by side
        st.subheader("📉 Risk Analysis")
        col_left, col_right = st.columns([6, 4])
        with col_left:
            st.pyplot(self.plot_drawdown(), use_container_width=True)
        with col_right:
            st.pyplot(self.plot_returns_dist(), use_container_width=True)

        st.divider()

        # Cumulative returns with log scale toggle
        st.subheader("📈 Performance Metrics Over Time")
        log_scale = st.checkbox("Use log scale for cumulative returns", value=False)
        st.pyplot(self.plot_cumulative_returns(log_scale=log_scale), use_container_width=True)

        # Rolling Sharpe ratio
        st.pyplot(self.plot_rolling_sharpe(), use_container_width=True)

        st.divider()

        # Metrics table
        self.display_metrics()

if __name__ == '__main__':
    selector = ParamsSelector()
    params = selector.render()
    p = Performance(**params)
    p.render()