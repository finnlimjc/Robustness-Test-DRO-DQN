import os
from datetime import date
import streamlit as st
import numpy as np
import pandas as pd

from src.evaluation.backtest import *
from src.evaluation.metrics import *
from src.evaluation.plots import *

@st.cache_data
def load_ticker(ticker:str, dataset_start:str, dataset_end:str) -> pd.DataFrame:
    return load_ticker_data(ticker, dataset_start, dataset_end)

@st.cache_resource
def load_qfunc(config_path:str, checkpoint_path:str) -> dict:
    return load_qfunc_from_checkpoint(config_path, checkpoint_path)

@st.cache_data
def run(config_path:str, checkpoint_path:str, ticker:str, dataset_start:str, dataset_end:str, rf_rate:float, trans_cost:float) -> dict:
    """
    Inputs:
    - config_path: path to runtime config.json
    - checkpoint_path: path to .pt checkpoint file
    - ticker: Yahoo Finance symbol
    - dataset_start: dataset start date string YYYY-MM-DD
    - dataset_end: dataset end date string YYYY-MM-DD
    - rf_rate: annualised risk-free rate
    - trans_cost: transaction cost fraction
    
    Outputs:
        Backtest result dict from run_backtest.
    """
    ckpt = load_qfunc(config_path, checkpoint_path)
    df = load_ticker(ticker, dataset_start, dataset_end)
    return run_backtest(ckpt['qfunc'], ckpt['action_values'], rf_rate, trans_cost, df=df)

class SelectParams:
    def __init__(self, repo:RunsRepository):
        self.repo = repo
    
    def _select_model(self) -> tuple[str, str, str]:
        models = self.repo.get_model_names()
        model = st.selectbox('Model', models, index=None, placeholder='Select a model...')
        
        runtime = None
        checkpoint = None
        
        if model:
            runtimes = self.repo.get_runtime_names(model)
            default_rt_idx = len(runtimes) - 1 if runtimes else None
            runtime = st.selectbox('Runtime', runtimes, index=default_rt_idx)
            
        if model and runtime:
            ckpt_files = self.repo.get_checkpoint_files(model, runtime)
            ckpt_basenames = [os.path.basename(f) for f in ckpt_files]
            default_ckpt_idx = len(ckpt_basenames) - 1 if ckpt_basenames else None
            ckpt_name = st.selectbox('Checkpoint', ckpt_basenames, index=default_ckpt_idx)
            checkpoint = ckpt_files[ckpt_basenames.index(ckpt_name)] if ckpt_name and ckpt_files else None
        
        return model, runtime, checkpoint
    
    def _select_financial_params(self) -> tuple[str, float, float]:
        ticker = st.text_input('Ticker', value='SPY').upper().strip()
        trans_cost = st.number_input('Transaction Cost (%)', min_value=0.0, max_value=5.0, value=0.05, format='%.4f')
        rf_rate = st.number_input('Risk-free Rate (%)', min_value=0.0, max_value=10.0, value=2.4, format='%.2f')
        return ticker, trans_cost/100, rf_rate/100
    
    def _select_dataset_dates(self) -> tuple[str, str, str, str]:
        dataset_start = st.date_input('Dataset Start', value=date(1995, 1, 1))
        dataset_end = st.date_input('Dataset End', value=date(2025, 12, 31))
        return dataset_start, dataset_end
    
    def _select_view_dates(self, dataset_start:date, dataset_end:date) -> tuple[str, str]:
        view_start = st.date_input('View Start', value=dataset_start)
        view_end = st.date_input('View End', value=dataset_end)
        return view_start, view_end

class SingleRunPage(SelectParams):
    def __init__(self, repo):
        super().__init__(repo)
    
    def _sidebar(self) -> dict:
        with st.sidebar:
            st.header('Single Run Parameters')
            model, runtime, checkpoint = self._select_model()
            st.divider()
            
            st.subheader('Financial Parameters')
            ticker, trans_cost, rf_rate = self._select_financial_params()
            st.divider()
            
            st.subheader('Dataset Date Range')
            dataset_start, dataset_end = self._select_dataset_dates()
            st.divider()
            
            st.subheader('View Date Range')
            view_start, view_end = self._select_view_dates(dataset_start, dataset_end)
            
            return {
                'model': model,
                'runtime': runtime,
                'checkpoint': checkpoint,
                'ticker': ticker,
                'trans_cost': trans_cost,
                'rf_rate': rf_rate,
                'dataset_start': dataset_start.strftime('%Y-%m-%d'),
                'dataset_end': dataset_end.strftime('%Y-%m-%d'),
                'view_start': view_start,
                'view_end': view_end,
            }
    
    def render(self):
        # Get parameters from sidebar
        params = self._sidebar()
        if params['model'] is None or params['runtime'] is None or params['checkpoint'] is None:
            st.info('Select a model, runtime, and checkpoint in the sidebar to begin.')
            st.stop()
        config_path = self.repo.get_config_path(params['model'], params['runtime'])
        
        # Run backtest with caching
        with st.spinner('Running backtest...'):
            result = run(
                config_path,
                params['checkpoint'],
                params['ticker'],
                params['dataset_start'],
                params['dataset_end'],
                params['rf_rate'],
                params['trans_cost'],
            )
        
        # Filter results to view date range and compute metrics
        filtered = filter_results(result, params['view_start'], params['view_end'])
        if len(filtered['dates']) == 0:
            st.warning('No data in the selected view date range. Adjust the View Date Range in the sidebar.')
            st.stop()
        agent_metrics = compute_metrics(filtered['agent_log_returns'])
        bm_metrics = compute_metrics(filtered['benchmark_log_returns'])
        
        # Render plots and metrics
        st.title('Single Run Analysis')
        col_ts, col_side = st.columns([7, 4])
        with col_ts:
            st.pyplot(plot_wealth(filtered['dates'], filtered['agent_log_returns'], filtered['benchmark_log_returns']), use_container_width=True)
            st.pyplot(plot_drawdown(filtered['dates'], filtered['agent_log_returns'], filtered['benchmark_log_returns']), use_container_width=True)
            st.pyplot(plot_position(filtered['dates'], filtered['positions']), use_container_width=True)
        
        with col_side:
            st.pyplot(
                plot_return_distribution(filtered['agent_log_returns'], filtered['benchmark_log_returns']),
                use_container_width=True,
            )
            
            st.subheader('Performance Metrics')
            st.dataframe(metrics_df(agent_metrics, bm_metrics), use_container_width=True)
            
            st.subheader('Top 5 Drawdown Episodes')
            try:
                mdd_df = compute_top_drawdowns(filtered['agent_wealth'], filtered['wealth_dates'])
                mdd_df['MDD (%)'] = mdd_df['MDD (%)'].apply(lambda x: f'{x:.2f}%')
                st.dataframe(mdd_df, use_container_width=True, hide_index=True)
            except ValueError:
                st.write('No drawdown episodes in view range.')

class ModelComparisonPage(SelectParams):
    def __init__(self, repo:RunsRepository):
        super().__init__(repo)
    
    def _select_model(self) -> tuple[str, list[str], str]:
        models = self.repo.get_model_names()
        model = st.selectbox('Model', models, index=None, placeholder='Select a model...')
        
        if model:
            runtimes = self.repo.get_runtime_names(model)
            if not runtimes:
                st.warning('No runtime folders found for this model.')
                st.stop()
            runtime = runtimes[-1] # Default to last runtime assuming that all runtimes have the same checkpoint naming convention
            
            ckpt_files = self.repo.get_checkpoint_files(model, runtime)
            ckpt_basenames = [os.path.basename(f) for f in ckpt_files]
            default_ckpt_idx = len(ckpt_basenames) - 1 if ckpt_basenames else None
            ckpt_name = st.selectbox('Checkpoint', ckpt_basenames, index=default_ckpt_idx)
            checkpoint = ckpt_name if ckpt_name else None
        
        return model, runtimes, checkpoint
    
    def _construct_checkpoint_path(self, model:str, runtime:str, checkpoint:str) -> str:
        ckpt_files = self.repo.get_checkpoint_files(model, runtime)
        ckpt_map = {os.path.basename(f): f for f in ckpt_files}
        checkpoint = ckpt_map.get(checkpoint)
        return checkpoint
    
    def _sidebar(self) -> dict:
        with st.sidebar:
            st.header('Model Comparison Parameters')
            model, runtimes, checkpoint = self._select_model()
            st.divider()
            
            st.subheader('Financial Parameters')
            ticker, trans_cost, rf_rate = self._select_financial_params()
            st.divider()
            
            st.subheader('Dataset Date Range')
            dataset_start, dataset_end = self._select_dataset_dates()
            st.divider()
            
            st.subheader('View Date Range')
            view_start, view_end = self._select_view_dates(dataset_start, dataset_end)
            
            return {
                'model': model,
                'runtimes': runtimes,
                'checkpoint': checkpoint,
                'ticker': ticker,
                'trans_cost': trans_cost,
                'rf_rate': rf_rate,
                'dataset_start': dataset_start.strftime('%Y-%m-%d'),
                'dataset_end': dataset_end.strftime('%Y-%m-%d'),
                'view_start': view_start,
                'view_end': view_end,
            }
    
    def _run_backtests(self, params:dict) -> list[dict]:
        runtimes = params['runtimes']
        total_runs = len(runtimes)
        all_results = []
        failed = []
        
        progress = st.progress(0, text='Loading runs...')
        for idx, runtime in enumerate(runtimes):
            config_path = self.repo.get_config_path(params['model'], runtime)
            checkpoint = self._construct_checkpoint_path(params['model'], runtime, params['checkpoint'])
            if checkpoint is None:
                failed.append(f'{runtime}: checkpoint {params["checkpoint"]} not found')
                continue
            
            try:
                result = run(config_path, checkpoint, params['ticker'], params['dataset_start'], params['dataset_end'], params['rf_rate'], params['trans_cost'])
                all_results.append(result)
            except Exception as e:
                failed.append(f'{runtime}: {e}')
            progress.progress((idx + 1)/total_runs, text=f'Loaded {idx + 1}/{total_runs} runs')
        progress.empty()
        
        if failed:
            st.warning(f'Could not load: {", ".join(str(f) for f in failed)}')
        
        if not all_results:
            st.error('No runs loaded successfully.')
            st.stop()
        
        return all_results
    
    def _compute_agent_wealth(self, all_results:list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        agent_wealths = np.stack([r['agent_wealth'] for r in all_results])
        agent_mean_wealth = agent_wealths.mean(axis=0)
        agent_q25_wealth = np.percentile(agent_wealths, 25, axis=0)
        agent_q75_wealth = np.percentile(agent_wealths, 75, axis=0)
        return agent_mean_wealth, agent_q25_wealth, agent_q75_wealth
    
    def render(self):
        params = self._sidebar()
        if not params['model']:
            st.info('Select a model in the sidebar to compare all its runs.')
            st.stop()
        
        # Run backtests for all runtimes of the selected model with caching and filter to view date range
        all_results = self._run_backtests(params)
        filtered_results = [filter_results(r, params['view_start'], params['view_end']) for r in all_results]
        if len(filtered_results[0]['dates']) == 0:
            st.warning('No data in the selected view date range. Adjust the View Date Range in the sidebar.')
            st.stop()
        
        # Data preparation for plots and metrics
        wealth_dates = filtered_results[0]['wealth_dates']
        bm_wealth = filtered_results[0]['benchmark_wealth']
        bm_log_returns = filtered_results[0]['benchmark_log_returns']
        
        agent_mean_wealth, agent_q25_wealth, agent_q75_wealth = self._compute_agent_wealth(filtered_results)
        agent_lr_stack = np.stack([f['agent_log_returns'] for f in filtered_results])
        mean_log_returns = agent_lr_stack.mean(axis=0)
        
        all_agent_metrics = [compute_metrics(f['agent_log_returns']) for f in filtered_results]
        bm_metrics = compute_metrics(bm_log_returns)
        
        # Plots and metrics
        st.title('Model Comparison — All Runs')
        st.write(f'{len(all_results)} runs loaded for **{params["model"]}**')
        st.pyplot(
            plot_wealth_multi(wealth_dates, agent_mean_wealth, agent_q25_wealth, agent_q75_wealth, bm_wealth),
            use_container_width=True,
        )
        st.pyplot(plot_drawdown_multi(wealth_dates, agent_mean_wealth, bm_wealth), use_container_width=True)
        
        col_dist, col_bars = st.columns([2, 3])
        with col_dist:
            st.pyplot(
                plot_return_distribution(mean_log_returns, bm_log_returns),
                use_container_width=True,
            )
        with col_bars:
            st.pyplot(plot_metrics_bars(all_agent_metrics, bm_metrics), use_container_width=True)
        
        st.subheader('Top 5 Drawdown Episodes (Mean Agent)')
        try:
            mdd_df = compute_top_drawdowns(agent_mean_wealth, wealth_dates)
            mdd_df['MDD (%)'] = mdd_df['MDD (%)'].apply(lambda x: f'{x:.2f}%')
            st.dataframe(mdd_df, use_container_width=True, hide_index=True, height=210)
        except ValueError:
            st.write('No drawdown episodes in view range.')

if __name__ == '__main__':
    st.set_page_config(layout='wide', page_title='DRO-DQN Dashboard')
    repo = RunsRepository()
    page = st.sidebar.radio('Page', ['Single Run', 'Model Comparison'])
    
    if page == 'Single Run':
        SingleRunPage(repo).render()
    else:
        ModelComparisonPage(repo).render()