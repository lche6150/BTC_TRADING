"""
Enhanced Triple Barrier Labeling System for BTC/USDT Trading
Based on 2024 research with genetic algorithm optimization

Implementation of the advanced triple barrier method that captures:
- Dynamic volatility-adjusted barriers
- Path-dependent price movements
- Optimal holding periods for cryptocurrency trading
- Multiple risk/reward profiles (HRHP and LRLP)

Features:
- Adaptive volatility regime detection
- Genetic algorithm parameter optimization
- Multiple labeling strategies
- Performance-optimized calculations
- Comprehensive evaluation metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from numba import jit
import warnings
warnings.filterwarnings('ignore')

# Genetic Algorithm imports
from deap import base, creator, tools, algorithms
import random
from tqdm import tqdm

@dataclass
class BarrierConfig:
    """Configuration for triple barrier labeling"""
    profit_threshold: float  # Take-profit threshold (% of volatility)
    stop_threshold: float    # Stop-loss threshold (% of volatility)
    max_holding_periods: int # Maximum holding time (in 15min periods)
    vol_lookback: int       # Volatility calculation window
    vol_multiplier: float   # Volatility scaling factor
    min_return_threshold: float = 0.001  # Minimum return to consider (0.1%)

@dataclass
class LabelingResults:
    """Results from triple barrier labeling"""
    labels: pd.Series
    barrier_hits: pd.Series  # Which barrier was hit first
    holding_periods: pd.Series  # Actual holding periods
    returns: pd.Series  # Realized returns
    hit_rates: Dict[str, float]  # Hit rates for each barrier
    avg_returns: Dict[str, float]  # Average returns by label
    label_distribution: Dict[int, float]  # Distribution of labels

class EnhancedTripleBarrierLabeler:
    """
    Enhanced Triple Barrier Labeling with genetic algorithm optimization
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.logger = logging.getLogger(__name__)
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Genetic algorithm setup
        self._setup_genetic_algorithm()

    def _setup_genetic_algorithm(self):
        """Setup genetic algorithm components"""
        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize fitness
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # Genetic algorithm parameters based on our BTC analysis
        # Profit threshold: 0.5x to 3.0x volatility (based on our 95th percentile analysis)
        self.toolbox.register("profit_threshold", random.uniform, 0.5, 3.0)
        # Stop threshold: 0.5x to 3.0x volatility
        self.toolbox.register("stop_threshold", random.uniform, 0.5, 3.0)
        # Max holding: 4 to 96 periods (1h to 24h based on our analysis)
        self.toolbox.register("max_holding", random.randint, 4, 96)
        # Volatility lookback: 24 to 192 periods (6h to 48h)
        self.toolbox.register("vol_lookback", random.randint, 24, 192)
        # Volatility multiplier: 1.0 to 2.5
        self.toolbox.register("vol_multiplier", random.uniform, 1.0, 2.5)

        # Create individual
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                             (self.toolbox.profit_threshold, self.toolbox.stop_threshold,
                              self.toolbox.max_holding, self.toolbox.vol_lookback,
                              self.toolbox.vol_multiplier), n=1)

        # Population
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate_individual, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self._evaluate_individual)

    def create_labels(
        self,
        df: pd.DataFrame,
        config: Optional[BarrierConfig] = None,
        optimize_parameters: bool = True
    ) -> LabelingResults:
        """
        Create triple barrier labels with optional genetic algorithm optimization

        Args:
            df: DataFrame with OHLCV data
            config: Manual configuration (if not optimizing)
            optimize_parameters: Whether to use genetic algorithm optimization

        Returns:
            LabelingResults object with labels and performance metrics
        """
        self.logger.info("Starting Enhanced Triple Barrier Labeling...")

        # Prepare data
        df = self._prepare_data(df)

        if optimize_parameters:
            self.logger.info("Optimizing parameters with genetic algorithm...")
            config = self._optimize_parameters(df)
            self.logger.info(f"Optimal parameters: {config}")
        else:
            config = config or self._get_default_config()

        # Apply triple barrier labeling
        results = self._apply_triple_barrier(df, config)

        # Evaluate results
        self._log_results_summary(results)

        return results

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for labeling"""
        df = df.copy()

        # Ensure we have required columns
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)

        # Calculate basic returns
        df['returns'] = df['close'].pct_change()

        return df

    def _get_default_config(self) -> BarrierConfig:
        """Get default configuration based on BTC analysis"""
        return BarrierConfig(
            profit_threshold=1.5,    # 1.5x volatility based on our analysis
            stop_threshold=1.5,      # 1.5x volatility
            max_holding_periods=64,  # 16 hours (64 * 15min)
            vol_lookback=96,         # 24 hours lookback
            vol_multiplier=1.0,      # No additional scaling
            min_return_threshold=0.001  # 0.1% minimum
        )

    def _optimize_parameters(self, df: pd.DataFrame, n_generations: int = 20, population_size: int = 50) -> BarrierConfig:
        """Optimize parameters using genetic algorithm"""

        # Store data for evaluation
        self.df_for_optimization = df

        # Create population
        population = self.toolbox.population(n=population_size)

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Hall of fame
        hall_of_fame = tools.HallOfFame(1)

        # Run genetic algorithm
        self.logger.info(f"Running GA: {n_generations} generations, {population_size} population size")

        with tqdm(total=n_generations, desc="GA Optimization") as pbar:
            def update_progress():
                pbar.update(1)

            # Custom evolution loop to show progress
            population, logbook = self._custom_evolution(
                population, self.toolbox, cxpb=0.5, mutpb=0.2,
                ngen=n_generations, stats=stats, halloffame=hall_of_fame,
                progress_callback=update_progress
            )

        # Get best individual
        best_individual = hall_of_fame[0]
        best_config = BarrierConfig(
            profit_threshold=best_individual[0],
            stop_threshold=best_individual[1],
            max_holding_periods=int(best_individual[2]),
            vol_lookback=int(best_individual[3]),
            vol_multiplier=best_individual[4]
        )

        self.logger.info(f"Best fitness: {best_individual.fitness.values[0]:.4f}")

        return best_config

    def _custom_evolution(self, population, toolbox, cxpb, mutpb, ngen, stats, halloffame, progress_callback):
        """Custom evolution loop with progress tracking"""
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))

            # Vary the pool of individuals
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            # Update progress
            progress_callback()

        return population, logbook

    def _mutate_individual(self, individual, indpb):
        """Custom mutation operator"""
        if random.random() < indpb:
            # Profit threshold: 0.5 to 3.0
            individual[0] = max(0.5, min(3.0, individual[0] + random.gauss(0, 0.2)))
        if random.random() < indpb:
            # Stop threshold: 0.5 to 3.0
            individual[1] = max(0.5, min(3.0, individual[1] + random.gauss(0, 0.2)))
        if random.random() < indpb:
            # Max holding: 4 to 96
            individual[2] = max(4, min(96, int(individual[2] + random.gauss(0, 5))))
        if random.random() < indpb:
            # Vol lookback: 24 to 192
            individual[3] = max(24, min(192, int(individual[3] + random.gauss(0, 10))))
        if random.random() < indpb:
            # Vol multiplier: 1.0 to 2.5
            individual[4] = max(1.0, min(2.5, individual[4] + random.gauss(0, 0.1)))

        return individual,

    def _evaluate_individual(self, individual):
        """Evaluate individual fitness"""
        config = BarrierConfig(
            profit_threshold=individual[0],
            stop_threshold=individual[1],
            max_holding_periods=int(individual[2]),
            vol_lookback=int(individual[3]),
            vol_multiplier=individual[4]
        )

        try:
            # Apply triple barrier with this configuration
            results = self._apply_triple_barrier(self.df_for_optimization, config)

            # Calculate fitness as combination of metrics
            fitness = self._calculate_fitness(results)

            return fitness,
        except Exception as e:
            # Return poor fitness for invalid configurations
            return -1.0,

    def _calculate_fitness(self, results: LabelingResults) -> float:
        """
        Calculate fitness score for binary classification (UP=1, DOWN=0)
        Based on directional accuracy and Sharpe ratio
        """
        # Component 1: Directional accuracy (50% weight)
        returns = results.returns.dropna()
        labels = results.labels.dropna()

        if len(returns) == 0:
            accuracy_score = 0
        else:
            # For UP signals (1), we want positive returns
            up_mask = labels == 1
            up_accuracy = (returns[up_mask] > 0).mean() if up_mask.any() else 0.5

            # For DOWN signals (0), we want negative returns
            down_mask = labels == 0
            down_accuracy = (returns[down_mask] < 0).mean() if down_mask.any() else 0.5

            # Overall directional accuracy
            accuracy_score = (up_accuracy + down_accuracy) / 2

        # Component 2: Sharpe ratio of realized returns (30% weight)
        if len(returns) == 0 or returns.std() == 0:
            sharpe = -1
        else:
            sharpe = returns.mean() / returns.std() * np.sqrt(24 * 4)  # Annualized Sharpe

        # Component 3: Label balance (20% weight)
        label_dist = results.label_distribution
        total_labels = sum(label_dist.values())
        if total_labels == 0:
            balance_score = 0
        else:
            # Calculate proportions for UP/DOWN
            up_prop = label_dist.get(1, 0) / total_labels
            down_prop = label_dist.get(0, 0) / total_labels

            # Penalize extreme imbalance (ideal is close to 50/50)
            balance_score = 1.0 - abs(up_prop - 0.5) * 2  # Convert to 0-1 scale

        # Combined fitness score
        fitness = (
            0.50 * min(1.0, accuracy_score) +  # Cap at 1.0
            0.30 * min(1.0, max(0.0, (sharpe + 3) / 6)) +  # Normalize Sharpe from -3:3 to 0:1
            0.20 * balance_score
        )

        return fitness

    def _apply_triple_barrier(self, df: pd.DataFrame, config: BarrierConfig) -> LabelingResults:
        """Apply triple barrier labeling to the dataset using ternary classification"""

        n = len(df)
        labels = np.full(n, np.nan)
        barrier_hits = np.full(n, '', dtype='U10')
        holding_periods = np.full(n, np.nan)
        returns = np.full(n, np.nan)

        # Calculate rolling volatility for dynamic thresholds
        df['volatility'] = df['returns'].rolling(window=config.vol_lookback).std() * config.vol_multiplier

        # Process each potential entry point
        for i in range(config.vol_lookback, n - config.max_holding_periods):
            if pd.isna(df['volatility'].iloc[i]) or df['volatility'].iloc[i] == 0:
                continue

            entry_price = df['close'].iloc[i]
            vol = df['volatility'].iloc[i]

            # Calculate volatility-adjusted dynamic thresholds
            # Buy threshold: expect significant upward movement
            buy_threshold = config.profit_threshold * vol
            # Sell threshold: expect significant downward movement
            sell_threshold = config.stop_threshold * vol

            # Look forward to find barrier hit for binary classification
            label, hit_type, hold_period, realized_return = self._check_barrier_hit(
                df, i, upper_barrier, lower_barrier, config.max_holding_periods, entry_price
            )

            # Only assign if return magnitude is significant
            if abs(realized_return) >= config.min_return_threshold:
                labels[i] = label
                barrier_hits[i] = hit_type
                holding_periods[i] = hold_period
                returns[i] = realized_return

        # Create results
        results = LabelingResults(
            labels=pd.Series(labels, index=df.index),
            barrier_hits=pd.Series(barrier_hits, index=df.index),
            holding_periods=pd.Series(holding_periods, index=df.index),
            returns=pd.Series(returns, index=df.index),
            hit_rates=self._calculate_hit_rates(barrier_hits),
            avg_returns=self._calculate_avg_returns(labels, returns),
            label_distribution=self._calculate_label_distribution(labels)
        )

        return results

    @staticmethod
    @jit(nopython=True)
    def _check_barrier_hit_numba(highs, lows, closes, start_idx, upper_barrier, lower_barrier, max_periods):
        """
        Numba-optimized binary classification for UP/DOWN prediction
        Returns: UP=1, DOWN=0 based on future price movement
        """
        entry_price = closes[start_idx]

        # Look forward for the specified period
        for j in range(1, min(max_periods + 1, len(highs) - start_idx)):
            current_idx = start_idx + j

            # Check if upper barrier hit (UP movement)
            if highs[current_idx] >= upper_barrier:
                exit_price = upper_barrier
                return_pct = (exit_price / entry_price) - 1
                return 1, 'up', j, return_pct  # UP = 1

            # Check if lower barrier hit (DOWN movement)
            if lows[current_idx] <= lower_barrier:
                exit_price = lower_barrier
                return_pct = (exit_price / entry_price) - 1
                return 0, 'down', j, return_pct  # DOWN = 0

        # Time barrier hit - classify based on final direction
        final_idx = min(start_idx + max_periods, len(closes) - 1)
        exit_price = closes[final_idx]
        return_pct = (exit_price / entry_price) - 1

        # Binary classification: UP=1 if positive return, DOWN=0 if negative
        label = 1 if return_pct > 0 else 0
        return label, 'time', max_periods, return_pct

    def _check_barrier_hit(self, df, start_idx, upper_barrier, lower_barrier, max_periods, entry_price):
        """Check which barrier gets hit first for binary UP/DOWN classification"""
        # Use numba-optimized version
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        return self._check_barrier_hit_numba(highs, lows, closes, start_idx, upper_barrier, lower_barrier, max_periods)

    def _calculate_hit_rates(self, barrier_hits):
        """Calculate hit rates for each barrier type"""
        valid_hits = barrier_hits[barrier_hits != '']
        total = len(valid_hits)

        if total == 0:
            return {'up': 0.0, 'down': 0.0, 'time': 0.0}

        return {
            'up': (valid_hits == 'up').sum() / total,
            'down': (valid_hits == 'down').sum() / total,
            'time': (valid_hits == 'time').sum() / total
        }

    def _calculate_avg_returns(self, labels, returns):
        """Calculate average returns by label (UP=1, DOWN=0)"""
        valid_mask = ~(pd.isna(labels) | pd.isna(returns))
        valid_labels = labels[valid_mask]
        valid_returns = returns[valid_mask]

        if len(valid_returns) == 0:
            return {1: 0.0, 0: 0.0}

        return {
            1: valid_returns[valid_labels == 1].mean() if (valid_labels == 1).any() else 0.0,  # UP
            0: valid_returns[valid_labels == 0].mean() if (valid_labels == 0).any() else 0.0   # DOWN
        }

    def _calculate_label_distribution(self, labels):
        """Calculate distribution of labels (UP=1, DOWN=0)"""
        valid_labels = labels[~pd.isna(labels)]

        if len(valid_labels) == 0:
            return {1: 0, 0: 0}

        unique, counts = np.unique(valid_labels, return_counts=True)
        distribution = dict(zip(unique.astype(int), counts))

        # Ensure all labels are represented
        for label in [1, 0]:  # UP, DOWN
            if label not in distribution:
                distribution[label] = 0

        return distribution

    def _log_results_summary(self, results: LabelingResults):
        """Log summary of labeling results"""
        self.logger.info("Triple Barrier Labeling Results:")
        self.logger.info(f"Label Distribution: {results.label_distribution}")
        self.logger.info(f"Hit Rates: {results.hit_rates}")
        self.logger.info(f"Average Returns: {results.avg_returns}")

        valid_labels = results.labels.dropna()
        if len(valid_labels) > 0:
            balance = min(results.label_distribution.values()) / max(results.label_distribution.values())
            self.logger.info(f"Label Balance Score: {balance:.3f}")

        valid_returns = results.returns.dropna()
        if len(valid_returns) > 0:
            sharpe = valid_returns.mean() / valid_returns.std() * np.sqrt(24 * 4)
            self.logger.info(f"Annualized Sharpe Ratio: {sharpe:.3f}")

    def create_multiple_strategies(self, df: pd.DataFrame) -> Dict[str, LabelingResults]:
        """
        Create multiple labeling strategies as suggested in 2024 research:
        - HRHP (High Risk High Profit)
        - LRLP (Low Risk Low Profit)
        - Balanced
        """

        self.logger.info("Creating multiple labeling strategies...")

        # Prepare data first
        df = self._prepare_data(df)

        strategies = {}

        # Strategy 1: High Risk High Profit (HRHP)
        hrhp_config = BarrierConfig(
            profit_threshold=2.5,    # Higher profit target
            stop_threshold=2.5,      # Higher stop loss
            max_holding_periods=16,  # Shorter holding period (4 hours)
            vol_lookback=48,         # Shorter volatility window
            vol_multiplier=1.5       # Higher volatility scaling
        )
        strategies['HRHP'] = self._apply_triple_barrier(df, hrhp_config)

        # Strategy 2: Low Risk Low Profit (LRLP)
        lrlp_config = BarrierConfig(
            profit_threshold=1.0,    # Lower profit target
            stop_threshold=1.0,      # Lower stop loss
            max_holding_periods=96,  # Longer holding period (24 hours)
            vol_lookback=192,        # Longer volatility window
            vol_multiplier=0.8       # Lower volatility scaling
        )
        strategies['LRLP'] = self._apply_triple_barrier(df, lrlp_config)

        # Strategy 3: Optimized (using genetic algorithm)
        strategies['Optimized'] = self.create_labels(df, optimize_parameters=True)

        # Log comparison
        self.logger.info("Strategy Comparison:")
        for name, results in strategies.items():
            valid_returns = results.returns.dropna()
            if len(valid_returns) > 0:
                sharpe = valid_returns.mean() / valid_returns.std() * np.sqrt(24 * 4)
                hit_rate = (valid_returns > 0).mean()
                self.logger.info(f"{name}: Sharpe={sharpe:.3f}, Hit Rate={hit_rate:.3f}, Labels={sum(results.label_distribution.values())}")

        return strategies