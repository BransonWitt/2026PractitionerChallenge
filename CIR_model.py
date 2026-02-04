import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Tuple, Optional
import os 

cwd = os.getcwd()

class CIRModel:
    """
    Cox-Ingersoll-Ross (CIR) Model for Interest Rate Modeling
    
    The CIR model follows the stochastic differential equation:
    dr(t) = κ(θ - r(t))dt + σ√r(t)dW(t)
    
    where:
    - κ (kappa): speed of mean reversion
    - θ (theta): long-term mean level
    - σ (sigma): volatility parameter
    - r(t): interest rate at time t
    - W(t): Wiener process (Brownian motion)
    """
    
    def __init__(self, kappa: float = None, theta: float = None, sigma: float = None):
        """
        Initialize CIR model with parameters
        
        Parameters:
        -----------
        kappa : float
            Speed of mean reversion
        theta : float
            Long-term mean level
        sigma : float
            Volatility parameter
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        
    def simulate(self, r0: float, T: float, n_steps: int, n_paths: int = 1, 
                 seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate interest rate paths using Euler-Maruyama discretization
        
        Parameters:
        -----------
        r0 : float
            Initial interest rate
        T : float
            Time horizon (in years)
        n_steps : int
            Number of time steps
        n_paths : int
            Number of simulation paths
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray
            Simulated interest rate paths (shape: n_steps+1 x n_paths)
        """
        if seed is not None:
            np.random.seed(seed)
            
        dt = T / n_steps
        rates = np.zeros((n_steps + 1, n_paths))
        rates[0] = r0
        
        for i in range(1, n_steps + 1):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            
            # Euler-Maruyama discretization with max to ensure non-negativity
            drift = self.kappa * (self.theta - rates[i-1])
            diffusion = self.sigma * np.sqrt(np.maximum(rates[i-1], 0))
            
            rates[i] = rates[i-1] + drift * dt + diffusion * dW
            
            # Ensure rates remain non-negative (Feller condition: 2κθ ≥ σ²)
            rates[i] = np.maximum(rates[i], 0)
            
        return rates
    
    def log_likelihood(self, params: np.ndarray, rates: np.ndarray, dt: float) -> float:
        """
        Calculate negative log-likelihood for parameter estimation
        
        Uses the transition density of the CIR process
        
        Parameters:
        -----------
        params : np.ndarray
            Parameters [kappa, theta, sigma]
        rates : np.ndarray
            Historical interest rate data
        dt : float
            Time step between observations
            
        Returns:
        --------
        float
            Negative log-likelihood
        """
        kappa, theta, sigma = params
        
        # Ensure positive parameters
        if kappa <= 0 or theta <= 0 or sigma <= 0:
            return 1e10
            
        n = len(rates) - 1
        log_lik = 0
        
        for i in range(n):
            r_t = rates[i]
            r_next = rates[i + 1]
            
            if r_t <= 0:
                continue
                
            # CIR transition parameters
            c = 2 * kappa / (sigma**2 * (1 - np.exp(-kappa * dt)))
            q = 2 * kappa * theta / sigma**2 - 1
            u = c * r_t * np.exp(-kappa * dt)
            v = c * r_next
            
            # Non-central chi-square parameters
            if v > 0:
                # Approximation of log-likelihood
                log_lik += (np.log(c) + u + v - (q/2) * np.log(v/u) 
                           - 0.5 * (u + v))
            
        return -log_lik
    
    def estimate_parameters(self, rates: np.ndarray, dt: float, 
                          initial_guess: Optional[Tuple[float, float, float]] = None) -> dict:
        """
        Estimate CIR model parameters using Maximum Likelihood Estimation
        
        Parameters:
        -----------
        rates : np.ndarray
            Historical interest rate data
        dt : float
            Time step between observations (in years)
        initial_guess : tuple, optional
            Initial parameter guess (kappa, theta, sigma)
            
        Returns:
        --------
        dict
            Dictionary containing estimated parameters and optimization results
        """
        if initial_guess is None:
            # Use simple moment-based initial estimates
            mean_rate = np.mean(rates)
            std_rate = np.std(rates)
            initial_guess = (0.5, mean_rate, std_rate * 0.5)
        
        # Optimization bounds
        bounds = [(0.01, 5.0),    # kappa
                  (0.001, 0.5),   # theta
                  (0.001, 1.0)]   # sigma
        
        result = minimize(
            self.log_likelihood,
            x0=initial_guess,
            args=(rates, dt),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        self.kappa, self.theta, self.sigma = result.x
        
        return {
            'kappa': self.kappa,
            'theta': self.theta,
            'sigma': self.sigma,
            'success': result.success,
            'log_likelihood': -result.fun,
            'message': result.message
        }
    
    def plot_simulation(self, rates: np.ndarray, time_points: np.ndarray, 
                       historical_rates: Optional[np.ndarray] = None,
                       historical_times: Optional[np.ndarray] = None):
        """
        Plot simulated interest rate paths
        
        Parameters:
        -----------
        rates : np.ndarray
            Simulated rates (from simulate method)
        time_points : np.ndarray
            Time points for simulation
        historical_rates : np.ndarray, optional
            Historical rates to overlay
        historical_times : np.ndarray, optional
            Time points for historical rates
        """
        plt.figure(figsize=(12, 6))
        
        # Plot simulated paths
        if rates.shape[1] > 1:
            # Multiple paths - show mean and confidence interval
            mean_path = np.mean(rates, axis=1)
            std_path = np.std(rates, axis=1)
            
            plt.plot(time_points, mean_path, 'b-', linewidth=2, label='Mean Path')
            plt.fill_between(time_points, 
                           mean_path - 1.96*std_path, 
                           mean_path + 1.96*std_path, 
                           alpha=0.3, label='95% CI')
            
            # Plot some individual paths
            n_sample_paths = min(10, rates.shape[1])
            for i in range(n_sample_paths):
                plt.plot(time_points, rates[:, i], 'gray', alpha=0.3, linewidth=0.5)
        else:
            # Single path
            plt.plot(time_points, rates[:, 0], 'b-', linewidth=2, label='Simulated Path')
        
        # Plot historical data if provided
        if historical_rates is not None and historical_times is not None:
            plt.plot(historical_times, historical_rates, 'ro-', 
                    linewidth=1.5, markersize=4, label='Historical Data')
        
        plt.xlabel('Time (years)', fontsize=12)
        plt.ylabel('Interest Rate', fontsize=12)
        plt.title(f'CIR Model: κ={self.kappa:.4f}, θ={self.theta:.4f}, σ={self.sigma:.4f}', 
                 fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()


# Example usage
if __name__ == "__main__":
    # Example 1: Generate synthetic data and estimate parameters
    print("="*60)
    print("Example 1: Parameter Estimation from Synthetic Data")
    print("="*60)
    
    # True parameters
    true_kappa = 0.5
    true_theta = 0.03
    true_sigma = 0.1
    
    # Generate synthetic data
    true_model = CIRModel(true_kappa, true_theta, true_sigma)
    T = 10  # 10 years
    n_steps = 252 * 10  # Daily data for 10 years
    synthetic_rates = true_model.simulate(r0=0.02, T=T, n_steps=n_steps, seed=42)[:, 0]
    
    print(f"\nTrue parameters:")
    print(f"  κ (kappa) = {true_kappa}")
    print(f"  θ (theta) = {true_theta}")
    print(f"  σ (sigma) = {true_sigma}")
    
    # Estimate parameters
    model = CIRModel()
    dt = T / n_steps
    results = model.estimate_parameters(synthetic_rates, dt)
    
    print(f"\nEstimated parameters:")
    print(f"  κ (kappa) = {results['kappa']:.6f}")
    print(f"  θ (theta) = {results['theta']:.6f}")
    print(f"  σ (sigma) = {results['sigma']:.6f}")
    print(f"  Log-likelihood: {results['log_likelihood']:.2f}")
    print(f"  Optimization success: {results['success']}")
    
    # Example 2: Simulate future paths
    print("\n" + "="*60)
    print("Example 2: Forecasting with Estimated Parameters")
    print("="*60)
    
    forecast_horizon = 2  # 2 years
    n_forecast_steps = 252 * 2  # Daily forecasts
    n_paths = 1000
    
    forecasted_rates = model.simulate(
        r0=synthetic_rates[-1], 
        T=forecast_horizon, 
        n_steps=n_forecast_steps, 
        n_paths=n_paths,
        seed=123
    )
    
    print(f"\nForecasted {n_paths} paths for {forecast_horizon} years")
    print(f"Starting rate: {synthetic_rates[-1]:.4f}")
    print(f"Mean final rate: {np.mean(forecasted_rates[-1]):.4f}")
    print(f"Std final rate: {np.std(forecasted_rates[-1]):.4f}")
    
    # Plot results
    time_points = np.linspace(0, forecast_horizon, n_forecast_steps + 1)
    historical_times = np.linspace(-T, 0, len(synthetic_rates))
    
    model.plot_simulation(
        forecasted_rates, 
        time_points,
        synthetic_rates[-252*2:],  # Last 2 years of historical data
        historical_times[-252*2:]
    )
    
    plt.savefig(f'{cwd}/cir_simulation.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: cir_simulation.png")
    
    # Example 3: Check Feller condition
    print("\n" + "="*60)
    print("Example 3: Model Diagnostics")
    print("="*60)
    
    feller_condition = 2 * model.kappa * model.theta / (model.sigma ** 2)
    print(f"\nFeller condition: 2κθ/σ² = {feller_condition:.4f}")
    print(f"Status: {'✓ Satisfied (≥ 1)' if feller_condition >= 1 else '✗ Not satisfied (< 1)'}")
    print("(Ensures the process stays positive)")