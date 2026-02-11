"""
Cox-Ingersoll-Ross (CIR) Interest Rate Model

This module implements the CIR model for interest rate dynamics:
dr_t = κ(θ - r_t)dt + σ√(r_t)dW_t

where:
- κ (kappa): speed of mean reversion
- θ (theta): long-term mean level
- σ (sigma): volatility parameter
- r_t: interest rate at time t
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class CIRModel:
    """
    Cox-Ingersoll-Ross interest rate model with automatic parameter calibration
    and synthetic data generation.
    """
    
    def __init__(self, historical_data=None):
        """
        Initialize the CIR model.
        
        Parameters:
        -----------
        historical_data : pd.DataFrame, optional
            DataFrame with columns 'observation_date' and 'interest_rate'
            If provided, parameters will be calibrated from this data
        """
        self.kappa = None
        self.theta = None
        self.sigma = None
        self.historical_data = None
        
        if historical_data is not None:
            self.load_historical_data(historical_data)
    
    def load_historical_data(self, data):
        """
        Load historical interest rate data.
        
        Parameters:
        -----------
        data : pd.DataFrame or str
            DataFrame with columns 'observation_date' and interest rate column,
            or path to CSV file
        """
        if isinstance(data, str):
            df = pd.read_csv(data)
            df['observation_date'] = pd.to_datetime(df['observation_date'])
            # Assume second column is interest rate
            rate_col = df.columns[1]
            df = df.dropna(subset=[rate_col])
            df['interest_rate'] = df[rate_col] / 100  # Convert percentage to decimal
            self.historical_data = df[['observation_date', 'interest_rate']].copy()
        else:
            self.historical_data = data.copy()
        
        print(f"Loaded {len(self.historical_data)} data points")
        print(f"Date range: {self.historical_data['observation_date'].min()} to {self.historical_data['observation_date'].max()}")
        print(f"Mean rate: {self.historical_data['interest_rate'].mean():.4%}")
    
    def calibrate(self, method='mle'):
        """
        Calibrate CIR model parameters from historical data.
        
        Parameters:
        -----------
        method : str
            Calibration method: 'mle' (Maximum Likelihood Estimation) or 'moments'
            
        Returns:
        --------
        dict : Calibrated parameters {kappa, theta, sigma}
        """
        if self.historical_data is None:
            raise ValueError("No historical data loaded. Use load_historical_data() first.")
        
        rates = self.historical_data['interest_rate'].values
        
        if method == 'moments':
            return self._calibrate_moments(rates)
        elif method == 'mle':
            return self._calibrate_mle(rates)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    
    def _calibrate_moments(self, rates):
        """
        Calibrate using method of moments (simpler but less accurate).
        """
        # Calculate sample statistics
        mean_r = np.mean(rates)
        var_r = np.var(rates)
        
        # Calculate differences for AR(1) estimation
        dr = np.diff(rates)
        r_lag = rates[:-1]
        
        # Estimate using discrete approximation
        # dr ≈ κ(θ - r)Δt + σ√(r)√(Δt)ε
        dt = 1/252  # Daily data, assuming 252 trading days per year
        
        # Use linear regression: dr = a + b*r + error
        from scipy import stats
        slope, intercept, _, _, _ = stats.linregress(r_lag, dr)
        
        # Extract parameters
        kappa = -slope / dt
        theta = mean_r
        
        # Estimate sigma from residuals
        predicted_dr = intercept + slope * r_lag
        residuals = dr - predicted_dr
        sigma = np.std(residuals) / np.sqrt(dt * np.mean(r_lag))
        
        # Ensure positive parameters
        kappa = max(abs(kappa), 0.1)
        theta = max(theta, 0.001)
        sigma = max(abs(sigma), 0.01)
        
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        
        print(f"\nCalibrated parameters (Method of Moments):")
        print(f"  κ (kappa): {self.kappa:.4f} - mean reversion speed")
        print(f"  θ (theta): {self.theta:.4%} - long-term mean")
        print(f"  σ (sigma): {self.sigma:.4f} - volatility")
        print(f"  Feller condition (2κθ > σ²): {2*self.kappa*self.theta:.6f} > {self.sigma**2:.6f} = {2*self.kappa*self.theta > self.sigma**2}")
        
        return {'kappa': self.kappa, 'theta': self.theta, 'sigma': self.sigma}
    
    def _calibrate_mle(self, rates):
        """
        Calibrate using Maximum Likelihood Estimation (more accurate).
        Uses Euler discretization for likelihood computation.
        """
        dt = 1/252  # Daily data
        
        def neg_log_likelihood(params):
            """Negative log-likelihood for CIR model."""
            kappa, theta, sigma = params
            
            # Ensure positive parameters
            if kappa <= 0 or theta <= 0 or sigma <= 0:
                return 1e10
            
            # Check Feller condition (loosely)
            if 2 * kappa * theta < 0.5 * sigma**2:
                return 1e10
            
            # Calculate likelihood using Euler approximation
            r = rates[:-1]
            r_next = rates[1:]
            
            # Prevent negative or zero rates in calculations
            r = np.maximum(r, 1e-6)
            
            # Expected change
            mu = r + kappa * (theta - r) * dt
            
            # Variance
            var = sigma**2 * r * dt
            var = np.maximum(var, 1e-10)  # Prevent division by zero
            
            # Log-likelihood (Gaussian approximation)
            log_lik = -0.5 * np.sum(np.log(2 * np.pi * var) + (r_next - mu)**2 / var)
            
            return -log_lik
        
        # Initial guess from moments
        mean_r = np.mean(rates)
        var_r = np.var(rates)
        
        x0 = [0.5, mean_r, 0.2]  # Initial guess [kappa, theta, sigma]
        
        # Bounds
        bounds = [(0.01, 5.0), (0.001, 0.2), (0.01, 1.0)]
        
        # Optimize
        result = minimize(neg_log_likelihood, x0, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            self.kappa, self.theta, self.sigma = result.x
            
            print(f"\nCalibrated parameters (Maximum Likelihood):")
            print(f"  κ (kappa): {self.kappa:.4f} - mean reversion speed")
            print(f"  θ (theta): {self.theta:.4%} - long-term mean")
            print(f"  σ (sigma): {self.sigma:.4f} - volatility")
            print(f"  Feller condition (2κθ > σ²): {2*self.kappa*self.theta:.6f} > {self.sigma**2:.6f} = {2*self.kappa*self.theta > self.sigma**2}")
            
            return {'kappa': self.kappa, 'theta': self.theta, 'sigma': self.sigma}
        else:
            print("MLE optimization failed, falling back to method of moments")
            return self._calibrate_moments(rates)
    
    def simulate(self, r0=0.03, T=1, n_steps=252, n_paths=1, random_seed=None):
        """
        Simulate interest rate paths using the CIR model.
        
        Parameters:
        -----------
        r0 : float
            Initial interest rate (default 0.03 = 3%)
        T : float
            Time horizon in years (default 1)
        n_steps : int
            Number of time steps (default 252 for daily over 1 year)
        n_paths : int
            Number of paths to simulate (default 1)
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray : Array of shape (n_steps+1, n_paths) with simulated rates
        """
        if self.kappa is None or self.theta is None or self.sigma is None:
            raise ValueError("Model parameters not set. Run calibrate() first or set parameters manually.")
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        dt = T / n_steps
        
        # Initialize rate paths
        rates = np.zeros((n_steps + 1, n_paths))
        rates[0, :] = r0
        
        # Simulate using Euler-Maruyama scheme with absorption at zero
        for i in range(n_steps):
            r = rates[i, :]
            
            # Ensure non-negative rates
            r = np.maximum(r, 0)
            
            # Generate random shocks
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            
            # CIR dynamics: dr = κ(θ - r)dt + σ√r dW
            drift = self.kappa * (self.theta - r) * dt
            diffusion = self.sigma * np.sqrt(r) * dW
            
            rates[i + 1, :] = r + drift + diffusion
            
            # Absorb at zero (reflection or truncation)
            rates[i + 1, :] = np.maximum(rates[i + 1, :], 0)
        
        return rates
    
    def generate_synthetic_data(self, r0=0.03, years=1, start_date=None, random_seed=None):
        """
        Generate a synthetic interest rate DataFrame for use in option pricing.
        
        Parameters:
        -----------
        r0 : float
            Initial interest rate (default 0.03 = 3%)
        years : float
            Time horizon in years (default 1)
        start_date : str or datetime, optional
            Starting date for the synthetic data (default: today)
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        pd.DataFrame : DataFrame with columns ['date', 'interest_rate']
        """
        if start_date is None:
            start_date = datetime.today()
        elif isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        
        # Calculate number of business days
        n_days = int(years * 252)
        
        # Simulate rates
        rates = self.simulate(r0=r0, T=years, n_steps=n_days, n_paths=1, random_seed=random_seed)
        
        # Create date range (business days only)
        dates = pd.bdate_range(start=start_date, periods=n_days + 1)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'interest_rate': rates[:, 0]
        })
        
        print(f"\nGenerated synthetic data:")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Number of observations: {len(df)}")
        print(f"  Initial rate: {df['interest_rate'].iloc[0]:.4%}")
        print(f"  Final rate: {df['interest_rate'].iloc[-1]:.4%}")
        print(f"  Mean rate: {df['interest_rate'].mean():.4%}")
        print(f"  Min rate: {df['interest_rate'].min():.4%}")
        print(f"  Max rate: {df['interest_rate'].max():.4%}")
        
        return df
    
    def set_parameters(self, kappa, theta, sigma):
        """
        Manually set CIR model parameters.
        
        Parameters:
        -----------
        kappa : float
            Mean reversion speed
        theta : float
            Long-term mean level
        sigma : float
            Volatility parameter
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        
        print(f"Parameters set manually:")
        print(f"  κ (kappa): {self.kappa:.4f}")
        print(f"  θ (theta): {self.theta:.4%}")
        print(f"  σ (sigma): {self.sigma:.4f}")
        print(f"  Feller condition (2κθ > σ²): {2*self.kappa*self.theta > self.sigma**2}")
    
    def plot_simulation(self, r0=0.03, years=1, n_paths=5, random_seed=None):
        """
        Plot simulated interest rate paths.
        
        Parameters:
        -----------
        r0 : float
            Initial interest rate
        years : float
            Time horizon in years
        n_paths : int
            Number of paths to plot
        random_seed : int, optional
            Random seed for reproducibility
        """
        import matplotlib.pyplot as plt
        
        n_steps = int(years * 252)
        rates = self.simulate(r0=r0, T=years, n_steps=n_steps, n_paths=n_paths, random_seed=random_seed)
        
        time_grid = np.linspace(0, years, n_steps + 1)
        
        plt.figure(figsize=(12, 6))
        for i in range(n_paths):
            plt.plot(time_grid, rates[:, i] * 100, alpha=0.7, linewidth=1.5)
        
        plt.axhline(y=self.theta * 100, color='r', linestyle='--', label=f'Long-term mean θ = {self.theta:.2%}')
        plt.xlabel('Time (years)', fontsize=12)
        plt.ylabel('Interest Rate (%)', fontsize=12)
        plt.title(f'CIR Model Simulation\nκ={self.kappa:.3f}, θ={self.theta:.2%}, σ={self.sigma:.3f}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()


# Example usage and testing
if __name__ == "__main__":
    # Load historical data
    print("="*70)
    print("CIR MODEL EXAMPLE")
    print("="*70)
    
    cir = CIRModel('/mnt/user-data/uploads/IUDSOIA.csv')
    
    # Calibrate parameters
    params = cir.calibrate(method='mle')
    
    # Generate synthetic data
    print("\n" + "="*70)
    synthetic_df = cir.generate_synthetic_data(r0=0.03, years=2, random_seed=42)
    
    print("\n" + "="*70)
    print("First 10 rows of synthetic data:")
    print(synthetic_df.head(10))
    
    print("\nLast 10 rows of synthetic data:")
    print(synthetic_df.tail(10))
