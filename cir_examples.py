"""
CIR Model - Quick Start Guide and Examples
==========================================

This guide shows you how to use the CIR model to generate synthetic interest rates
for your Black-Scholes option pricing model.
"""

from cir_model import CIRModel
import pandas as pd

# ============================================================================
# EXAMPLE 1: Basic Usage - Calibrate and Generate Synthetic Data
# ============================================================================

print("EXAMPLE 1: Basic Usage")
print("="*70)

# Step 1: Initialize the model with historical data
cir = CIRModel('/mnt/user-data/uploads/IUDSOIA.csv')

# Step 2: Calibrate parameters from historical data
cir.calibrate(method='mle')  # or method='moments' for faster but less accurate

# Step 3: Generate synthetic data
synthetic_df = cir.generate_synthetic_data(
    r0=0.03,           # Starting rate: 3%
    years=1,           # 1 year horizon
    start_date='2026-02-04',  # Start date
    random_seed=42     # For reproducibility (optional)
)

print("\nSynthetic data preview:")
print(synthetic_df.head())
print(f"\nShape: {synthetic_df.shape}")

# ============================================================================
# EXAMPLE 2: Different Time Horizons
# ============================================================================

print("\n\nEXAMPLE 2: Different Time Horizons")
print("="*70)

# Generate 6 months of data
synthetic_6m = cir.generate_synthetic_data(r0=0.03, years=0.5, random_seed=123)

# Generate 2 years of data
synthetic_2y = cir.generate_synthetic_data(r0=0.03, years=2, random_seed=123)

# Generate 5 years of data
synthetic_5y = cir.generate_synthetic_data(r0=0.03, years=5, random_seed=123)

print(f"6 months: {len(synthetic_6m)} observations")
print(f"2 years: {len(synthetic_2y)} observations")
print(f"5 years: {len(synthetic_5y)} observations")

# ============================================================================
# EXAMPLE 3: Different Starting Rates
# ============================================================================

print("\n\nEXAMPLE 3: Different Starting Rates")
print("="*70)

# Low rate scenario
low_rate = cir.generate_synthetic_data(r0=0.02, years=1, random_seed=111)
print(f"Starting at 2%: Mean = {low_rate['interest_rate'].mean():.4%}")

# Medium rate scenario  
med_rate = cir.generate_synthetic_data(r0=0.04, years=1, random_seed=111)
print(f"Starting at 4%: Mean = {med_rate['interest_rate'].mean():.4%}")

# High rate scenario
high_rate = cir.generate_synthetic_data(r0=0.06, years=1, random_seed=111)
print(f"Starting at 6%: Mean = {high_rate['interest_rate'].mean():.4%}")

# ============================================================================
# EXAMPLE 4: Integration with Black-Scholes
# ============================================================================

print("\n\nEXAMPLE 4: Integration with Black-Scholes")
print("="*70)

# Generate interest rates for option pricing
df_rates = cir.generate_synthetic_data(r0=0.03, years=1, random_seed=999)

# Example: Get the average interest rate for BS model
avg_rate = df_rates['interest_rate'].mean()
print(f"Average interest rate for BS: {avg_rate:.4%}")

# Or use specific date's rate
rate_at_expiry = df_rates['interest_rate'].iloc[-1]
print(f"Rate at expiry: {rate_at_expiry:.4%}")

# Or interpolate for specific date
print(f"\nFirst 5 rates with dates:")
print(df_rates.head())

# ============================================================================
# EXAMPLE 5: Multiple Scenarios (Monte Carlo)
# ============================================================================

print("\n\nEXAMPLE 5: Multiple Interest Rate Scenarios")
print("="*70)

# Generate 5 different scenarios
scenarios = []
for i in range(5):
    scenario_df = cir.generate_synthetic_data(
        r0=0.03, 
        years=1, 
        random_seed=1000+i  # Different seed for each scenario
    )
    scenario_df['scenario'] = f'Scenario_{i+1}'
    scenarios.append(scenario_df)

# Combine all scenarios
all_scenarios = pd.concat(scenarios, ignore_index=True)
print(f"\nGenerated {len(scenarios)} scenarios")
print(f"Total observations: {len(all_scenarios)}")

# Calculate statistics across scenarios
pivot = all_scenarios.pivot_table(
    values='interest_rate', 
    index='date', 
    columns='scenario'
)
print(f"\nMean rate across scenarios: {pivot.mean().mean():.4%}")
print(f"Std dev across scenarios: {pivot.std().mean():.4%}")

# ============================================================================
# EXAMPLE 6: Manual Parameter Setting (Advanced)
# ============================================================================

print("\n\nEXAMPLE 6: Manual Parameter Setting")
print("="*70)

# Create new model without historical data
cir_manual = CIRModel()

# Set parameters manually
cir_manual.set_parameters(
    kappa=0.5,    # Mean reversion speed
    theta=0.04,   # Long-term mean (4%)
    sigma=0.1     # Volatility
)

# Generate data with manual parameters
manual_df = cir_manual.generate_synthetic_data(r0=0.03, years=1, random_seed=777)
print(f"Mean rate: {manual_df['interest_rate'].mean():.4%}")

# ============================================================================
# EXAMPLE 7: Saving Results
# ============================================================================

print("\n\nEXAMPLE 7: Saving Results")
print("="*70)

# Generate data
output_df = cir.generate_synthetic_data(r0=0.03, years=2)

# Save to CSV
output_df.to_csv('/home/claude/synthetic_interest_rates.csv', index=False)
print("Saved to: /home/claude/synthetic_interest_rates.csv")

# Save to Excel (if openpyxl is available)
try:
    output_df.to_excel('/home/claude/synthetic_interest_rates.xlsx', index=False)
    print("Saved to: /home/claude/synthetic_interest_rates.xlsx")
except ImportError:
    print("Excel export requires openpyxl package")

print("\n" + "="*70)
print("All examples completed!")
print("="*70)
