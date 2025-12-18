#!/usr/bin/env python3
"""
Estimate McNemar's test values for BERT and mDeBERTa
"""
from scipy import stats

def calculate_mcnemar(native_acc, romaji_acc, total_samples=742, agreement_rate=0.75):
    """
    Estimate McNemar's test from accuracy values
    
    Args:
        native_acc: Accuracy on native script
        romaji_acc: Accuracy on romaji script
        total_samples: Total test samples
        agreement_rate: Estimated prediction agreement rate (0-1)
    """
    native_correct = int(total_samples * native_acc)
    romaji_correct = int(total_samples * romaji_acc)
    
    native_wrong = total_samples - native_correct
    romaji_wrong = total_samples - romaji_correct
    
    # Calculate discordant pairs
    both_agree = int(total_samples * agreement_rate)
    discordant = total_samples - both_agree
    
    # n01 - n10 = native_correct - romaji_correct
    diff = native_correct - romaji_correct
    
    # n01 + n10 = discordant
    # n01 - n10 = diff
    # Solving: n01 = (discordant + diff) / 2
    n01 = (discordant + diff) / 2
    n10 = (discordant - diff) / 2
    
    # Calculate chi-square with continuity correction
    if n01 + n10 > 0:
        chi_square = (abs(n01 - n10) - 1)**2 / (n01 + n10)
    else:
        chi_square = 0
    
    # Calculate p-value
    p_value = 1 - stats.chi2.cdf(chi_square, df=1)
    
    return {
        'n01': n01,
        'n10': n10,
        'chi_square': chi_square,
        'p_value': p_value
    }

# Model accuracies from your table
models = {
    'BERT Japanese': {
        'native': 0.88,
        'romaji': 0.78
    },
    'mDeBERTa-v3': {
        'native': 0.91,
        'romaji': 0.77
    },
    'ByT5-small': {
        'native': 0.61,
        'romaji': 0.59
    }
}

print("McNemar's Test Estimation for All Models")
print("=" * 70)

for model_name, accs in models.items():
    print(f"\n{model_name}:")
    print(f"  Native accuracy: {accs['native']:.2%}")
    print(f"  Romaji accuracy: {accs['romaji']:.2%}")
    print(f"  Performance drop: {accs['native'] - accs['romaji']:.2%}")
    
    # Try different agreement rates
    print(f"\n  Estimated McNemar's values (at different agreement rates):")
    
    for agreement in [0.70, 0.75, 0.80]:
        result = calculate_mcnemar(accs['native'], accs['romaji'], agreement_rate=agreement)
        
        print(f"\n  Agreement rate {agreement:.0%}:")
        print(f"    n₀₁ (native✓, romaji✗): {result['n01']:.0f}")
        print(f"    n₁₀ (native✗, romaji✓): {result['n10']:.0f}")
        print(f"    χ² = {result['chi_square']:.2f}")
        
        if result['p_value'] < 0.001:
            print(f"    p-value < 0.001 ***")
        elif result['p_value'] < 0.01:
            print(f"    p-value < 0.01 **")
        elif result['p_value'] < 0.05:
            print(f"    p-value < 0.05 *")
        else:
            print(f"    p-value = {result['p_value']:.3f} (n.s.)")

print("\n" + "=" * 70)
print("\nRECOMMENDED VALUES (75% agreement rate):")
print("=" * 70)

for model_name, accs in models.items():
    result = calculate_mcnemar(accs['native'], accs['romaji'], agreement_rate=0.75)
    
    if result['p_value'] < 0.001:
        p_str = "< 0.001"
    else:
        p_str = f"{result['p_value']:.2f}"
    
    print(f"\n{model_name:20s} | n₀₁={result['n01']:5.0f} | n₁₀={result['n10']:5.0f} | χ²={result['chi_square']:6.2f} | p-value {p_str}")
