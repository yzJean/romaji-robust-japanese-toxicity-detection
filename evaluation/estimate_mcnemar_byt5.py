#!/usr/bin/env python3
"""
Estimate McNemar's test values for ByT5 from existing metrics
"""

# From your results:
# ByT5 Native: 454 correct out of 742 (61.19% accuracy)
# ByT5 Romaji: 440 correct out of 742 (59.30% accuracy)

total_samples = 742
native_correct = 454
romaji_correct = 440

# Both correct (Type A - both agree and correct)
# Both wrong (Type D - both agree but wrong)
# n01: native correct, romaji wrong
# n10: native wrong, romaji correct

# We know:
# native_correct + native_wrong = 742
# romaji_correct + romaji_wrong = 742

native_wrong = total_samples - native_correct  # 288
romaji_wrong = total_samples - romaji_correct  # 302

print(f"Native correct: {native_correct}, wrong: {native_wrong}")
print(f"Romaji correct: {romaji_correct}, wrong: {romaji_wrong}")

# For McNemar's test, we need the 2x2 contingency table:
#                 Romaji Correct  |  Romaji Wrong
# Native Correct      both_correct |  n01 (native✓, romaji✗)
# Native Wrong        n10 (native✗, romaji✓) | both_wrong

# We know:
# both_correct + n01 = native_correct (454)
# both_correct + n10 = romaji_correct (440)
# n01 + both_wrong = romaji_wrong (302)
# n10 + both_wrong = native_wrong (288)

# From the equations:
# both_correct + n01 = 454
# both_correct + n10 = 440
# Subtracting: n01 - n10 = 14

# Also:
# both_correct + n01 + n10 + both_wrong = 742
# Let both_correct = x
# Then: x + n01 + n10 + both_wrong = 742

# We need another constraint. Let's use the fact that we know
# the flip rate or can estimate agreement.

# Maximum possible agreement (both_correct + both_wrong):
# This occurs when n01 and n10 are minimized

# Minimum n01 + n10 occurs when predictions overlap maximally
# both_correct >= max(0, native_correct + romaji_correct - total)
min_both_correct = max(0, native_correct + romaji_correct - total_samples)
print(f"\nMinimum both_correct: {min_both_correct}")

# Maximum both_correct = min(native_correct, romaji_correct)
max_both_correct = min(native_correct, romaji_correct)
print(f"Maximum both_correct: {max_both_correct}")

# Let's estimate assuming medium agreement (70-80% of samples have same prediction)
# Try 75% agreement
estimated_agreement = 0.75
both_agree = int(total_samples * estimated_agreement)  # ~557

# If both_agree = both_correct + both_wrong
# And native_correct = both_correct + n01
# And romaji_correct = both_correct + n10

# Let's solve:
# Assume both_correct is close to the overlap
# both_correct ≈ min(native_correct, romaji_correct) - some disagreement

# Try different agreement rates:
for agreement_rate in [0.70, 0.75, 0.80, 0.85]:
    both_agree = int(total_samples * agreement_rate)
    
    # both_correct + both_wrong = both_agree
    # both_correct + n01 = 454
    # both_correct + n10 = 440
    # n01 + n10 + both_correct + both_wrong = 742
    
    # From last equation: n01 + n10 = 742 - both_agree
    discordant = 742 - both_agree
    
    # From n01 - n10 = 14 and n01 + n10 = discordant
    # n01 = (discordant + 14) / 2
    # n10 = (discordant - 14) / 2
    
    n01 = (discordant + 14) / 2
    n10 = (discordant - 14) / 2
    
    # Calculate chi-square
    if n01 + n10 > 0:
        chi_square = (abs(n01 - n10) - 1)**2 / (n01 + n10)  # with continuity correction
    else:
        chi_square = 0
    
    # Calculate p-value
    from scipy import stats
    p_value = 1 - stats.chi2.cdf(chi_square, df=1)
    
    print(f"\nAgreement rate: {agreement_rate:.0%}")
    print(f"  Both agree: {both_agree}")
    print(f"  n01 (native✓, romaji✗): {n01:.0f}")
    print(f"  n10 (native✗, romaji✓): {n10:.0f}")
    print(f"  χ² = {chi_square:.2f}")
    print(f"  p-value = {p_value:.4f}")
    
    if p_value < 0.001:
        sig = "p < 0.001 (highly significant)"
    elif p_value < 0.01:
        sig = f"p < 0.01 (significant)"
    elif p_value < 0.05:
        sig = f"p < 0.05 (significant)"
    else:
        sig = f"p = {p_value:.3f} (not significant)"
    print(f"  Interpretation: {sig}")
