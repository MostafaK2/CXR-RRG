import re

def extract_findings(report):
    """Extract medical entities — simple keyword approach"""
    keywords = [
        'pneumonia', 'effusion', 'cardiomegaly', 'consolidation',
        'atelectasis', 'pneumothorax', 'edema', 'opacity',
        'fracture', 'normal', 'clear', 'enlarged'
    ]
    found = []
    report_lower = report.lower()
    for kw in keywords:
        if kw in report_lower:
            found.append(kw)
    return set(found)

hallucination_rates = []

for sample in test_set:
    generated = model.generate(sample['image'], sample['text'])[0]
    reference = sample['reference']

    gen_findings = extract_findings(generated)
    ref_findings = extract_findings(reference)

    # Hallucinated = in generated but NOT in reference
    hallucinated = gen_findings - ref_findings

    if len(gen_findings) > 0:
        rate = len(hallucinated) / len(gen_findings)
        hallucination_rates.append(rate)

print(f"Mean hallucination rate: {np.mean(hallucination_rates):.3f}")
print(f"% reports with any hallucination: {np.mean([r>0 for r in hallucination_rates])*100:.1f}%")