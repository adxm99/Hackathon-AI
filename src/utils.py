def hr_recommendation(probability: float) -> str:
    if probability >= 0.75:
        return "High risk: schedule a retention interview, review engagement, workload and compensation."
    if probability >= 0.50:
        return "Moderate risk: manager follow-up and targeted engagement review recommended."
    return "Low risk: continue regular monitoring."