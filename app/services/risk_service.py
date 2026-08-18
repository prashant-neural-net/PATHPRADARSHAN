def build_risk_ranking(corridors, limit=20):

    if corridors.empty:
        return []

    ranking = corridors.copy()

    # Higher criticality + lower confidence = higher risk
    ranking["risk_score"] = (
        ranking["criticality_score"]
        *
        (1.0 - ranking["mean_confidence"])
    )

    ranking = (
        ranking
        .sort_values(
            "risk_score",
            ascending=False,
        )
        .head(limit)
    )

    return [
        {
            "rank": rank,
            "corridor_id": int(
                row["corridor_id"]
            ),
            "risk_score": float(
                row["risk_score"]
            ),
            "criticality_score": float(
                row["criticality_score"]
            ),
            "confidence": float(
                row["mean_confidence"]
            ),
            "length_m": float(
                row["total_length_m"]
            ),
            "status": row[
                "dominant_status"
            ],
        }
        for rank, (_, row)
        in enumerate(
            ranking.iterrows(),
            start=1,
        )
    ]