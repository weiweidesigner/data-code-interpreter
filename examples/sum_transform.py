def transform(raw_data, question):
    return {"total": sum(row["value"] for row in raw_data), "count": len(raw_data)}
