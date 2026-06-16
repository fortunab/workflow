def generate_report(tokens):
    text = "\n".join(tokens) if isinstance(tokens, list) else str(tokens)

    if "count='0'" in text or 'count="0"' in text:
        return (
            "No relevant lesion was detected. The uncertainty is high because no detection or "
            "segmentation evidence is available. Additional clinical review is recommended if needed."
        )

    if "level='low'" in text and "UNCERTAINTY" in text:
        return (
            "A consistent lesion-like region was identified. Detection and segmentation tokens support "
            "a focused finding. This is decision-support output and must be reviewed by a clinician."
        )

    if "low_confidence_detection" in text:
        return (
            "A possible lesion-like region was suggested, but confidence is limited. The case should be "
            "flagged for expert review rather than treated as a definitive diagnosis."
        )

    return (
        "The pipeline produced structured evidence. The result should be interpreted as decision support, "
        "not as a standalone diagnosis."
    )
