"""Small, dependency-free resource guards for local model stages."""
import os


def available_memory_gib() -> float:
    values = {}
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0])
    return values["MemAvailable"] / 1024 / 1024


def require_memory(stage: str, minimum_gib: float) -> None:
    """Stop before model loading when Linux reports too little available memory."""
    if os.getenv("HEALTHRAG_SKIP_MEMORY_GUARD") == "1":
        return
    available = available_memory_gib()
    if available < minimum_gib:
        raise RuntimeError(
            f"{stage} requires at least {minimum_gib:.1f} GiB available RAM; "
            f"only {available:.1f} GiB is available. Stop other model services and retry."
        )
