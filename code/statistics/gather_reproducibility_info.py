"""
Gathers exactly the reproducibility information needed for the
manuscript's Reproducibility statement: software versions, hardware,
OS, and confirms the seed ranges used throughout the project.

Run this on EACH machine used for the experiments (your MacBook Air M4
and the ASUS TUF A15), since they may have different library versions
or hardware. Prints everything to console AND saves a JSON file per
machine - send me both and I'll merge them into one accurate section.

Usage: python3 gather_reproducibility_info.py [machine_label]
Example: python3 gather_reproducibility_info.py "MacBook Air M4"
         python3 gather_reproducibility_info.py "ASUS TUF A15"
"""
import json
import platform
import subprocess
import sys
from pathlib import Path


def get_version(package_name, import_name=None):
    import_name = import_name or package_name
    try:
        module = __import__(import_name)
        return getattr(module, "__version__", "unknown (no __version__ attribute)")
    except ImportError:
        return None


def get_cpu_info():
    system = platform.system()
    try:
        if system == "Darwin":
            result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                     capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout.strip()
        elif system == "Windows":
            result = subprocess.run(["wmic", "cpu", "get", "name"],
                                     capture_output=True, text=True, timeout=5, shell=True)
            if result.returncode == 0:
                lines = [l.strip() for l in result.stdout.splitlines() if l.strip() and l.strip() != "Name"]
                if lines:
                    return lines[0]
        elif system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
    except Exception as e:
        return f"[could not detect: {e}]"
    return platform.processor() or "[unknown]"


def get_ram_gb():
    system = platform.system()
    try:
        if system == "Darwin":
            result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return round(int(result.stdout.strip()) / (1024 ** 3), 1)
        elif system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        kb = int(line.split()[1])
                        return round(kb / (1024 ** 2), 1)
        elif system == "Windows":
            result = subprocess.run(
                ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                capture_output=True, text=True, timeout=5, shell=True)
            if result.returncode == 0:
                lines = [l.strip() for l in result.stdout.splitlines() if l.strip().isdigit()]
                if lines:
                    return round(int(lines[0]) / (1024 ** 3), 1)
    except Exception:
        pass
    return None


def main():
    machine_label = sys.argv[1] if len(sys.argv) > 1 else "unlabeled machine"
    print(f"Gathering reproducibility information for: {machine_label}\n")
    print("=" * 70)
    print("SYSTEM")
    print("=" * 70)
    system_info = {
        "machine_label": machine_label,
        "os": platform.system(),
        "os_version": platform.version(),
        "os_release": platform.release(),
        "python_version": platform.python_version(),
        "cpu": get_cpu_info(),
        "ram_gb": get_ram_gb(),
        "architecture": platform.machine(),
    }
    for k, v in system_info.items():
        print(f"  {k}: {v}")

    print(f"\n{'=' * 70}")
    print("PACKAGE VERSIONS")
    print("=" * 70)
    packages_to_check = [
        ("numpy", "numpy"), ("pandas", "pandas"), ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"), ("shap", "shap"), ("scipy", "scipy"),
        ("statsmodels", "statsmodels"), ("matplotlib", "matplotlib"),
    ]
    package_versions = {}
    for display_name, import_name in packages_to_check:
        version = get_version(display_name, import_name)
        package_versions[display_name] = version
        status = version if version else "NOT INSTALLED / not found"
        print(f"  {display_name}: {status}")

    print(f"\n{'=' * 70}")
    print("OLLAMA / LLM (checked separately - not a pip package)")
    print("=" * 70)
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
        ollama_version = result.stdout.strip() if result.returncode == 0 else "[ollama command found but failed]"
    except FileNotFoundError:
        ollama_version = "[ollama not found on this machine - expected only where the XAI layer ran]"
    except Exception as e:
        ollama_version = f"[error checking: {e}]"
    print(f"  ollama: {ollama_version}")

    print(f"\n{'=' * 70}")
    print("SEEDS USED (fixed across the project - confirm these match your records)")
    print("=" * 70)
    seed_info = {
        "primary_arbitration_experiments": "0-9 (10 seeds)",
        "ugaa_k_sensitivity_and_budget_comparison": "0-9 (10 seeds, same as primary - the 'selection' seeds)",
        "ugaa_primary_locked_evaluation": "10-19 (10 seeds, disjoint from k-selection)",
        "note": "If any experiment used a different seed range, note it explicitly when replying.",
    }
    for k, v in seed_info.items():
        print(f"  {k}: {v}")

    all_results = {
        "system_info": system_info, "package_versions": package_versions,
        "ollama_version": ollama_version, "seed_info": seed_info,
    }
    out_path = Path(f"reproducibility_info_{machine_label.replace(' ', '_')}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("\nSend this file's contents (or just paste the console output above) back,")
    print("along with the same from your other machine if experiments ran on both.")


if __name__ == "__main__":
    main()
