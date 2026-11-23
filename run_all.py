"""
run_all.py  --  SATU PERINTAH untuk menjalankan seluruh proyek
===============================================================
    python run_all.py

Urutan yang dikerjakan otomatis:
    0. Cek & pasang paket Python yang belum ada
    1. Unduh data IHSG asli terbaru (Yahoo Finance)
    2. Unduh kurs USD/IDR asli (Yahoo Finance)  -> variabel X2
    3. Unduh Fed Funds Rate asli (FRED)
    4. Salin BI Rate asli (data_bawaan/bi_rate.csv)  -> variabel X1
    5. Gabungkan semua data                       (build_dataset.py)
    6. Latih model LSTM + evaluasi                (train.py)
    7. Buka dashboard di browser                  (app.py / Streamlit)

Opsi:
    --skip-install   lewati pengecekan/instalasi paket
    --skip-fetch     jangan unduh data (pakai data/ihsg.csv & fed_funds_rate.csv yang sudah ada)
    --skip-train     jangan latih ulang (pakai model/ yang sudah ada)
    --no-app         jangan buka dashboard di akhir
    --walkforward    setelah training, jalankan evaluasi walk-forward (lebih lama)
    --lengkap        jalankan juga experiments.py (M0-M4) dan econometrics.py

Tekan Ctrl+C di terminal untuk menghentikan dashboard.
"""
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)   # semua path relatif (data/, model/) mengacu ke folder proyek

PACKAGES = {  # nama modul -> nama paket pip
    "pandas": "pandas", "numpy": "numpy", "yfinance": "yfinance",
    "requests": "requests", "openpyxl": "openpyxl", "sklearn": "scikit-learn", "matplotlib": "matplotlib",
    "joblib": "joblib", "torch": "torch",
    "streamlit": "streamlit", "plotly": "plotly", "scipy": "scipy", "statsmodels": "statsmodels",
}


def banner(n, text):
    print(f"\n{'=' * 70}\n LANGKAH {n}: {text}\n{'=' * 70}", flush=True)


def run(script, *args):
    """Jalankan skrip proyek; hentikan seluruh proses jika gagal."""
    result = subprocess.run([sys.executable, script, *args])
    if result.returncode != 0:
        print(f"\n[GAGAL] {script} berhenti dengan kode {result.returncode}. "
              f"Baca pesan error di atasnya, perbaiki, lalu jalankan ulang: python run_all.py")
        sys.exit(result.returncode)


def ensure_packages():
    missing = [pip for mod, pip in PACKAGES.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        print("Semua paket sudah terpasang.")
        return
    print("Memasang paket yang belum ada:", ", ".join(missing))
    print("(torch berukuran besar; pertama kali bisa beberapa menit)")
    code = subprocess.call([sys.executable, "-m", "pip", "install", *missing])
    if code != 0:
        print("[GAGAL] Instalasi paket gagal. Coba manual: pip install -r requirements.txt")
        sys.exit(code)


def fetch_with_fallback(script, output_file, label):
    """Unduh data; jika gagal tapi file lama masih ada, lanjut dengan file lama."""
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        if os.path.exists(output_file):
            print(f"[peringatan] Gagal memperbarui {label}; memakai file lama: {output_file}")
        else:
            print(f"[GAGAL] Tidak bisa mengunduh {label} dan belum ada file lama. "
                  f"Periksa koneksi internet lalu jalankan ulang.")
            sys.exit(1)


def prepare_streamlit_no_prompt():
    """Streamlit meminta email saat pertama dijalankan (memblokir terminal).
    Buat credentials kosong agar langsung jalan."""
    cred_dir = os.path.join(os.path.expanduser("~"), ".streamlit")
    cred = os.path.join(cred_dir, "credentials.toml")
    if not os.path.exists(cred):
        os.makedirs(cred_dir, exist_ok=True)
        with open(cred, "w", encoding="utf-8") as f:
            f.write('[general]\nemail = ""\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--no-app", action="store_true")
    ap.add_argument("--walkforward", action="store_true")
    ap.add_argument("--lengkap", action="store_true",
                    help="jalankan juga experiments.py dan econometrics.py (lama)")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs("data", exist_ok=True)
    os.makedirs("model", exist_ok=True)

    banner(0, "Cek paket Python")
    if args.skip_install:
        print("(dilewati)")
    else:
        ensure_packages()

    banner(1, "Data IHSG asli (Yahoo Finance)")
    if args.skip_fetch:
        print("(dilewati)")
    else:
        fetch_with_fallback("fetch_ihsg.py", "data/ihsg.csv", "data IHSG")

    banner(2, "Kurs USD/IDR JISDOR (Bank Indonesia) - variabel X2")
    if args.skip_fetch:
        print("(dilewati)")
    else:
        fetch_with_fallback("fetch_kurs.py", "data/kurs_usd_idr.csv", "data kurs USD/IDR")

    banner(3, "Fed Funds Rate asli (FRED)")
    if args.skip_fetch:
        print("(dilewati)")
    else:
        # Hanya bagian Fed; BI Rate memakai data bawaan yang sudah diverifikasi.
        code = ("import fetch_interest_rate as f; f.fetch_fed_funds_rate()")
        result = subprocess.run([sys.executable, "-c", code])
        if result.returncode != 0:
            if os.path.exists("data/fed_funds_rate.csv"):
                print("[peringatan] Gagal memperbarui Fed Rate; memakai file lama.")
            else:
                print("[GAGAL] Tidak bisa mengunduh Fed Rate dan belum ada file lama.")
                sys.exit(1)

    banner(4, "BI Rate asli (keputusan RDG Bank Indonesia) - variabel X1")
    shutil.copyfile("data_bawaan/bi_rate.csv", "data/bi_rate.csv")
    with open("data/bi_rate.csv", encoding="utf-8") as f:
        rows = f.read().strip().splitlines()[1:]
    print(f"BI Rate ({len(rows)} keputusan RDG, {rows[0][:10]} s.d. {rows[-1][:10]}) -> data/bi_rate.csv")

    banner(5, "Gabungkan semua data")
    if os.path.exists("data/dataset_final.csv"):
        os.remove("data/dataset_final.csv")   # pastikan tidak memakai versi lama
    run("build_dataset.py")

    banner(6, "Latih model LSTM + evaluasi")
    if args.skip_train and os.path.exists("model/lstm_ihsg.pt"):
        print("(dilewati, memakai model yang sudah ada)")
    else:
        run("train.py")
        if args.walkforward:
            banner("6b", "Evaluasi walk-forward (5 lipatan)")
            run("evaluate_walkforward.py")
        if args.lengkap:
            banner("6c", "Eksperimen M0-M4, horizon, kepentingan fitur")
            run("experiments.py")
            banner("6d", "Analisis ekonometrika (Granger/VAR, event study RDG)")
            run("econometrics.py")
            banner("6e", "Grafik resolusi tinggi untuk naskah (figures/)")
            run("figures.py")

    print(f"\nSelesai menyiapkan semuanya dalam {time.time() - t0:.0f} detik.")

    if args.no_app:
        print("Dashboard tidak dibuka (--no-app). Jalankan manual: streamlit run app.py")
        return

    banner(7, "Membuka dashboard di browser")
    print("Alamat: http://localhost:8501   (Ctrl+C di terminal ini untuk berhenti)")
    prepare_streamlit_no_prompt()
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py",
                    "--browser.gatherUsageStats", "false"])


if __name__ == "__main__":
    main()
