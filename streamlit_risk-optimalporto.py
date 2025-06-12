import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from scipy.optimize import minimize

st.set_page_config(page_title="PortoRisk", layout="wide")
# Ucapan selamat datang dan judul
st.title("👋 Selamat Datang di PortoRisk")
st.subheader("📊 Aplikasi Evaluasi Risiko & Optimasi Portofolio Investasi")

# Penjelasan ringkas
st.markdown("PortoRisk dirancang untuk membantu Anda memahami dan menganalisis portofolio investasi berdasarkan data historis.")

# Fitur aplikasi dalam bentuk poin
st.markdown("### 🔧 Fitur Utama:")
st.markdown("""
- 🔗 **Visualisasi Korelasi Aset**  
- ⚠️ **Simulasi Risiko (Value at Risk / VaR)**  
- 📈 **Optimasi Portofolio Berdasarkan Return dan Risiko**
""")

# Disclaimer atau penafian
st.markdown("---")
st.markdown("⚠️ *Catatan:*")
st.markdown("""
- Aplikasi ini **bukan alat prediksi masa depan** maupun **saran keuangan profesional**.  
- Seluruh hasil bersifat **edukatif** dan berdasarkan data historis.  
- Selalu lakukan riset pribadi dan konsultasi sebelum mengambil keputusan investasi (**Do Your Own Research**).
""")

# 1. Upload Excel File
st.header("1. Upload File Excel")
st.markdown("""
Silakan upload file Excel (.xlsx).  
Download data Excel (bonus) [di sini](https://tinyurl.com/DataSetBonusApp-1234) atau buat sendiri dengan format yang sama [panduan & contoh](https://tinyurl.com/ContohSampelDanPanduanApp-1234).
""")

uploaded_file = st.file_uploader("Upload file Excel", type=[".xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, engine="openpyxl")

    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df.columns.values[0] = "Date"
    dates = df["Date"]
    tickers = df.columns[1:].tolist()

    st.subheader("\U0001F4C4 Preview Data")
    st.dataframe(df.head(), use_container_width=True)

    # 2. Pilih Aset
    st.header("2. Pilih Aset untuk Dievaluasi")
    selected_tickers = st.multiselect("Pilih aset (minimal 2):", tickers, default=tickers[:6])

    if len(selected_tickers) >= 2:
        # 3. Pilih Tanggal
        st.header("3. Pilih Rentang Tanggal")
        min_date, max_date = dates.min(), dates.max()
        selected_date_range = st.date_input("Pilih rentang tanggal yang ingin dievaluasi:", [min_date, max_date], min_value=min_date, max_value=max_date)

        if len(selected_date_range) == 2:
            start_date, end_date = selected_date_range
            filtered_df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]
            filtered_df.set_index("Date", inplace=True)
            filtered_df[selected_tickers] = filtered_df[selected_tickers].apply(pd.to_numeric, errors='coerce')

            # 4. Correlation Heatmap
            st.header("4. Correlation Heatmap")
            st.markdown("""
            Menunjukkan hubungan antara pergerakan harga aset.  
            Warna mendekati merah (+1) = korelasi tinggi, warna mendekati biru (-1) = korelasi negatif.  
            Angka lebih besar berarti aset bergerak searah, angka kecil atau negatif berarti cenderung berlawanan arah.
            """)
            returns = filtered_df[selected_tickers].pct_change().dropna()
            corr = returns.corr()

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
            st.pyplot(fig)

            # Menemukan pasangan aset dengan korelasi tertinggi dan terendah
            corr_values = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            min_corr = corr_values.min().min()
            max_corr = corr_values.max().max()

            min_pair = corr_values.stack().idxmin()
            max_pair = corr_values.stack().idxmax()

            # Tampilkan dalam kotak hijau
            st.success(f"""
            📊 **Analisis Korelasi:**

            - Pasangan aset dengan korelasi **terendah**: **{min_pair[0]}** dan **{min_pair[1]}** dengan nilai korelasi **{min_corr:.2f}**.
            - Pasangan aset dengan korelasi **tertinggi**: **{max_pair[0]}** dan **{max_pair[1]}** dengan nilai korelasi **{max_corr:.2f}**.
            """)



            # 5. Proyeksi VaR
            st.header("5. Proyeksi dan Simulasi VaR (Value at Risk)")
            st.markdown("""
            Proyeksi & simulasi berikut ini membantu memperkirakan potensi kerugian di masa depan berdasarkan pergerakan historis harga dan simulasi acak (Geometric Brownian Motion). VaR 95% artinya ada 95% kemungkinan kerugian tidak melebihi nilai tertentu dalam jangka waktu yang dipilih.
            """)
            projection_map = {
                "1 Bulan (21 hari)": 21,
                "3 Bulan (63 hari)": 63,
                "6 Bulan (126 hari)": 126,
                "236 hari": 236,
                "1 Tahun (252 hari)": 252
            }

            horizon = st.selectbox("Pilih horizon proyeksi:", list(projection_map.keys()))
            forecast_days = projection_map[horizon]

            confidence_level = 0.95
            num_simulations = 10000
            summary = []

            for ticker in selected_tickers:
                price_series = filtered_df[ticker].dropna()
                ticker_returns = price_series.pct_change().dropna()
                log_returns = np.log(1 + ticker_returns)

                S0 = price_series.iloc[-1]
                mu = log_returns.mean()
                sigma = log_returns.std()
                dt = 1
                N = forecast_days
                t = np.arange(1, N + 1)

                simulated_prices = np.zeros((num_simulations, N + 1))
                for i in range(num_simulations):
                    Z = np.random.normal(0, 1, N)
                    W = np.cumsum(Z) * np.sqrt(dt)
                    drift = (mu - 0.5 * sigma**2) * t
                    diffusion = sigma * W
                    S_t = S0 * np.exp(drift + diffusion)
                    simulated_prices[i, :] = np.insert(S_t, 0, S0)

                ending_prices = simulated_prices[:, -1]
                portfolio_changes = ending_prices - S0
                sorted_changes = np.sort(portfolio_changes)
                var = -np.percentile(sorted_changes, confidence_level * 100)
                mean_change = np.mean(portfolio_changes)

                summary.append({
                    "Aset": ticker,
                    "Harga Terakhir": f"Rp{S0:,.0f}",
                    "Rata-rata Simulasi": f"Rp{mean_change:,.0f}",
                    f"VaR {int(confidence_level*100)}%": f"Rp{var:,.0f}"
                })

                fig, axs = plt.subplots(1, 2, figsize=(14, 5))
                cmap = plt.get_cmap('plasma')
                colors = cmap(np.linspace(0, 1, 100))
                for i, color in zip(range(100), colors):
                    axs[0].plot(simulated_prices[i], color=color, alpha=0.6, lw=1)
                axs[0].axhline(S0, color='black', linestyle='--', label='Harga Awal')
                axs[0].set_title(f'Simulasi Harga: {ticker}')
                axs[0].legend()
                axs[0].grid(True, linestyle='--', alpha=0.5)

                axs[1].hist(portfolio_changes, bins=80, color='gray', edgecolor='black', alpha=0.7, density=True)
                axs[1].axvline(x=-var, color='red', linestyle='--', linewidth=2,
                            label=f'VaR ({int(confidence_level*100)}%) = Rp {var:,.0f}')
                axs[1].set_title(f'Distribusi Perubahan Nilai: {ticker}')
                axs[1].legend()
                axs[1].grid(True, linestyle='--', alpha=0.5)

                st.pyplot(fig)

            st.subheader("\U0001F4CA Ringkasan Proyeksi VaR")
            st.dataframe(pd.DataFrame(summary))

            # 6. Mean-Variance Portfolio Optimization
            st.header("6. Mean-Variance Portfolio Optimization")
            st.markdown("""
            Portofolio optimal adalah kombinasi aset yang memberikan rasio imbal hasil terhadap risiko terbaik. Artinya, portofolio ini dirancang agar **imbal hasil (return) maksimal dibandingkan risiko yang diambil**. Perhitungan pada aplikasi ini didasarkan pada data historis, dan dihitung menggunakan metode Mean-Variance (Markowitz).
            """)

            st.subheader("Harga Mentah")
            st.markdown("Menampilkan harga asli dari masing-masing aset.")
            fig, ax = plt.subplots(figsize=(10, 4))
            filtered_df[selected_tickers].plot(ax=ax)
            ax.set_ylabel("Harga")
            st.pyplot(fig)

            st.subheader("Harga Ternormalisasi")
            st.markdown("Harga dinormalisasi agar mudah dibandingkan (semua dimulai dari angka 1).")
            norm_prices = filtered_df[selected_tickers] / filtered_df[selected_tickers].iloc[0]
            fig, ax = plt.subplots(figsize=(10, 4))
            norm_prices.plot(ax=ax)
            ax.set_ylabel("Harga Ternormalisasi")
            st.pyplot(fig)

            log_returns_df = np.log(filtered_df[selected_tickers] / filtered_df[selected_tickers].shift(1)).dropna()
            mean_returns = log_returns_df.mean()
            cov_matrix = log_returns_df.cov()

            # Cek apakah jumlah data cukup untuk simulasi portofolio
            if len(filtered_df) <= 235:
                st.warning("⚠️ Data yang Anda pilih pada rentang tanggal hanya memiliki "
                           f"{len(filtered_df)} data. Tambah rentang tanggal Anda agar portofolio dapat disimulasikan "
                           "(minimal 236 data).")
                st.stop()


            # === Simulasi Portofolio Acak & Efficient Frontier ===
            st.subheader("Simulasi Portofolio Acak")

            @st.cache_data
            def simulate_portfolios(mean_returns, cov_matrix, num_portfolios=10000):
                results = np.zeros((3, num_portfolios))
                weight_array = []

                for i in range(num_portfolios):
                    weights = np.random.random(len(mean_returns))
                    weights /= np.sum(weights)
                    weight_array.append(weights)

                    portfolio_return = np.sum(mean_returns * weights) * 252
                    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
                    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0

                    results[0, i] = portfolio_return
                    results[1, i] = portfolio_volatility
                    results[2, i] = sharpe_ratio

                return results, weight_array

            results, weight_array = simulate_portfolios(mean_returns, cov_matrix)
            results_df = pd.DataFrame(results.T, columns=["Return", "Volatility", "Sharpe Ratio"])
            weights_df = pd.DataFrame(weight_array, columns=selected_tickers)

            # Validasi isi dataframe
            results_df = results_df.replace([np.inf, -np.inf], np.nan).dropna()

            if not results_df.empty:
                max_sharpe_idx = results_df["Sharpe Ratio"].idxmax()
                min_vol_idx = results_df["Volatility"].idxmin()

                fig2, ax2 = plt.subplots(figsize=(10, 6))
                scatter = ax2.scatter(results_df["Volatility"], results_df["Return"], c=results_df["Sharpe Ratio"], cmap='viridis', alpha=0.7)

                # Validasi nilai plotting
                try:
                    max_vol = results_df.loc[max_sharpe_idx, "Volatility"]
                    max_ret = results_df.loc[max_sharpe_idx, "Return"]
                    min_vol = results_df.loc[min_vol_idx, "Volatility"]
                    min_ret = results_df.loc[min_vol_idx, "Return"]

                    ax2.scatter(max_vol, max_ret, marker='*', color='red', s=200, label='Max Sharpe Ratio')
                    ax2.scatter(min_vol, min_ret, marker='*', color='blue', s=200, label='Min Volatility')
                except Exception as e:
                    st.warning(f"Gagal menampilkan titik spesifik pada grafik: {e}")

                ax2.set_xlabel("Volatility (Risk)")
                ax2.set_ylabel("Expected Return (Annual)")
                ax2.set_title("Random Portfolios")
                ax2.legend()
                plt.colorbar(scatter, label='Sharpe Ratio')
                st.pyplot(fig2)

                # === Slider Return Target ===
                st.subheader("🎯 Optimasi Portofolio Berdasarkan Return yang Diinginkan")
                min_ret = results_df["Return"].min()
                max_ret = results_df["Return"].max()

                if round(min_ret * 100, 2) == round(max_ret * 100, 2):
                    st.warning("Rentang return terlalu sempit. Slider diatur ke default range.")
                    slider_min = round(min_ret * 100, 2) - 10
                    slider_max = round(max_ret * 100, 2) + 10
                else:
                    slider_min = round(min_ret * 100, 2)
                    slider_max = round(max_ret * 100, 2)

                target_return = st.slider("Tentukan expected return tahunan yang diinginkan (%):",
                                          slider_min,
                                          slider_max,
                                          slider_max + (slider_max - slider_min) / 2) / 100

                # === Fungsi Optimasi Portofolio ===
                def portfolio_performance(weights, mean_returns, cov_matrix):
                    returns = np.sum(weights * mean_returns) * 252
                    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
                    return returns, volatility

                def min_variance(weights, mean_returns, cov_matrix, target_return):
                    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

                def constraint_return(weights, mean_returns, target):
                    return np.sum(weights * mean_returns) * 252 - target

                num_assets = len(selected_tickers)
                initial_guess = num_assets * [1. / num_assets, ]
                args = (mean_returns, cov_matrix, target_return)
                constraints = (
                    {'type': 'eq', 'fun': constraint_return, 'args': (mean_returns, target_return)},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
                )
                bounds = tuple((0, 1) for _ in range(num_assets))

                optimized = minimize(min_variance, initial_guess, args=args,
                                     method='SLSQP', bounds=bounds, constraints=constraints)

                if optimized.success:
                    opt_weights = optimized.x
                    opt_return, opt_volatility = portfolio_performance(opt_weights, mean_returns, cov_matrix)

                    st.write(f"📌 **Expected Return**: {opt_return*100:.2f}% per tahun")
                    st.write(f"📌 **Expected Volatility (Risk)**: {opt_volatility*100:.2f}%")

                    allocation_df = pd.DataFrame({
                        "Aset": selected_tickers,
                        "Bobot Optimal (%)": np.round(opt_weights * 100, 2)
                    })
                    st.dataframe(allocation_df)
                else:
                    st.error("Optimasi gagal menemukan solusi. Coba ubah nilai return atau data.")
            else:
                st.error("Data simulasi kosong atau tidak valid. Silakan pastikan data input memadai.")

    # Feedback
    st.markdown("---")
    st.markdown("💬 Beri komentar atau masukan aplikasi [di sini](https://tinyurl.com/FeedbackAppERPO-1234)")

else:
    st.info("Silakan upload file Excel terlebih dahulu.")

