import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from scipy.optimize import minimize

st.set_page_config(page_title="Aplikasi Evaluasi Risiko Investasi", layout="wide")
st.title("\U0001F4C8 Aplikasi Evaluasi Risiko & Portofolio Optimal Investasi")

st.markdown("""
Aplikasi ini membantu Anda memahami risiko dan strategi alokasi portofolio untuk berbagai aset investasi (seperti saham dan lainnya).  
Fitur: visualisasi korelasi, simulasi risiko (VaR), serta optimasi portofolio berdasarkan return dan risiko.
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
    selected_tickers = st.multiselect("Pilih aset (minimal 2):", tickers, default=tickers[:2])

    if len(selected_tickers) >= 2:
        # 3. Pilih Tanggal
        st.header("3. Pilih Rentang Tanggal")
        st.markdown("Pilih rentang tanggal yang ingin dievaluasi:")
        min_date, max_date = dates.min(), dates.max()
        selected_date_range = st.date_input("Pilih rentang tanggal:", [min_date, max_date], min_value=min_date, max_value=max_date)

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

            # 5. Proyeksi VaR
            st.header("5. Proyeksi dan Simulasi VaR")
            st.markdown("""
            Proyeksi & simulasi berikut ini membantu memperkirakan potensi kerugian di masa depan berdasarkan pergerakan historis harga dan simulasi acak (Geometric Brownian Motion). VaR 95% artinya ada 95% kemungkinan kerugian tidak melebihi nilai tertentu dalam jangka waktu yang dipilih.
            """)
            projection_map = {"Mingguan (5 hari)": 5, "Bulanan (21 hari)": 21, "Tahunan (252 hari)": 252}
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

            mean_returns = returns.mean()
            cov_matrix = returns.cov()

            st.subheader("Simulasi Random Portfolio")
            st.markdown("Menampilkan kombinasi alokasi aset acak dengan risiko dan return yang berbeda-beda.")
            num_portfolios = 5000
            results = np.zeros((3, num_portfolios))
            weights_record = []

            for i in range(num_portfolios):
                weights = np.random.random(len(selected_tickers))
                weights /= np.sum(weights)
                weights_record.append(weights)

                port_return = np.sum(mean_returns * weights)
                port_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

                results[0,i] = port_stddev
                results[1,i] = port_return
                results[2,i] = (port_return / port_stddev)

            fig, ax = plt.subplots(figsize=(10,6))
            scatter = ax.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis', marker='o')
            ax.set_xlabel('Volatilitas')
            ax.set_ylabel('Return')
            ax.set_title('Simulasi Portofolio Acak')
            st.pyplot(fig)

            st.subheader("Portofolio Optimal (Mean-Variance)")
            st.markdown("Menampilkan kombinasi aset terbaik dengan rasio return terhadap risiko yang dimiliki.")
            def neg_sharpe(weights):
                ret = np.sum(mean_returns * weights)
                vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return -ret / vol

            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in selected_tickers)
            initial_guess = len(selected_tickers) * [1. / len(selected_tickers)]

            opt_result = minimize(neg_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            opt_weights = opt_result.x

            opt_return = np.sum(mean_returns * opt_weights)
            annualized_return = opt_return * 252
            opt_volatility = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights)))

            st.write("### Alokasi Aset Optimal")
            for ticker, weight in zip(selected_tickers, opt_weights):
                st.write(f"{ticker}: {weight:.2%}")

            st.markdown(f"""
            **Expected Return (Rata-rata Imbal Hasil Harian):** {opt_return:.2%}  
            > Ini adalah estimasi keuntungan rata-rata *harian* dari portofolio Anda.  
            > Berdasarkan data historis, potensi keuntungan dalam setahun diperkirakan sekitar **{annualized_return * 100:.2f}%**, dengan asumsi pola return harian berlanjut sepanjang tahun.

            **Volatility (Tingkat Risiko / Fluktuasi):** {opt_volatility:.2%}  
            > Semakin tinggi angkanya, semakin besar kemungkinan nilai portofolio Anda naik turun.  
            > Volatilitas mewakili ketidakpastian atau risiko pergerakan harga.
            """)


    # Feedback
    st.markdown("---")
    st.markdown("💬 Beri komentar atau masukan aplikasi [di sini](https://tinyurl.com/FeedbackAppERPO-1234)")

else:
    st.info("Silakan upload file Excel terlebih dahulu.")
