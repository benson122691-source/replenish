# ReplenishAI

需求預測與補貨決策助理：預測未來 28 天銷量，換算成「現在該補多少貨」，並用**總成本（持有＋缺貨）**而不只是預測誤差來比較方法。

> **目前狀態（請先讀）**：所有結果來自**合成資料**（M5 格式，由 `replenishai/synthetic.py` 產生）。
> 合成資料內建星期與節日效應，模型容易學到，所以數字**不代表真實零售表現**，不能當成果引用。
> 真實 M5 尚未跑；成本參數（持有 25%/年、缺貨 0.4×單價）也是假設值。

## 做了什麼
- **資料管線**：M5 格式載入；所有 lag／滾動特徵至少位移 28 天，並有測試防止資料洩漏。
- **策略比較**：seasonal naive、28 天移動平均、TSB（間歇需求）、LightGBM（Poisson）、LightGBM 分位數（`lightgbm_q`）、oracle（完美預測＝成本下界）。
- **補貨策略與模擬**：order-up-to `S = μ(L+R) + zσ√(L+R)`，用保留期真實需求逐日重播，計算持有與缺貨成本。
- **統計檢查**：3 個滾動原點；缺貨成本比例敏感度；以序列為單位的配對 bootstrap 95% 信賴區間。
- **服務**：FastAPI（`/health` `/series` `/forecast` `/replenish` `/compare`）＋單頁儀表板（`/`）。

## 執行
```bash
pip install -r requirements.txt
python scripts/run_experiment.py        # 沒有 data/raw 時自動產生合成資料；結果寫入 artifacts/ 與 docs/results.md
uvicorn replenishai.api:app             # 開 http://127.0.0.1:8000/  （API 文件在 /docs）
python -m pytest -q
```

### 用 Docker
```bash
docker compose up --build               # 開 http://127.0.0.1:8000/
```
- 建置時會在映像檔內跑 `scripts/run_experiment.py`（合成資料，約 15 秒）產生 `artifacts/`，執行時不訓練模型。
- 監聽環境變數 `PORT`（預設 8000），可直接用於 Railway 等平台；內建 `/health` 健康檢查。
- 映像檔只安裝 `requirements-runtime.txt`（固定版本）；測試工具留在 `requirements.txt`。

### 用真實 M5 重跑（建議的下一步）
1. 到 Kaggle 的 M5 Forecasting - Accuracy 頁面接受規則，下載 `calendar.csv`、`sell_prices.csv`、`sales_train_evaluation.csv`。
2. 放進 `data/raw/`，並刪掉 `data/raw/SYNTHETIC.txt`。
3. `python scripts/run_experiment.py --stores CA_1 --max-items 100`，結果會自動標示 `data_source: m5`。

## 合成資料上的初步觀察（僅供除錯，非結論）
- LightGBM 比 28 天移動平均低約 19.6% 總成本，bootstrap 95% 信賴區間 [−28.5, −9.5]%。
- `lightgbm_q`（直接用每日需求分位數當補貨輸入）低約 70%，但提高服務水準到 0.99 後差距仍在，原因未明，
  很可能與合成資料內建的需求尖峰有關；**必須在真實資料驗證**後才能下結論。
- TSB 比移動平均差：這份資料零值不多，不算間歇需求。

## 已知限制
- 真實 M5、爬蟲（外部特徵）、LLM 助理、雲端部署尚未完成。
- Dockerfile 的建置步驟已在乾淨的 Python 3.12 venv 中逐步模擬驗證（實驗、全部端點、健康檢查），但開發機沒有 Docker，**尚未實際 `docker build`**。
- 起始庫存、σ 的估計方式簡化；沒有訂購成本，在途量初始值為 0。
- 儀表板的 JavaScript 只驗證了後端端點，尚未在瀏覽器實際操作截圖。

文件：`docs/`（PRD、SRS、SAD、SDS、測試計畫、實驗結果）。
