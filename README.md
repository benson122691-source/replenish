# ReplenishAI — 需求預測與補貨決策助理

預測每個品項未來 28 天的銷量，把預測轉成「現在該補多少貨」，再用**持有成本＋缺貨成本**比較不同預測方法，用金額而不只是誤差來評估模型。

> ⚠ **目前所有數字都來自合成資料（synthetic）**。資料欄位與 M5（Walmart）格式相同，但數字是程式產生的，**不代表真實零售表現**。API 每個回應和頁面都會標示 `data_source`。換成真實 M5 資料的方法見[下方說明](#換成真實-m5-資料)。

## 解決什麼問題

中小型零售店的採購人員每週要決定上百個品項各補多少貨。常見做法是用 Excel 算移動平均，結果旺季缺貨、淡季積壓。而且「預測準不準」和「實際多花多少錢」沒有被放在一起看。

ReplenishAI 提供一頁看板：選擇門市與品項 → 看 28 天預測 → 輸入目前庫存、交期、服務水準、MOQ → 得到再訂購點與建議訂購量；同時列出不同預測方法的成本比較。

## 結果（合成資料，3 個滾動原點平均）

每個原點都只用原點以前的資料重新訓練，再預測接下來 28 天。完整表格見 [docs/results.md](docs/results.md)。

| 方法 | RMSSE ↓ | 總成本 ↓ | vs 28 天移動平均 | 滿足率 |
|---|---:|---:|---:|---:|
| 季節性 naive（上週同日） | 0.961 | 605.9 | +47.2% | 95.3% |
| 28 天移動平均（基準） | 0.721 | 411.7 | — | 97.0% |
| **LightGBM** | **0.678** | **331.1** | **−19.6%** | **97.7%** |
| 完美預測（成本下界） | 0 | 75.7 | −81.6% | 100% |

- 缺貨成本比例從 0.2 調到 1.6 倍單價時，LightGBM 都是成本最低的方法（不含完美預測），排名不變。
- 成本參數（持有成本年化 25%、缺貨成本 0.4× 單價）是**假設值**，M5 沒有真實成本資料。
- 所有方法共用相同的起始庫存、σ、服務水準與成本，差別只在預測。

### 對照 project-brief 的成功標準（Q5）

| 標準 | 狀態 |
|---|---|
| 1. 預測誤差優於季節性 naive | 合成資料上達成（0.678 vs 0.961）；真實 M5 未測 |
| 2. 總成本優於移動平均基準 | 合成資料上達成（−19.6%）；真實 M5 未測 |
| 3. API 本機與雲端可操作，p95 < 1 秒 | 本機與 Docker 可操作，本機 p95 ≈ 5 ms；雲端尚未部署 |
| 4. LLM 工具呼叫正確率 ≥ 80% | 未開始（LLM 助理尚未排入時程） |
| 5. Demo、Repo、測試紀錄 | Repo 與測試紀錄已有；Demo 網址或影片尚未完成 |

## 怎麼做的

```
data/raw/ (M5 格式 CSV)
   │  replenishai/data.py, features.py       讀檔、轉長表；lag/滾動特徵至少位移 28 天（防資料洩漏）
   ▼
scripts/run_experiment.py                    3 個滾動原點 × 4 種方法：預測 → 誤差 → 庫存模擬 → 成本
   │                                         （baselines.py, models.py, policy.py, simulate.py）
   ▼
artifacts/ (results.json, holdout_forecasts.csv, series_stats.csv)
   │
   ▼
replenishai/api.py (FastAPI)  ──  static/index.html（同一個服務提供，免 CORS）
```

- **模型**：LightGBM，Poisson 目標（銷量是非負計數）。基準是季節性 naive 和 28 天移動平均，另外用完美預測當成本下界。
- **補貨**：再訂購點 `ROP = μ·L + z·σ·√L`；庫存 ≤ ROP 時訂 `max(MOQ, ⌈ROP − 庫存⌉)`，否則不訂。
- **評估**：用保留期的真實需求逐日模擬庫存，計算持有與缺貨成本，並做缺貨成本比例的敏感度分析。
- **API 不在請求時訓練**：啟動時讀取 `artifacts/`，缺檔就拒絕啟動，並提示要先跑哪個腳本。

## 快速開始

需要 Python 3.12。

```bash
pip install -r requirements.txt
python scripts/run_experiment.py          # 沒有 M5 檔案時會自動產生合成資料；約 20 秒
uvicorn replenishai.api:app --reload      # 開 http://127.0.0.1:8000（看板）或 /docs（API 文件）
```

### Docker

```bash
docker compose up --build                 # 建置時會跑實驗；開 http://localhost:8000
```

容器會讀取環境變數 `PORT`（預設 8000），可以直接部署到 Railway 等平台，見 [docs/deploy-railway.md](docs/deploy-railway.md)。

### 測試

```bash
pytest -q                                 # 43 項；API 測試會用 artifacts 重算並比對 API 回傳的數字
```

## API

| 方法 | 路徑 | 用途 |
|---|---|---|
| GET | `/health` | 狀態、資料來源、預測原點、序列數 |
| GET | `/series` | 可選的門市與品項（填看板選單） |
| GET | `/forecast?item_id=&store_id=` | 單一品項 28 天 LightGBM 預測與實際值（回測） |
| POST | `/replenish` | 再訂購點、安全庫存、建議訂購量；交期限 1–28 天、服務水準 0.5–0.999，超出範圍回 422，查無品項回 404 |
| GET | `/compare` | 各方法的誤差、成本與敏感度分析 |

範例：

```bash
curl -X POST http://127.0.0.1:8000/replenish -H "Content-Type: application/json" \
  -d '{"item_id":"FOODS_1_001","store_id":"CA_1","on_hand":3,"lead_time_days":7,"service_level":0.95,"moq":0}'
```

## 換成真實 M5 資料

1. 從 Kaggle 的 [M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) 下載 `calendar.csv`、`sales_train_evaluation.csv`、`sell_prices.csv`（需登入並同意競賽規則；檔案很大，不放進 Repo）。
2. 刪除 `data/raw/` 裡的合成資料（包括 `SYNTHETIC.txt` 標記檔），放入上述三個檔案。
3. 重跑 `python scripts/run_experiment.py`，`data_source` 會變成 `m5`，`docs/results.md` 也會重新產生。

真實 M5 有約 3 萬條序列，訓練時間與記憶體需求會比合成資料大很多，目前還沒在真實資料上跑過。

## 限制

- 結果只來自合成資料，真實零售表現未知。
- 成本參數是假設值；沒有考慮訂購成本、在途量（MVP 視為 0）與供應商限制。
- σ 取訓練期最後 56 天的日銷量標準差，屬簡化設定。
- `/forecast` 回傳的是最後 28 天的回測預測，不是真正的未來預測。
- 尚未雲端部署；LLM 查詢助理尚未實作。

## 文件

| 文件 | 內容 |
|---|---|
| [project-brief.md](project-brief.md) | 發想與成功標準 |
| [docs/02-prd.md](docs/02-prd.md) ・ [03-srs](docs/03-srs.md) ・ [04-sad](docs/04-sad.md) ・ [05-sds](docs/05-sds.md) | 需求、規格、架構、設計（需求編號可追溯到測試） |
| [docs/06-test-plan.md](docs/06-test-plan.md) | 測試案例與實際證據 |
| [docs/results.md](docs/results.md) | 實驗結果（由腳本自動產生） |
| [docs/dev-notes.md](docs/dev-notes.md) | 開發手記 |

進度：`python scripts/git_monitor.py` 會比對 `schedule.json` 與 git 紀錄。
