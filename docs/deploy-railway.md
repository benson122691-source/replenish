# Railway 部署計畫（針對 ReplenishAI）

狀態：**計畫，尚未驗證**。API（M3）、頁面與 Dockerfile（M4）已寫好；Docker 映像已在本機建置並通過 TC-22（2026-11-10），雲端部署尚未執行。平台方案與費用請在建立資源前到官方頁面確認，不一定永久免費。

官方文件（以此為準，影片畫面可能已改版）：
- Quick Start（從 GitHub 部署）：https://docs.railway.com/quick-start
- FastAPI 部署指南：https://docs.railway.com/guides/fastapi
- Monorepo 指南：https://docs.railway.com/guides/monorepo

## 這個專題的建議做法

- **單一服務**：FastAPI 同時提供 API 與 `static/index.html`，不需要分開部署前端，也就不用處理 CORS。
- **不需要資料庫服務**：MVP 用檔案，不必新增 PostgreSQL（避免多餘費用與設定）。
- **展示用資料是合成資料**：真實 M5 檔案很大且來自 Kaggle，不放進 Repo。部署版在建置時產生合成資料並跑實驗，頁面與 API 都會標示 `data_source: synthetic`。
- **不需要任何密鑰**：MVP 沒有外部 API Key。若之後加入 LLM，金鑰只放 Railway Variables，不進 Repo。

## 步驟（對應指南 4.3）

| # | 步驟 | 驗收 | 狀態 |
|---|---|---|---|
| 1 | 本機可執行，pytest 全過；搜尋 Repo 確認沒有 `.env`、Token、密碼（TC-21） | 搜尋無結果 | 未執行 |
| 2 | 推到 GitHub（公開前再看一次檔案清單） | Repo 可開啟 | 未執行 |
| 3 | Railway 建立 Project → Deploy from GitHub repo | 服務建立 | 未執行 |
| 4 | 使用 Dockerfile 建置（M4 提供）；若不用 Docker，Start Command 為 `uvicorn replenishai.api:app --host 0.0.0.0 --port $PORT` | 建置成功 | 未執行 |
| 5 | Networking → Generate Domain | 取得網址 | 未執行 |
| 6 | 開 `/health` 與 `/docs` | 200 | 未執行 |
| 7 | 無痕視窗開首頁，選品項、看圖與補貨建議，重新整理後仍正常；截圖留證（TC-18） | 可操作 | 未執行 |

## 要在 M3、M4 補上的東西（部署前置）

- `GET /health` 回 200（M3）。
- 程式監聽環境變數 `PORT`（Dockerfile 的 CMD 使用 `${PORT:-8000}`）（M4，已寫入 Dockerfile）。
- 建置時執行 `python scripts/run_experiment.py` 以產生 `artifacts/`（因為 artifacts 被 .gitignore 排除）（M4，已寫入 Dockerfile）。
- 建置時間與記憶體：LightGBM 訓練約 20 秒（本機合成資料量），Railway 上實際數字待確認。

## 失敗時的備案（指南 4 節第 6 點）

部署受阻就上傳 2 分鐘操作影片與測試截圖，並在作品手記註明「未完成雲端部署」。不要把未完成寫成已完成。
