# 06 測試計畫

版本：0.1（2026-11-08）｜「未執行」＝尚未跑過；證據欄只填真實結果。

| TC | 對應 | 情境 | 預期 | 實際 / 證據 |
|---|---|---|---|---|
| TC-1 | FR-1 | 合成資料檔符合 M5 欄位 | 必要欄位齊全 | 通過（pytest `test_synthetic_files_follow_m5_schema`，2026-11-08） |
| TC-2 | FR-1、BR-5 | 載入後長表無重複 (id,date)、銷量 ≥ 0、價格非空 | 條件成立 | 通過（`test_load_is_tidy`） |
| TC-3 | FR-2、NFR-2 | lag_28 等於 28 天前銷量 | 無洩漏 | 通過（`test_features_have_no_future_leak`） |
| TC-4 | FR-2 | 保留期正好 28 天，且訓練最後日 < 測試首日 | 條件成立 | 通過（`test_split_holds_out_exactly_horizon_days`） |
| TC-5 | FR-3 | 完美預測的誤差為 0 | RMSE=0、RMSSE=0 | 通過（`test_perfect_forecast_scores_zero`） |
| TC-6 | FR-3 | 基準預測涵蓋每個測試列且無缺值 | 列數相同、無 NaN | 通過（`test_baselines_cover_every_test_row`） |
| TC-7 | FR-3 | 無雜訊週期序列上 seasonal naive 完全正確 | 誤差為 0 | 通過（`test_seasonal_naive_is_exact_on_a_noise_free_weekly_series`） |
| TC-8 | FR-4、M-1 | LightGBM RMSSE 優於兩個基準 | 較低 | 通過（合成資料、3 個滾動原點平均）：RMSSE 0.678 < 移動平均 0.721 < seasonal naive 0.961；見 docs/results.md。真實 M5 待驗證 |
| TC-9 | FR-5 | 庫存充足時補貨量為 0 | 0 | 通過（`test_no_order_when_stock_is_enough`） |
| TC-10 | FR-5、AC-2.2 | 設定 MOQ，需補貨時補貨量 ≥ MOQ | 成立 | 通過（`test_order_respects_moq_and_is_integer`） |
| TC-11 | FR-5、BR-3 | 補貨量為整數、≥ 0 | 成立 | 通過（同上） |
| TC-12 | FR-6、M-2 | 模擬成本：模型 vs 基準 | 產出比較表 | 通過（合成資料）：LightGBM 總成本比移動平均低 19.6%；四種缺貨成本假設下排名不變；見 docs/results.md。真實 M5 待驗證 |
| TC-13 | FR-7 | 有效品項回 28 筆預測 | 200、28 筆 | 通過（`test_forecast_returns_28_days_from_artifacts`，2026-11-09；並比對值等於 artifacts/holdout_forecasts.csv） |
| TC-14 | FR-7、BR-1 | 不存在品項 | 404 | 通過（`test_forecast_unknown_item_or_store_is_404`：品項與門市各一例，訊息含欄位名） |
| TC-15 | FR-8、BR-4 | 服務水準 = 1.2 或庫存 = −5 | 422 | 通過（`test_replenish_rejects_invalid_input`：另含 SL=0.4、交期 0/29、MOQ=−1） |
| TC-16 | FR-8 | 正常補貨請求 | 200，含補貨點與補貨量 | 通過（`test_replenish_matches_policy_formula`：以 artifacts 獨立重算 ROP 與補貨量並比對；另有庫存充足=0、MOQ 測試） |
| TC-17 | FR-9、NFR-5 | /compare 含 data_source | 欄位存在 | 通過（`test_compare_returns_artifact_numbers_and_data_source`：每個數字等於 artifacts/results.json；`test_synthetic_results_are_labelled`） |
| TC-18 | FR-10 | 無痕視窗開頁、選品項、看到圖與建議、重新整理後正常 | 可操作 | 通過（2026-11-14）：`scripts/check_dashboard.py` 以 Playwright 驅動本機 Edge（無頭、全新 context，等同無痕）實際操作頁面，10/10 項通過：選單由 `/series` 載入、6 種方法與 bootstrap 表顯示、合成資料標示、改選品項與輸入後出現預測圖與補貨建議、交期 40 顯示欄位錯誤、重新整理後正常、390px 寬無橫向捲動、無 console 錯誤（修正前 favicon 404，已加內嵌 icon）。截圖：`docs/evidence/tc18-desktop.png`、`tc18-mobile.png`、`tc18-invalid-input.png` |
| TC-19 | NFR-1 | 同設定重跑兩次 | 指標一致 | 通過（`test_lgbm_is_reproducible`；僅測 LightGBM 預測） |
| TC-20 | NFR-3 | 單品項預測 API p95 | < 1 秒 | 部分：本機 uvicorn、同一品項連續 200 次 GET /forecast，p95 ≈ 5 ms（2026-11-09，手動量測，非自動測試）；雲端未測 |
| TC-21 | NFR-4 | 提交前搜尋 Repo 的金鑰與 .env | 無 | 通過（2026-11-10）：對所有追蹤檔搜尋 api_key/secret/token/password/ghp_/sk-/AKIA/私鑰標頭，無結果；無 .env；commit 作者信箱為 noreply@example.com。2026-11-14 複查：4 個 commit 的作者信箱曾是個人信箱並已推上 GitHub；已改寫這些 commit（只改作者/提交者，內容不變；已簽章的舊 commit 未動）並 force push，複查後全部為 noreply |
| TC-22 | NFR-6 | `docker compose up` 後可開頁面 | 成功 | 通過（2026-11-10，本機 Docker 29.9.2）：`docker compose build` 約 2 分鐘（映像 879MB），建置時跑實驗的成本數字與本機一致；`docker compose up` 後 `/`、`/series`、`/forecast`、`/compare`、`/replenish` 皆 200，錯誤輸入 422、未知品項 404，healthcheck 為 healthy；設 `PORT=9000` 時改聽 9000。頁面的瀏覽器操作見 TC-18。**注意**：2026-11-14 Dockerfile 改為固定版本 `requirements-runtime.txt`、非 root、HEALTHCHECK，修改後只在乾淨 venv 模擬建置步驟（通過），尚未重跑 `docker compose build`，部署前需重跑本案例 |

## 執行方式

```
python -m pytest -q
python scripts/git_monitor.py
```

## 已知限制

- 目前實驗資料為合成資料；使用真實 M5 後須重跑並更新本表。
- TC-8、TC-12 的結論取決於實測；目前只在合成資料上通過，不可當作真實零售成果。
- 另有補充測試：模擬守恆與成本非負、低估預測會增加缺貨、缺貨成本比例上升會提高總成本（tests/test_policy.py）。
