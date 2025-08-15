# Lab Video Cell Analysis Tools

## 主要應用程式

### integrated_cell_analysis_app.py
整合所有細胞分析功能的主要應用程式，提供完整的圖形界面進行細胞追蹤分析。

**功能特色:**
- 🎥 視頻文件載入與管理
- 📏 尺度校準 (Scale Calibration)
- ⚙️ 參數調整與優化
- 🎯 ROI 區域設定
- 🔬 細胞追蹤分析
- 📊 結果儀表板

**使用方法:**
```bash
python integrated_cell_analysis_app.py
```

## 輔助工具

### roi_tracking_dashboard_v2.py
獨立的儀表板應用程式，當整合應用程式的儀表板出現卡頓時使用。

**使用場景:**
- 主應用程式儀表板響應緩慢
- 需要更流暢的數據可視化體驗
- 專門用於結果分析和數據探索

**使用方法:**
```bash
python roi_tracking_dashboard_v2.py [settings_file.txt]
```

**Dashboard 操作說明:**

🔧 **基本控制:**
- **Use KF 勾選框**: 切換顯示原始數據或卡爾曼濾波後的數據 (一般建議使用原始數據，無需 KF 濾波)
- **KF Q/R 滑桿**: 調整卡爾曼濾波參數 (Q: 過程噪聲, R: 觀測噪聲) - 可忽略
- **Apply KF 按鈕**: 應用新的濾波參數 - 可忽略

📊 **視圖控制:**
- **X/Y Zoom 滑桿**: 調整圖表縮放比例
- **方向鍵按鈕**: 移動視圖位置 (←→↑↓)
- **Auto Scale**: 自動調整視圖範圍

🎯 **速度測量:**
1. 點擊圖表上的兩個點設定測量線
2. **Select**: 選擇測量點
3. **Delete**: 刪除選中的測量點
4. **Compute**: 計算兩點間的速度
5. **Apply Filter**: 應用時間範圍過濾器

💾 **數據匯出:**
- **Save**: 匯出當前過濾的數據為 CSV 文件

**操作流程:**
1. 載入設定檔案後會自動顯示三個 ROI 區域的細胞軌跡
2. **建議保持 KF 關閉，使用原始數據進行分析**
3. 調整視圖參數進行詳細分析
4. 點擊兩個位置進行速度測量
5. 匯出需要的分析結果

### multi_plot.py
用於批量分析和繪製多個實驗數據的工具。

**功能:**
- 批量處理多個速度分析檔案
- 生成頻率對速度的綜合分析圖表
- 統計分析與可視化

**使用方法:**
```bash
python multi_plot.py [folder_path]
```

## 建議的工作流程

1. **開始分析:** 使用 `integrated_cell_analysis_app.py` 進行完整的細胞分析流程
2. **性能優化:** 如果儀表板出現卡頓，切換至 `roi_tracking_dashboard_v2.py` 進行結果查看
3. **批量分析:** 使用 `multi_plot.py` 處理多組實驗數據並生成綜合報告

## 環境設置

### 方法一：使用 Conda 環境 (推薦)
```bash
# 創建環境
conda env create -f environment.yml

# 激活環境
conda activate poly
```

### 方法二：使用 pip 安裝
```bash
pip install opencv-python numpy pandas matplotlib pillow scipy filterpy scikit-learn seaborn trackpy
```

### 系統要求
- Python 3.9+
- macOS / Windows / Linux

## 核心依賴套件
- **opencv-python**: 影像處理
- **numpy**: 數值計算
- **pandas**: 數據處理
- **matplotlib**: 數據可視化
- **pillow**: 圖像操作
- **scipy**: 科學計算
- **filterpy**: 卡爾曼濾波 (Dashboard 功能)# piano_recital_0815
