![範例 GIF](img/clip0615.gif)

[View English Documentation (查看英文文档)](./README_en.md)

# 手勢控制機械臂模擬器

本專案允許您使用網路攝影機偵測到的手勢來控制模擬的六軸機械臂。它具有多種控制模式並提供即時視覺回饋。

## 功能特色

*   **即時手勢控制：** 使用手勢控制六軸機械臂。
*   **多種控制模式：**
    *   **相對連續模式：** 透過上下移動左手來調整所選關節的角度。
    *   **絕對模式：** 將您左手在定義控制區域內的垂直位置直接映射到所選關節的角度。
    *   **離散模式：** 使用特定的左手手勢（握拳/豎起大拇指表示增加，食指伸出表示減少，張開手掌表示停止）來改變所選關節的角度。
*   **動態操作指南：** 螢幕上的指示會根據所選的控制模式自動調整。
*   **視覺回饋：** 即時查看您的手部標記點和機械臂的運動。
*   **互動式滑桿：** 使用 Matplotlib 滑桿手動控制每個關節。
*   **模組化程式碼：** 專案結構清晰，易於閱讀和維護。

## 先決條件

*   Python 3.7+ (建議初學者使用 Anaconda 發行版)
*   連接到電腦的網路攝影機。

## 安裝說明 (通用 Python / pip)

這些說明適用於熟悉標準 Python 環境和 `pip` 的使用者。如果您是新手或偏好使用 Anaconda，請參閱下一節。

1.  **下載或複製專案：**
    *   **複製 (如果您已安裝 git)：**
        ```bash
        git clone <repository_url> 
        cd Robot_ARM_simulator
        ```
        (請將 `<repository_url>` 替換為實際的儲存庫網址)
    *   **下載 ZIP：** 從儲存庫頁面下載專案檔案的 ZIP 壓縮包，然後將其解壓縮到您電腦上的資料夾中。

2.  **建立虛擬環境 (建議)：**
    在專案目錄中開啟您的終端機或命令提示字元，然後執行：
    ```bash
    python -m venv venv
    ```
    啟動虛擬環境：
    *   Windows (PowerShell/CMD)：
        ```powershell
        .\venv\Scripts\Activate.ps1 
        ```
        或
        ```cmd
        venv\Scripts\activate.bat
        ```
    *   macOS/Linux：
        ```bash
        source venv/bin/activate
        ```

3.  **安裝依賴套件：**
    啟動虛擬環境後，使用 `requirements.txt` 檔案安裝所需的 Python 套件：
    ```bash
    pip install -r requirements.txt
    ```
    這將安裝 `numpy`、`matplotlib`、`mediapipe` 和 `opencv-python`。

## Anaconda 使用者替代設定 (Spyder IDE)

本指南適用於初學者或偏好使用 Anaconda 和 Spyder IDE 的使用者。

1.  **下載並安裝 Anaconda：**
    *   前往 [Anaconda 發行版頁面](https://www.anaconda.com/products/distribution)。
    *   下載適用於您作業系統 (Windows、macOS 或 Linux) 的安裝程式。
    *   執行安裝程式並按照螢幕上的指示操作。通常可以接受預設設定。

2.  **下載專案檔案：**
    *   前往本專案的 GitHub 儲存庫頁面。
    *   點擊 "Code" 按鈕，然後選擇 "Download ZIP"。
    *   將 ZIP 檔案儲存到您的電腦，然後將其解壓縮到一個已知位置 (例如 `文件/Robot_ARM_simulator`)。

3.  **建立 Conda 環境：**
    *   開啟 **Anaconda Prompt** (在您系統的應用程式中搜尋它)。
    *   為本專案建立一個新的環境。這樣可以將其套件與其他 Python 專案分開。我們將其命名為 `robot_arm_env`：
        ```bash
        conda create -n robot_arm_env python=3.9 
        ```
        (您可以選擇 Python 版本，如 3.8、3.9 或 3.10。使用 3.9 是一個安全的選擇。)
        當提示是否繼續時，請按 `y` 並按 Enter。
    *   啟動新環境：
        ```bash
        conda activate robot_arm_env
        ```
        您應該會在提示字元行的開頭看到 `(robot_arm_env)`。

4.  **在 Conda 環境中安裝依賴套件：**
    *   在 Anaconda Prompt 中，導覽至專案目錄 (您解壓縮檔案的位置)。例如，如果您將其解壓縮到 `文件/Robot_ARM_simulator`：
        ```bash
        cd path/to/your/Robot_ARM_simulator 
        ```
        (請將 `path/to/your/` 替換為實際路徑，例如 `cd C:/Users/您的使用者名稱/Documents/Robot_ARM_simulator`)
    *   使用 `pip` (在您的 conda 環境中可用) 和 `requirements.txt` 檔案安裝所需的套件：
        ```bash
        pip install -r requirements.txt
        ```
        這會將 `numpy`、`matplotlib`、`mediapipe` 和 `opencv-python` 安裝到您的 `robot_arm_env` 環境中。

5.  **啟動 Spyder 並開啟專案：**
    *   在同一個 Anaconda Prompt 中 (已啟動 `robot_arm_env` 環境)，輸入：
        ```bash
        spyder
        ```
        這將啟動 Spyder IDE。第一次啟動可能需要一些時間。
    *   在 Spyder 中，前往 "File" > "Open..." 並導覽至 `Robot_ARM_simulator` 資料夾。選擇 `main.py` 檔案並點擊 "Open"。
    *   您也可以使用 Spyder 的 "Projects" 功能："Projects" > "New Project..." > "Existing Directory"，然後選擇您的 `Robot_ARM_simulator` 資料夾。

6.  **在 Spyder 中執行模擬器：**
    *   在 Spyder 中開啟並啟用 `main.py` 後，點擊工具列中的綠色 "Run file" 按鈕 (看起來像一個播放圖示)，或按 `F5`。
    *   應該會出現兩個視窗：一個是 Matplotlib 視窗，顯示 3D 機械臂；另一個是 OpenCV 視窗，顯示您的網路攝影機畫面和手勢指示。

## 使用方法

1.  **啟動應用程式：**
    *   **如果使用通用 Python/pip：** 在終端機中導覽至專案目錄 (已啟動虛擬環境)，然後執行：
        ```bash
        python main.py
        ```
    *   **如果使用 Anaconda/Spyder：** 如上所述，在 Spyder 中執行 `main.py`。

2.  **理解視窗：**
    *   **機械臂視窗 (Matplotlib)：** 顯示六軸機械臂的 3D 視覺化效果。您可以使用底部的滑桿手動控制每個關節。
    *   **攝影機畫面與指示視窗 (OpenCV)：** 顯示您的網路攝影機畫面。手勢偵測在此進行。目前控制模式的指示會顯示在這裡。

3.  **控制方式：**
    *   **攝影機：** 應用程式將使用您的預設網路攝影機。請確保它沒有被遮擋且光線充足。
    *   **右手 (選擇關節)：** 您的右手用於選擇要控制的機械臂關節。
        *   **手勢 '1' (食指伸出)：** 控制關節 1
        *   **手勢 '2' (食指和中指伸出 - V 字手勢)：** 控制關節 2
        *   **手勢 '3' (食指、中指和無名指伸出)：** 控制關節 3
        *   **手勢 '4' (食指、中指、無名指和小指伸出)：** 控制關節 4
        *   **手勢 '5' (張開手掌 - 五指伸出)：** 控制關節 5
        *   **手勢 '0' (握拳)：** 控制關節 6
    *   **左手 (調整角度)：** 您的左手用於根據目前的控制模式調整*已選擇*關節的角度。
        *   **控制點：** 左手的 `MIDDLE_FINGER_MCP` (中指的指掌關節) 被用作主要的控制點。

4.  **切換控制模式：**
    *   在鍵盤上按 **'m'** 鍵 (當 OpenCV 視窗處於活動狀態時) 以循環切換不同的控制模式：
        1.  `relative_continuous` (相對連續)
        2.  `absolute` (絕對)
        3.  `discrete` (離散)
    *   目前的模式和該模式的特定指示將顯示在 OpenCV 視窗上。

5.  **控制模式詳情：**

    *   **`relative_continuous` (相對連續) 模式：**
        *   **增加角度：** 將左手的 `MIDDLE_FINGER_MCP` 向上移動。
        *   **減少角度：** 將左手的 `MIDDLE_FINGER_MCP` 向下移動。
        *   可以在 `main.py` 中調整移動的靈敏度和步長 (請參閱「修改程式碼」部分)。

    *   **`absolute` (絕對) 模式：**
        *   定義了一個垂直的活動區域 (為了減少畫面混亂，在最近版本中預設不顯式繪製)。
        *   將左手的 `MIDDLE_FINGER_MCP` 在垂直方向上定位。攝影機畫面的頂部對應一個角度限制，底部對應另一個角度限制。
        *   您的手的垂直位置直接映射到關節的角度。

    *   **`discrete` (離散) 模式：**
        *   **增加角度：** 用左手做出**握拳**或**豎起大拇指**的手勢。
        *   **減少角度：** 用左手伸出**食指** (像 '1' 的手勢)。
        *   **停止改變角度：** 用左手張開**手掌**。
        *   角度以預先定義的步長變化 (可在 `main.py` 中調整)。

6.  **退出應用程式：**
    *   在鍵盤上按 **'q'** 鍵 (當 OpenCV 視窗處於活動狀態時) 以關閉應用程式。
    *   或者，關閉 Matplotlib 視窗也會終止程式。

## 理解與修改程式碼 (初學者適用)

本專案分为數個 Python 檔案。以下是簡單的說明：

*   **`main.py`：** 這是應用程式的主要「大腦」。
    *   它設定機械臂、攝影機和顯示視窗。
    *   它包含主迴圈，不斷獲取手勢、更新機械臂並在螢幕上顯示所有內容。
    *   **`main.py` 中適合初學者調整的參數：**
        *   `CONTROL_MODE`：在腳本頂部，您可以更改預設的啟動模式。選項有 `"absolute"`、`"relative_continuous"` 或 `"discrete"`。
        *   `OPENCV_WINDOW_INITIAL_WIDTH`, `OPENCV_WINDOW_INITIAL_HEIGHT`: OpenCV 視窗的初始寬度和高度。
        *   `SENSITIVITY_THRESHOLD` (適用於 `relative_continuous` 模式)：您的手需要移動多少 (像素) 才被視為一次變化。增加此值可降低靈敏度，減少則提高靈敏度。
        *   `ANGLE_STEP_CONTINUOUS` (適用於 `relative_continuous` 模式)：對於給定的手部移動，角度變化多少。較小的值意味著更精細的控制。
        *   `DISCRETE_ANGLE_STEP` (適用於 `discrete` 模式)：每次「增加」或「減少」手勢使角度變化多少度。
        *   `dh_params`：這些參數定義了機械臂的物理尺寸。更改這些參數將改變機械臂的形狀和可及範圍。(進階)
        *   `init_angles`：六個關節各自的初始角度 (以度為單位)。

*   **`gesture_api.py`：** 此檔案處理所有手部追蹤和手勢辨識。
    *   它使用 `mediapipe` 函式庫從網路攝影機影像中找到您的手及其標記點 (指關節、手掌等)。
    *   然後，它解釋您手的位置和形狀，以判斷您正在做出哪個手勢 (例如「握拳」、「張開手掌」、「食指伸出」)。
    *   它還處理不同控制模式下左手的位置。
    *   它將手部標記點和操作指南文字繪製到攝影機畫面上。

*   **`kinematics.py`：** 此檔案包含描述機械臂結構及其關節如何移動的數學公式 (Denavit-Hartenberg 參數)。
    *   `get_joint_positions()` 函數根據目前的角度計算每個關節的 3D 座標。這對於繪製機械臂至關重要。

*   **`visualization.py`：** 此檔案負責在 Matplotlib 視窗中繪製 3D 機械臂。
    *   它接收由 `kinematics.py` 計算出的關節位置，並繪製圓柱體作為連桿，繪製球體/軸作為關節。
    *   它還設定用於手動關節控制的滑桿。

*   **`requirements.txt`：** 這是一個簡單的文字檔案，列出專案執行所需的外部 Python 套件。`pip install -r requirements.txt` 命令使用此檔案來安裝它們。

## 檔案結構 (概覽)

```
Robot_ARM_simulator/
├── main.py                 # 主要應用程式腳本
├── gesture_api.py          # 處理手勢辨識和攝影機畫面
├── kinematics.py           # 機械臂運動學計算
├── visualization.py        # 使用 Matplotlib 的 3D 機械臂視覺化
├── requirements.txt        # 專案依賴套件
└── README.md               # 本檔案 (說明文件)
```

## 疑難排解

*   **未偵測到網路攝影機 / "Error: Cannot open camera" (錯誤：無法開啟攝影機)：**
    *   確保您的網路攝影機已正確連接並在系統設定中啟用。
    *   檢查是否有其他應用程式正在使用網路攝影機。如果是，請關閉它。
    *   在 `main.py` 中，如果您有多個攝影機或者預設 (0) 無法運作，請嘗試將 `cap = cv2.VideoCapture(0)` 更改為 `cap = cv2.VideoCapture(1)` (或其他數字，如 2、3)。
*   **效能緩慢 / 影像延遲：**
    *   確保您的電腦符合執行即時影像處理的合理規格。
    *   關閉其他耗用大量資源的應用程式。
    *   如果可能，嘗試降低網路攝影機的解析度 (儘管本專案目前在 `gesture_api.py` 中將影像大小調整為固定尺寸)。
*   **手勢辨識不正確：**
    *   確保光線良好且一致。避免非常暗的房間或強烈的背光。
    *   保持雙手在攝影機畫面中清晰可見，不要太近或太遠。
    *   避免可能干擾手部偵測的雜亂背景。
    *   清晰明確地做出手勢。
*   **Anaconda 環境問題：**
    *   在執行 `pip install -r requirements.txt` 或啟動 `spyder` *之前*，請確保您已在 Anaconda Prompt 中*啟動*正確的 conda 環境 (`conda activate robot_arm_env`)。
    *   如果 Spyder 似乎使用了錯誤的 Python 解譯器，您可能需要進行設定。在 Spyder 中，前往 "Tools" > "Preferences" > "Python interpreter"，然後選擇 "Use the following Python interpreter"，接著將其指向您 conda 環境中的 Python 執行檔 (例如 `C:\Users\您的使用者名稱\anaconda3\envs\robot_arm_env\python.exe`)。

---

本 README 文件旨在為各級使用者提供全面的指南。祝您操作機械臂愉快！
