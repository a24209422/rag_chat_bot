#Requires -Version 5.1
<#
  一鍵把這個 RAG 聊天機器人跑起來。

  它不是新的啟動方式，只是把 README 裡那幾道手動指令綁在一起。每個服務開在
  自己的視窗，日誌看得見——出事的時候要知道是哪一個在叫，藏起來反而難查。

      .\start.ps1            後端 + 前端（雲端 Gemini 可用）
      .\start.ps1 -Local     再加上 llama.cpp server，地端推論才跑得動
      .\start.ps1 -Stop      把上面那些通通關掉
#>
[CmdletBinding()]
param(
    # 地端推論要 llama.cpp。只用雲端的話不開，省一塊顯卡記憶體
    [switch]$Local,
    [switch]$Stop,
    [switch]$NoBrowser,
    # 臨時換別的模型試，不用去改 .env
    [string]$LlamaExe,
    [string]$LlamaModel
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

# 埠號集中在這裡：settings.py 的 api_url、CORS 的預設白名單、README 的每一道
# 指令都指著這三個號碼，散著放的話改一個就得改一圈。
$PORT_API = 8000
$PORT_WEB = 5173
$PORT_LLM = 8080

# 這兩個不是偏好，是硬需求：
#   -c 8192  k=5 撈五個完整職缺約 3700 token，-c 4096 沒空間留給生成，server 直接回 400
#   -ngl 99  全部丟上顯卡；少了它落在 CPU 上，慢到示範不能看
$LLAMA_CTX = 8192
$LLAMA_NGL = 99


function Get-Listener {
    param([int]$Port)
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -First 1
    if (-not $conn) { return $null }
    Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
}

function Read-DotEnv {
    # 只給這支腳本用。Python 那側有 pydantic-settings，不靠這個。
    $path = Join-Path $root '.env'
    $map  = @{}
    if (-not (Test-Path $path)) { return $map }
    foreach ($line in (Get-Content $path -Encoding UTF8)) {
        $t = $line.Trim()
        if (-not $t -or $t.StartsWith('#')) { continue }
        $i = $t.IndexOf('=')
        if ($i -lt 1) { continue }
        $map[$t.Substring(0, $i).Trim()] = $t.Substring($i + 1).Trim().Trim('"')
    }
    return $map
}

function Start-Pane {
    # 路徑用單引號包起來：這個專案的目錄名有空白，讓 PowerShell 自己拆會拆錯。
    param([string]$Title, [string]$Exe, [string[]]$ArgList, [string]$WorkDir)
    $quoted = ($ArgList | ForEach-Object { "'" + ($_ -replace "'", "''") + "'" }) -join ' '
    $inner  = "`$Host.UI.RawUI.WindowTitle = '$Title'; Set-Location '$WorkDir'; & '$Exe' $quoted"
    Start-Process powershell -ArgumentList '-NoExit', '-NoProfile', '-Command', $inner | Out-Null
}

function Wait-For {
    param([scriptblock]$Probe, [int]$TimeoutSec, [string]$What)
    $sw = [Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
        $r = & $Probe
        if ($r) {
            Write-Host ("  {0} 起來了（{1:N1} 秒）" -f $What, $sw.Elapsed.TotalSeconds) -ForegroundColor Green
            return $r
        }
        Start-Sleep -Milliseconds 700
    }
    Write-Host ("  {0} 等了 {1} 秒還沒起來——去它自己的視窗看錯誤訊息" -f $What, $TimeoutSec) -ForegroundColor Yellow
    return $null
}


# ── -Stop：關掉 ──────────────────────────────────────────────────────
if ($Stop) {
    Write-Host "`n關閉服務" -ForegroundColor Cyan
    foreach ($svc in @(
        @{ Port = $PORT_WEB; Name = '前端 vite   ' },
        @{ Port = $PORT_API; Name = '後端 API    ' },
        @{ Port = $PORT_LLM; Name = 'llama-server' }
    )) {
        $p = Get-Listener $svc.Port
        if ($p) {
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
            Write-Host ("  {0} :{1} 已關閉（PID {2}）" -f $svc.Name, $svc.Port, $p.Id) -ForegroundColor Green
        } else {
            Write-Host ("  {0} :{1} 沒在跑" -f $svc.Name, $svc.Port) -ForegroundColor DarkGray
        }
    }
    # 手動用 npm run dev 起的話會多一層殼，殺了 vite 它會變孤兒。不主動殺——
    # 別的專案也可能正在跑 npm run dev，認錯人的代價比留著大，只報給人看。
    $shells = Get-CimInstance Win32_Process -Filter "Name = 'node.exe'" -ErrorAction SilentlyContinue |
              Where-Object { $_.CommandLine -like '*npm-cli.js*' -and $_.CommandLine -like '*dev*' }
    foreach ($s in $shells) {
        Write-Host ("  ! 還有一個 npm 殼沒關（PID {0}）：{1}" -f $s.ProcessId, $s.CommandLine) -ForegroundColor Yellow
    }
    Write-Host ""
    return
}


# ── 出發前檢查 ──────────────────────────────────────────────────────
Write-Host "`nRAG 聊天機器人" -ForegroundColor Cyan
Write-Host "檢查" -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "PATH 上找不到 python。"
}

$jobs = Join-Path $root 'data\jobs.json'
if (-not (Test-Path $jobs)) {
    throw "沒有 data\jobs.json，知識庫是空的。先跑：python -m tools.build_jobs ""<職缺 PDF 資料夾>"""
}
Write-Host "  data\jobs.json 在" -ForegroundColor Green

$modules = Join-Path $root 'frontend\node_modules'
if (-not (Test-Path $modules)) {
    Write-Host "  frontend\node_modules 不在，先跑 npm install（第一次會久一點）" -ForegroundColor Yellow
    Push-Location (Join-Path $root 'frontend')
    try { & npm install } finally { Pop-Location }
}
Write-Host "  frontend\node_modules 在" -ForegroundColor Green

$dotenv = Read-DotEnv
if (-not $dotenv['GEMINI_API_KEY']) {
    Write-Host "  .env 沒有 GEMINI_API_KEY——雲端那邊會回 503，地端不受影響" -ForegroundColor Yellow
}


# ── llama.cpp（只有 -Local 才開）─────────────────────────────────────
if ($Local) {
    Write-Host "`n地端模型" -ForegroundColor Cyan
    if (-not $LlamaExe)   { $LlamaExe   = $dotenv['LLAMA_EXE'] }
    if (-not $LlamaModel) { $LlamaModel = $dotenv['LLAMA_MODEL'] }
    $device = $dotenv['LLAMA_DEVICE']

    if (-not $LlamaExe -or -not $LlamaModel) {
        throw ("不知道 llama.cpp 在哪。在 .env 補這兩行（或用 -LlamaExe / -LlamaModel 傳）：`n" +
               "    LLAMA_EXE=D:\llama.cpp\llama-server.exe`n" +
               "    LLAMA_MODEL=D:\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf")
    }
    foreach ($pair in @{ LLAMA_EXE = $LlamaExe; LLAMA_MODEL = $LlamaModel }.GetEnumerator()) {
        if (-not (Test-Path $pair.Value)) { throw ("{0} 指到的檔案不存在：{1}" -f $pair.Key, $pair.Value) }
    }

    if (Get-Listener $PORT_LLM) {
        Write-Host "  :$PORT_LLM 已經有人在聽，沿用現有的 server" -ForegroundColor DarkGray
    } else {
        $a = @('-m', $LlamaModel, '--port', "$PORT_LLM", '-c', "$LLAMA_CTX", '-ngl', "$LLAMA_NGL")
        # --device 不給就讓 llama.cpp 自己選，而它可能把模型拆一半到內顯上，會慢很多。
        # 該填哪個編號用 llama-server.exe --list-devices 查。
        if ($device) { $a += @('--device', $device) }
        Start-Pane 'llama.cpp :8080' $LlamaExe $a (Split-Path $LlamaExe)
        Wait-For { Get-Listener $PORT_LLM } 120 'llama-server' | Out-Null
    }
}


# ── 後端 ────────────────────────────────────────────────────────────
Write-Host "`n後端" -ForegroundColor Cyan
if (Get-Listener $PORT_API) {
    Write-Host "  :$PORT_API 已經有人在聽，沿用現有的後端" -ForegroundColor DarkGray
} else {
    # 預熱地端做的是把 e5 載進記憶體（實測第一個請求 47.6 秒 → 2.6 秒），
    # 不是重算索引——索引本來就在 data\store_onperm.npz 裡。
    # 雲端不預熱：它沒有本機模型要載，第一個請求本來就快。
    if ($Local) { $env:API_WARM = 'onperm' } else { $env:API_WARM = '' }
    # 不加 --reload：它會多一個 reloader 子行程，關的時候殺了父的還留著子的。
    # 這支腳本是拿來「跑起來看」的；要改程式碼自己開 uvicorn --reload。
    Start-Pane 'RAG 後端 :8000' 'python' @('-m', 'uvicorn', 'api.main:app', '--host', '127.0.0.1', '--port', "$PORT_API") $root
    $timeout = if ($Local) { 150 } else { 60 }   # 預熱 onperm 要載 470MB 的 e5
    $health = Wait-For { try { Invoke-RestMethod "http://127.0.0.1:$PORT_API/health" -TimeoutSec 3 } catch { $null } } $timeout '後端 API'
    if ($health) {
        if ($health.loaded) { $loaded = $health.loaded -join '、' } else { $loaded = '無（延遲到第一次請求）' }
        Write-Host ("    索引已載入：{0}" -f $loaded) -ForegroundColor DarkGray
    }
}


# ── 前端 ────────────────────────────────────────────────────────────
Write-Host "`n前端" -ForegroundColor Cyan
if (Get-Listener $PORT_WEB) {
    Write-Host "  :$PORT_WEB 已經有人在聽，沿用現有的前端" -ForegroundColor DarkGray
} else {
    # 直接叫 vite，不走 npm run dev：npm 會多包一層殼，關的時候殺了 vite
    # 那層殼會留下來變孤兒。package.json 的 dev 本來就只是 "vite"。
    $vite = Join-Path $root 'frontend\node_modules\vite\bin\vite.js'
    Start-Pane '前端 vite :5173' 'node' @($vite, '--port', "$PORT_WEB", '--strictPort') (Join-Path $root 'frontend')
    Wait-For { Get-Listener $PORT_WEB } 60 '前端 vite' | Out-Null
}


# ── 收尾 ────────────────────────────────────────────────────────────
Write-Host "`n開好了" -ForegroundColor Cyan
Write-Host "  前端      http://localhost:$PORT_WEB"
Write-Host "  API 文件  http://localhost:$PORT_API/docs"
if ($Local) {
    Write-Host "  llama.cpp http://localhost:$PORT_LLM"
} else {
    Write-Host "  側欄的「地端」這時候按下去會回 503——要地端就用 .\start.ps1 -Local" -ForegroundColor DarkGray
}
Write-Host "`n  關掉：.\start.ps1 -Stop`n" -ForegroundColor DarkGray

if (-not $NoBrowser) { Start-Process "http://localhost:$PORT_WEB" }
