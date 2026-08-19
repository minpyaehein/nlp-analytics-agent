param(
    [string]$PrimaryModel = "qwen3:4b",
    [string]$FallbackModel = "qwen3:0.6b",
    [int]$MaxAttempts = 10,
    [int]$RetryDelaySeconds = 20
)

$ErrorActionPreference = "Continue"

function Write-Step {
    param(
        [string]$Message
    )

    Write-Host ""
    Write-Host "============================================================" `
        -ForegroundColor DarkGray
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "============================================================" `
        -ForegroundColor DarkGray
}

function Test-CommandExists {
    param(
        [string]$CommandName
    )

    return $null -ne (
        Get-Command $CommandName -ErrorAction SilentlyContinue
    )
}

function Test-InternetEndpoint {
    param(
        [string]$ComputerName,
        [int]$Port = 443
    )

    try {
        $testResult = Test-NetConnection `
            -ComputerName $ComputerName `
            -Port $Port `
            -InformationLevel Quiet `
            -WarningAction SilentlyContinue

        return [bool]$testResult
    }
    catch {
        return $false
    }
}

function Get-InstalledOllamaModels {
    try {
        $output = ollama list 2>&1

        if ($LASTEXITCODE -ne 0) {
            return @()
        }

        return $output
    }
    catch {
        return @()
    }
}

function Test-OllamaModelInstalled {
    param(
        [string]$ModelName
    )

    $modelOutput = Get-InstalledOllamaModels

    if (-not $modelOutput) {
        return $false
    }

    $baseModelName = $ModelName.Split(":")[0]

    return [bool](
        $modelOutput |
        Select-String `
           pe($baseModelName) `
            -Quiet
    )
}

function Pull-OllamaModel {
    param(
        [string]$ModelName,
        [int]$Attempts,
        [int]$DelaySeconds
    )

    for (
        $attempt = 1;
        $attempt -le $Attempts;
        $attempt++
    ) {
        Write-Host ""
        Write-Host (
            "Downloading {0}. Attempt {1} of {2}." -f `
                $ModelName,
                $attempt,
                $Attempts
        ) -ForegroundColor Yellow

        ollama pull $ModelName

        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host (
                "Model download completed successfully: {0}" -f `
                    $ModelName
            ) -ForegroundColor Green

            return $true
        }

        Write-Host ""
        Write-Warning (
            "Download attempt {0} failed for {1}." -f `
                $attempt,
                $ModelName
        )

        if ($attempt -lt $Attempts) {
            Write-Host (
                "Partial model files will be preserved. " +
                "Retrying in $DelaySeconds seconds."
            ) -ForegroundColor DarkYellow

            Start-Sleep -Seconds $DelaySeconds
        }
    }

    return $false
}

function Test-OllamaModel {
    param(
        [string]$ModelName
    )

    Write-Host ""
    Write-Host (
        "Testing local model: {0}" -f $ModelName
    ) -ForegroundColor Cyan

    $testPrompt = @"
Return only valid JSON.

Convert this analytics request into JSON:

Show the top 5 products by revenue.

Required fields:
intent
metric
dimension
sort_direction
limit
visualization
"@

    try {
        $testPrompt | ollama run $ModelName

        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "Local model test completed." `
                -ForegroundColor Green

            return $true
        }

        Write-Warning "The local model test did not complete."
        return $false
    }
    catch {
        Write-Warning (
            "Unable to test the model: {0}" -f `
                $_.Exception.Message
        )

        return $false
    }
}


Write-Step "Step 1: Check Ollama installation"

if (-not (Test-CommandExists -CommandName "ollama")) {
    Write-Error (
        "Ollama was not found. Install Ollama for Windows, " +
        "close PowerShell, and open a new terminal."
    )

    exit 1
}

$ollamaVersion = ollama --version

Write-Host "Ollama detected:" -ForegroundColor Green
Write-Host $ollamaVersion


Write-Step "Step 2: Check available disk space"

Get-PSDrive -PSProvider FileSystem |
    Select-Object `
        Name,
        @{
            Name = "UsedGB"
            Expression = {
                [math]::Round(
             $_.Used / 1GB,
                    2
                )
            }
        },
        @{
            Name = "FreeGB"
            Expression = {
                [math]::Round(
             $_.Free / 1GB,
                    2
                )
            }
        } |
    Format-Table -AutoSize


Write-Step "Step 3: Check network endpoints"

$registryAvailable = Test-InternetEndpoint `
    -ComputerName "registry.ollama.ai" `
    -Port 443

$storageAvailable = Test-InternetEndpoint `
    -ComputerName (
        "dd20bb891979d25aebc8bec07b2b3bbc." +
        "r2.cloudflarestorage.com"
    ) `
    -Port 443

Write-Host (
    "Ollama registry available: {0}" -f `
        $registryAvailable
)

Write-Host (
    "Cloudflare model storage available: {0}" -f `
        $storageAvailable
)

if (-not $registryAvailable) {
    Write-Warning (
        "The Ollama registry cannot be reached. " +
        "Check the internet connection, DNS, proxy, or firewall."
    )
}

if (-not $storageAvailable) {
    Write-Warning (
        "Cloudflare model storage cannot be reached. " +
        "Try another network or a mobile hotspot."
    )
}


Write-Step "Step 4: Check whether the primary model is installed"

if (
    Test-OllamaModelInstalled `
        -ModelName $PrimaryModel
) {
    Write-Host (
        "The primary model is already installed: {0}" -f `
            $PrimaryModel
    ) -ForegroundColor Green

    Test-OllamaModel -ModelName $PrimaryModel

    exit 0
}


Write-Step "Step 5: Download the primary model"

$primarySuccess = Pull-OllamaModel `
    -ModelName $PrimaryModel `
    -Attempts $MaxAttempts `
    -DelaySeconds $RetryDelaySeconds

if ($primarySuccess) {
    Write-Step "Step 6: Test the primary model"

    Test-OllamaModel -ModelName $PrimaryModel

    Write-Step "Ollama model setup completed"

    Write-Host (
        "Primary model ready: {0}" -f `
            $PrimaryModel
    ) -ForegroundColor Green

    exit 0
}


Write-Step "Primary model download failed"

Write-Warning (
    "The primary model could not be downloaded after " +
    "$MaxAttempts attempts."
)

$fallbackChoice = Read-Host (
    "Try the smaller fallback model " +
    "$FallbackModel? Enter Y or N"
)

if (
    $fallbackChoice -notin @(
        "Y",
        "y",
        "Yes",
        "yes"
    )
) {
    Write-Host ""
    Write-Host (
        "Model setup stopped. You can retry later with:"
    ) -ForegroundColor Yellow

    Write-Host (
        "ollama pull {0}" -f $PrimaryModel
    )

    exit 1
}


Write-Step "Step 7: Download the fallback model"

$fallbackSuccess = Pull-OllamaModel `
    -ModelName $FallbackModel `
    -Attempts $MaxAttempts `
    -DelaySeconds $RetryDelaySeconds

if (-not $fallbackSuccess) {
    Write-Error (
        "Both model downloads failed. " +
        "Try another network or a mobile hotspot."
    )

    exit 1
}


Write-Step "Step 8: Test the fallback model"

Test-OllamaModel -ModelName $FallbackModel


Write-Step "Ollama fallback setup completed"

Write-Host (
    "Fallback model ready: {0}" -f `
        $FallbackModel
) -ForegroundColor Green

Write-Host ""
Write-Host (
    "When creating local_llm.py, use this setting:"
) -ForegroundColor Cyan

Write-Host (
    'MODEL_NAME = "' + $FallbackModel + '"'
)