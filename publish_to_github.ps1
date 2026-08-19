param(
    [Parameter(Mandatory = $true)]
    [string]$GitHubUsername,

    [string]$RepositoryName = "nlp-analytics-agent",

    [ValidateSet("https", "ssh")]
    [string]$ConnectionType = "https"
)

$ErrorActionPreference = "Stop"


function Write-Step {
    param(
        [string]$Message
    )

    Write-Host ""
    Write-Host "============================================================" `
        -ForegroundColor DarkGray

    Write-Host $Message `
        -ForegroundColor Cyan

    Write-Host "============================================================" `
        -ForegroundColor DarkGray
}


function Test-CommandExists {
    param(
        [string]$CommandName
    )

    return $null -ne (
        Get-Command `
            -Name $CommandName `
            -ErrorAction SilentlyContinue
    )
}


function Invoke-Git {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & git @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw (
            "Git command failed: git " +
            ($Arguments -join " ")
        )
    }
}


Write-Step "Step 1: Check Git installation"

if (-not (Test-CommandExists -CommandName "git")) {
    throw (
        "Git is not installed or is not available in PATH. " +
        "Install Git for Windows and open a new PowerShell terminal."
    )
}

$gitVersion = git --version

Write-Host "Git detected:" -ForegroundColor Green
Write-Host $gitVersion


Write-Step "Step 2: Find the local repository"

try {
    $repositoryRoot = (
        git rev-parse --show-toplevel
    ).Trim()
}
catch {
    throw (
        "The current directory is not inside a Git repository. " +
        "Open PowerShell in D:\nlp-analytics-agent and retry."
    )
}

if (-not $repositoryRoot) {
    throw "Unable to determine the Git repository location."
}

Write-Host "Repository root:" -ForegroundColor Green
Write-Host $repositoryRoot

Set-Location $repositoryRoot


Write-Step "Step 3: Check repository status"

git status

if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect the Git repository."
}


Write-Step "Step 4: Display recent commits"

git log `
    --oneline `
    --decorate `
    -10

if ($LASTEXITCODE -ne 0) {
    throw "Unable to read the commit history."
}


Write-Step "Step 5: Check for uncommitted changes"

$porcelainStatus = git status --porcelain

if ($porcelainStatus) {
    Write-Warning (
        "The repository contains uncommitted changes."
    )

    Write-Host ""
    Write-Host $porcelainStatus

    $commitChoice = Read-Host (
        "Commit all current changes before publishing? Enter Y or N"
    )

    if (
        $commitChoice -in @(
            "Y",
            "y",
            "Yes",
            "yes"
        )
    ) {
        $commitMessage = Read-Host (
            "Enter the commit message"
        )

        if ([string]::IsNullOrWhiteSpace($sage)) {
            $commitMessage = (
                "Update NLP analytics agent project"
            )
        }

        Invoke-Git add .

        Invoke-Git commit `
            -m $commitMessage

        Write-Host (
            "Changes committed successfully."
        ) -ForegroundColor Green
    }
    else {
        Write-Warning (
            "Uncommitted files will not be included in the GitHub push."
        )
    }
}
else {
    Write-Host (
        "Working tree is clean."
    ) -ForegroundColor Green
}


Write-Step "Step 6: Build the GitHub repository URL"

if ($ConnectionType -eq "ssh") {
    $remoteUrl = (
        "git@github.com:" +
        $GitHubUsername +
        "/" +
        $RepositoryName +
        ".git"
    )
}
else {
    $remoteUrl = (
        "https://github.com/" +
        $GitHubUsername +
        "/" +
        $RepositoryName +
        ".git"
    )
}

Write-Host "Remote URL:" -ForegroundColor Green
Write-Host $remoteUrl


Write-Step "Step 7: Check the current remote"

$existingRemote = git remote get-url origin 2>$null

if ($LASTEXITCODE -eq 0 -and $existingRemote) {
    Write-Host "Existing origin:" -ForegroundColor Yellow
    Write-Host $existingRemote

    if ($existingRemote.Trim() -ne $remoteUrl) {
        $replaceChoice = Read-Host (
            "Replace the existing origin with the new URL? Enter Y or N"
        )

        if (
            $replaceChoice -notin @(
                "Y",
                "y",
                "Yes",
                "yes"
            )
        ) {
            Write-Host (
                "The remote was not changed."
            ) -ForegroundColor Yellow

            exit 1
        }

        Invoke-Git remote `
            set-url `
            origin `
            $remoteUrl

        Write-Host (
            "The origin remote was updated."
        ) -ForegroundColor Green
    }
    else {
        Write-Host (
            "The correct origin remote is already configured."
        ) -ForegroundColor Green
    }
}
else {
    Invoke-Git remote `
        add `
        origin `
        $remoteUrl

    Write-Host (
        "The origin remote was added."
    ) -ForegroundColor Green
}


Write-Step "Step 8: Rename the branch to main"

$currentBranch = (
    git branch --show-current
).Trim()

Write-Host "Current branch:" -ForegroundColor Green
Write-Host $currentBranch

if ($currentBranch -ne "main") {
    Invoke-Git branch `
        -M `
        main

    Write-Host (
        "The branch was renamed to main."
    ) -ForegroundColor Green
}
else {
    Write-Host (
        "The repository is already using the main branch."
    ) -ForegroundColor Green
}


Write-Step "Step 9: Verify the configured remote"

git remote -v

if ($LASTEXITCODE -ne 0) {
    throw "Unable to display the configured remote."
}


Write-Step "Step 10: Push the project to GitHub"

Write-Host (
    "Make sure the empty GitHub repository already exists:"
) -ForegroundColor Yellow

Write-Host (
    "https://github.com/" +
    $GitHubUsername +
    "/" +
    $RepositoryName
)

Write-Host ""

$pushChoice = Read-Host (
    "Push the project now? Enter Y or N"
)

if (
    $pushChoice -notin @(
        "Y",
        "y",
        "Yes",
        "yes"
    )
) {
    Write-Host (
        "Push cancelled. Run the following command later:"
    ) -ForegroundColor Yellow

    Write-Host "git push -u origin main"

    exit 0
}

Invoke-Git push `
    -u `
    origin `
    main


Write-Step "GitHub publication completed"

Write-Host "Local repository:" -ForegroundColor Green
Write-Host $repositoryRoot

Write-Host ""

Write-Host "GitHub repository:" -ForegroundColor Green
Write-Host (
    "https://github.com/" +
    $GitHubUsername +
    "/" +
    $RepositoryName
)

Write-Host ""

Write-Host "Configured remotes:" -ForegroundColor Green
git remote -v

Write-Host ""

Write-Host "Current branch:" -ForegroundColor Green
git branch --show-current

Write-Host ""

Write-Host "Final repository status:" -ForegroundColor Green
git status
