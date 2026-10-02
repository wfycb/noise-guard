# 커밋 전 검사: black --check → ruff → pytest(기본) → pytest -m slow
# 하나라도 실패하면 즉시 멈추고 그 단계의 종료 코드로 끝난다.
# 사용: powershell -NoProfile -ExecutionPolicy Bypass -File tools\check.ps1
#
# 각 단계의 성공 여부는 $LASTEXITCODE로만 판단한다. 출력을 파이프로 넘기면 종료 코드가
# 가려질 수 있어서, 출력은 그대로 콘솔에 두고 파이프를 쓰지 않는다.
# (PowerShell 5.1에서 ErrorActionPreference=Stop이면 네이티브 명령의 stderr가 예외가 되므로 Continue로 둔다.)

$ErrorActionPreference = "Continue"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
# 시나리오 테스트가 한국어를 출력하므로 콘솔 코드페이지와 무관하게 UTF-8로 쓴다.
$env:PYTHONIOENCODING = "utf-8"

$steps = @(
    @{ Name = "black --check"; Arguments = @("-m", "black", "--check", "-q", ".") },
    @{ Name = "ruff"; Arguments = @("-m", "ruff", "check", ".") },
    @{ Name = "pytest (기본)"; Arguments = @("-m", "pytest", "-q", "-p", "no:cacheprovider") },
    @{ Name = "pytest -m slow"; Arguments = @("-m", "pytest", "-m", "slow", "-q", "-p", "no:cacheprovider") }
)

Push-Location $projectRoot
try {
    foreach ($step in $steps) {
        Write-Host "===== $($step.Name) ====="
        $arguments = $step.Arguments
        & $python @arguments
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            Write-Host "===== 실패: $($step.Name) (종료 코드 $exitCode) ====="
            exit $exitCode
        }
        Write-Host "===== 통과: $($step.Name) ====="
    }
    Write-Host "===== 전체 통과 ====="
    exit 0
}
finally {
    Pop-Location
}
