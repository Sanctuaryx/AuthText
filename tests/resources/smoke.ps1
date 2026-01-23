$ErrorActionPreference = "Stop"

$BASE = "http://127.0.0.1:8001"

# Lee MIN_WORDS y DEFAULT_MODEL desde la propia API
$MIN_WORDS = (Invoke-RestMethod "$BASE/health").min_words
$DEFAULT_MODEL = (Invoke-RestMethod "$BASE/models").default

# Genera textos corto/largo
$SHORT_TEXT = ("palabra " * ($MIN_WORDS - 1)).Trim()
$LONG_TEXT  = ("palabra " * $MIN_WORDS).Trim()

Write-Host "BASE=$BASE"
Write-Host "MIN_WORDS=$MIN_WORDS"
Write-Host "DEFAULT_MODEL=$DEFAULT_MODEL"

function Assert-Status([string]$Name, [int]$Expected, [scriptblock]$Call) {
  try {
    & $Call
    Write-Host "OK  $Name"
  } catch {
    $resp = $_.Exception.Response
    if ($null -ne $resp) {
      $code = [int]$resp.StatusCode
      if ($code -eq $Expected) {
        Write-Host "OK  $Name (got $code)"
        return
      }
      Write-Host "FAIL $Name (expected $Expected got $code)"
      throw
    } else {
      Write-Host "FAIL $Name (no HTTP response)"
      throw
    }
  }
}

# 1) /health 200
Assert-Status "/health" 200 { Invoke-WebRequest "$BASE/health" | Out-Null }

# 2) /models 200
Assert-Status "/models" 200 { Invoke-WebRequest "$BASE/models" | Out-Null }

# 3) texto vacío -> 400
Assert-Status "empty text -> 400" 400 {
  Invoke-WebRequest "$BASE/predict" -Method POST -ContentType "application/json" -Body '{"text":""}' | Out-Null
}

# 4) texto corto -> 200 indeterminado
Assert-Status "short text -> 200" 200 {
  Invoke-WebRequest "$BASE/predict" -Method POST -ContentType "application/json" -Body ("{`"text`":`"$SHORT_TEXT`"}") | Out-Null
}

# 5) modelo desconocido -> 400
Assert-Status "unknown model -> 400" 400 {
  Invoke-WebRequest "$BASE/predict" -Method POST -ContentType "application/json" -Body ("{`"text`":`"$LONG_TEXT`",`"model`":`"no_existe`"}") | Out-Null
}

# 6) default -> 200
Assert-Status "default model -> 200" 200 {
  Invoke-WebRequest "$BASE/predict" -Method POST -ContentType "application/json" -Body ("{`"text`":`"$LONG_TEXT`"}") | Out-Null
}

# 7) query bilstm_rand -> 200
Assert-Status "query bilstm_rand -> 200" 200 {
  Invoke-WebRequest "$BASE/predict?model=bilstm_rand" -Method POST -ContentType "application/json" -Body ("{`"text`":`"$LONG_TEXT`"}") | Out-Null
}

# 8) query bilstm_w2v -> 200
Assert-Status "query bilstm_w2v -> 200" 200 {
  Invoke-WebRequest "$BASE/predict?model=bilstm_w2v" -Method POST -ContentType "application/json" -Body ("{`"text`":`"$LONG_TEXT`"}") | Out-Null
}

# 9) query bert -> 200 (si el modelo está cargado; si no, devolverá 500 y lo mostramos)
try {
  Invoke-WebRequest "$BASE/predict?model=bert" -Method POST -ContentType "application/json" -Body ("{`"text`":`"$LONG_TEXT`"}") | Out-Null
  Write-Host "OK  query bert -> 200"
} catch {
  $resp = $_.Exception.Response
  if ($null -ne $resp) {
    $code = [int]$resp.StatusCode
    Write-Host "WARN query bert -> got $code (bert quizá no cargado)"
  } else {
    throw
  }
}

# 10) precedence body over query -> 200
Assert-Status "body over query -> 200" 200 {
  Invoke-WebRequest "$BASE/predict?model=bert" -Method POST -ContentType "application/json" -Body ("{`"text`":`"$LONG_TEXT`",`"model`":`"bilstm_rand`"}") | Out-Null
}

# 11) /predict/<model> quirk check -> 200
Assert-Status "/predict/bert -> 200" 200 {
  Invoke-WebRequest "$BASE/predict/bert" -Method POST -ContentType "application/json" -Body ("{`"text`":`"$LONG_TEXT`"}") | Out-Null
}

Write-Host "OK: smoke tests completados"
