# API Documentation

## Financial Risk Governance BI Platform API

### Base URL
```
http://localhost:8000/api
```

### Authentication
Currently, the API is open for development. In production, JWT authentication will be required.

## Business Intelligence Endpoints

### GET /api/bi/dashboard-data
Get main dashboard data and KPIs.

**Response:**
```json
{
  "kpis": {
    "total_portfolio_value": 1500000000,
    "risk_weighted_assets": 1200000000,
    "tier1_capital_ratio": 0.125,
    "liquidity_coverage_ratio": 1.15
  },
  "alerts": [
    {
      "level": "warning",
      "message": "Credit concentration limit approaching"
    }
  ]
}
```

### GET /api/bi/reports
Get list of available reports.

### GET /api/bi/charts/{chart_type}
Get data for specific chart types.

**Parameters:**
- `chart_type`: Type of chart (e.g., "risk_trend")

## Risk Assessment Endpoints

### GET /api/risk/credit-risk/portfolio
Get credit risk metrics for the portfolio.

**Response:**
```json
{
  "total_exposure": 2500000000,
  "weighted_average_pd": 0.045,
  "expected_loss": 112500000,
  "var_95": 180000000,
  "concentration_risk": {
    "top_10_exposures": 0.35,
    "hhi_index": 0.08
  }
}
```

### GET /api/risk/market-risk/var
Get Value at Risk calculations.

### GET /api/risk/operational-risk/indicators
Get Key Risk Indicators for operational risk.

### GET /api/risk/liquidity-risk/ratios
Get liquidity risk ratios and metrics.

## Predictive Modeling Endpoints

### GET /api/models/models/status
Get status of all deployed models.

### POST /api/models/models/{model_id}/predict
Make predictions using specified model.

**Parameters:**
- `model_id`: ID of the model to use

**Request Body:**
```json
{
  "features": {
    "credit_score": 720,
    "debt_to_income": 0.35,
    "loan_amount": 250000
  }
}
```

### GET /api/models/forecasts/risk-metrics
Get forecasted risk metrics.

### GET /api/models/scenarios/{scenario_name}
Run scenario analysis for risk assessment.

## Governance Endpoints

### GET /api/governance/compliance-status
Get overall compliance status across regulations.

### GET /api/governance/regulatory-reports
Get list of regulatory reports and their status.

### GET /api/governance/risk-appetite
Get risk appetite framework metrics and limits.

### GET /api/governance/audit-findings
Get recent audit findings and their remediation status.

### GET /api/governance/model-governance
Get model governance and validation status.

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request
```json
{
  "detail": "Invalid request parameters"
}
```

### 404 Not Found
```json
{
  "detail": "Resource not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error"
}
```

## Rate Limiting

- 100 requests per minute per IP address
- Burst limit of 20 requests

## Health Check

### GET /health
Get application health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-09-29T12:00:00",
  "services": {
    "database": "connected",
    "cache": "connected",
    "ml_models": "loaded"
  }
}
```