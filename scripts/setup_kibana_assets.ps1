param(
    [string]$KibanaUrl = "http://localhost:5601",
    [string]$DataViewName = "bag-count-events*",
    [string]$DashboardTitle = "Bag Count Monitoring Dashboard"
)

$ErrorActionPreference = "Stop"

$headers = @{
    "kbn-xsrf" = "true"
    "Content-Type" = "application/json"
}

Write-Host "Creating Kibana data view..."
$dvPayload = @{
    data_view = @{
        name = $DataViewName
        title = $DataViewName
        timeFieldName = "@timestamp"
    }
} | ConvertTo-Json -Depth 10

Invoke-RestMethod -Method Post -Uri "$KibanaUrl/api/data_views/data_view" -Headers $headers -Body $dvPayload | Out-Null

Write-Host "Creating empty dashboard shell..."
$dashboardPayload = @{
    attributes = @{
        title = $DashboardTitle
        description = "Starter dashboard for bag counting events routed through Logstash."
        timeRestore = $false
        optionsJSON = '{"useMargins":true,"syncColors":false,"hidePanelTitles":false}'
        panelsJSON = "[]"
        version = 1
        kibanaSavedObjectMeta = @{
            searchSourceJSON = '{"query":{"query":"","language":"kuery"},"filter":[]}'
        }
    }
} | ConvertTo-Json -Depth 10

$dashboardResponse = Invoke-RestMethod -Method Post -Uri "$KibanaUrl/api/saved_objects/dashboard" -Headers $headers -Body $dashboardPayload

Write-Host "Data view and dashboard created successfully."
Write-Host "Open Kibana at $KibanaUrl"
Write-Host "Dashboard ID: $($dashboardResponse.id)"
