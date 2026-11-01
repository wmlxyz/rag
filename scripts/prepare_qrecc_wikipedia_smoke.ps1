param([string]$OutputDirectory = 'data/qrecc/wikipedia-smoke')

$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Output directory exists; choose a new path.' }
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$docs = Join-Path $OutputDirectory 'documents'
New-Item -ItemType Directory -Path $docs | Out-Null
$titles = @('William Dampier', 'Affirmed', 'Banana')
$manifest = @()
foreach ($title in $titles) {
    $encoded = [Uri]::EscapeDataString($title.Replace(' ', '_'))
    $headers = @{ 'User-Agent' = 'ConversationalRAGResearch/0.1 contact research' }
    $revUri = "https://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=ids%7Ctimestamp&titles=$encoded&format=json&formatversion=2"
    $extractUri = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles=$encoded&format=json&formatversion=2"
    $before = (Invoke-RestMethod -Uri $revUri -Headers $headers).query.pages[0]
    $page = (Invoke-RestMethod -Uri $extractUri -Headers $headers).query.pages[0]
    $after = (Invoke-RestMethod -Uri $revUri -Headers $headers).query.pages[0]
    if (-not $page.extract -or -not $before.revisions[0].revid -or $before.revisions[0].revid -ne $after.revisions[0].revid) { throw "Missing extract or page changed during download: $title" }
    $filename = $title.Replace(' ', '_') + '.txt'
    [IO.File]::WriteAllText((Join-Path (Resolve-Path $docs).Path $filename), $page.extract, [Text.UTF8Encoding]::new($false))
    $manifest += @{ source = $filename; title = $page.title; page_id = $page.pageid; revision_id = $before.revisions[0].revid; revision_timestamp = $before.revisions[0].timestamp; url = "https://en.wikipedia.org/wiki/$($title.Replace(' ', '_'))"; fetched_utc = [DateTime]::UtcNow.ToString('o') }
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'sources.json') -Encoding utf8
@{
    'William_Dampier.txt' = @('https://en.wikipedia.org/wiki/William_Dampier', 'https://en.wikipedia.org/wiki/William_Dampier#First_circumnavigation')
    'Affirmed.txt' = @('https://en.wikipedia.org/wiki/Affirmed', 'https://en.wikipedia.org/wiki/Affirmed#1977:_two-year-old_season')
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'mapping.json') -Encoding utf8
Write-Output 'Downloaded 2 reference pages and 1 distractor. This is a restricted candidate pool, not the QReCC collection.'
